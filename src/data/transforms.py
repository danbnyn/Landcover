import jax
import jax.numpy as jnp
from functools import partial
from jax import lax
import numpy as np
from jaxtyping import Array, Float, Int, PyTree, Key
from jax.image import resize
import grain.python as grain

@partial(jax.jit )
def _ToJax(
        arr: Float[Array, "..."]
) -> Float[Array, "..."]:

    return arr

@partial(jax.jit )
def _MinMaxScale(
    arr: Float[Array, "C H W"]
) -> Float[Array, "C H W"]:
    """
    Applies min-max scaling to each 2D slice of the input 4D array across the last two dimensions (H, W).
    The scaling is performed independently for each slice, ensuring that the values are scaled between 0 and 1.

    Parameters:
        arr: Input array with shape (C, H, W), where:
             - C is the number of channels
             - H and W are the height and width of the image, respectively.

    Returns:
        Scaled array with the same shape as the input.
    """

    def scale_arr(
            inner_arr: Float[Array, "H W"]
    ) -> Float[Array, "H W"]:
        """
        Scales a 2D array (H, W) using min-max scaling.

        Parameters:
            inner_arr: 2D input array with shape (H, W).

        Returns:
            Scaled 2D array with values between 0 and 1.
        """
        array = inner_arr.astype(jnp.float32)  # Convert to float32 for precision
        min_value = jnp.min(array)  # Find the minimum value in the array
        max_value = jnp.max(array)  # Find the maximum value in the array
        return (array - min_value) / (max_value - min_value + 1e-7)  # Scale the array to [0, 1], with epsilon for stability

    # Apply the scaling function across the channel dim
    return jax.vmap(scale_arr, in_axes=0)(arr)

@partial(jax.jit, static_argnums=(1,))
def _BinaryEncode(
        arr: Int[Array, "H W"],
        num_classes: Int[Array, ""]
) -> Int[Array, "C H W"]:
    """
    Encodes the input integer array into its binary representation across a specified number of classes.
    Each element in the input array is converted to its binary equivalent.

    Parameters:
        arr: Input array with shape (H, W)
        num_classes: The number of classes to encode. This determines the number of bits used.

    Returns:
        Binary encoded array with shape (C, H, W), where C is the number of binary classes.
    """

    # Compute the number of binary digits required to represent the given number of classes
    num_bin_classes = np.ceil(np.log2(num_classes + 1)).astype(int)

    # Create a range of bit positions for binary encoding
    bit_range = jnp.arange(num_bin_classes, dtype=jnp.int32)

    def encode_single(
            x: Int[Array, ""]
    ) -> Int[Array, "C"]:
        """
        Encodes a single integer into its binary representation.

        Parameters:
            x: Scalar integer value to encode.

        Returns:
            A 1D array of binary digits representing the integer.
        """
        return jnp.right_shift(jnp.int32(x), bit_range) & 1  # Perform bitwise right shift and AND operation

    # Flatten the input array to a 1D array for easier processing
    original_shape = arr.shape
    flattened = arr.reshape(-1)

    # Apply the binary encoding to each element in the flattened array
    encoded = jax.vmap(encode_single)(flattened)

    # Reshape the encoded array back to its original shape with an additional dimension for the binary bits
    new_shape = original_shape + (num_bin_classes,)
    return encoded.reshape(new_shape)


@partial(jax.jit, static_argnums=(1,), )
def _OneHotEncode(
        arr: Int[Array, "H W"],
        num_classes: int
) -> Int[Array, "C H W"]:
    """
    One-hot encodes the input array across the specified number of classes.
    Each element in the input array is converted to a one-hot encoded vector.

    Parameters:
        arr: Input array with shape (H, W)
        num_classes: The number of classes to encode.

    Returns:
        One-hot encoded array with shape (C, H, W), where C is the number of classes.
    """

    # Determine the position of the new axis for one-hot encoding
    axis = arr.ndim  # This should be 2 ( H, W)

    # Expand the input array by adding an axis for the classes
    lhs = lax.expand_dims(arr, (axis,))

    # Create a shape for the RHS array (used for comparison) with one-hot encoded vectors
    rhs_shape = [1] * arr.ndim  # Create a shape of all ones, (H, W) -> [1, 1, 1]
    rhs_shape.insert(axis, num_classes)  # Insert the number of classes at the appropriate position, [1, 1, C]

    # Generate an array with incrementing values along the specified axis
    rhs = lax.broadcasted_iota(arr.dtype, rhs_shape, axis)

    # Perform the comparison to generate the one-hot encoded array
    one_hot_encoded = jnp.asarray(lhs == rhs, dtype=jnp.int32)  # Ensure the result is an integer array

    # Transpose to get the output shape (C, H, W)
    return jnp.transpose(one_hot_encoded, (2, 0, 1))


def create_mapping_array(
        classes_to_ignore: Int[Array, "M"],
        classes: Int[Array, "N"]
) -> Int[Array, "N-M+1"]:
    """
    Creates a mapping array that remaps class indices, setting ignored classes to 0.

    Args:
        classes_to_ignore (Array): Array of class indices to be ignored.
        classes (Array): Array of all class indices.

    Returns:
        Array: Mapping array where ignored classes are set to 0, others remain as their indices.
    """
    classes_to_ignore_array = jnp.array(classes_to_ignore)
    classes_array = jnp.array(classes)

    # Create a boolean mask to identify which classes should be ignored
    ignore_mask = jnp.isin(classes_array, classes_to_ignore_array)

    # Create an array of indices for all classes
    indices = jnp.arange(0, len(classes))

    # Set the indices of ignored classes to 0
    mapping = jnp.where(ignore_mask, 0, indices)

    return mapping.astype(jnp.uint8)


@partial(jax.jit, )
def _RemapMasks(
        batch: Int[Array, "H W"],
        classes_to_background: Int[Array, "M"],
        original_classes: Int[Array, "K"]
) -> Int[Array, "H W"]:
    """
    Remaps class labels in a batch of masks according to a mapping array, setting background classes to 0.

    Args:
        batch (Array): Batch of mask images with class indices.
        classes_to_background (Array): Array of class indices to be remapped to background.
        original_classes (Array): Array of all original class indices.

    Returns:
        Array: Batch of masks with remapped class indices.
    """
    # Create a mapping array to remap the class labels
    mapping_array = create_mapping_array(classes_to_background, original_classes)

    # Apply the mapping to the batch of masks
    return mapping_array[batch]  # JAX automatically broadcasts the mapping array to the shape of the batch



@partial(jax.jit, )
def _RandomFlip(
    sample_batch: PyTree[Float[Array, "..."]],
    key: Key[Array, ''],
    p: Float[Array, '']
) -> PyTree[Float[Array, "..."]]:
    """
    Randomly flips images and masks vertically and/or horizontally with probability `p`.

    Args:
        sample_batch (PyTree): Dictionary containing the batch of images and masks.
            - The image should have shape (N, C, H, W).
            - The mask should have shape (N, H, W).
        key (Key): Random key for generating random numbers.
        p (float): Probability of flipping the image and mask.

    Returns:
        PyTree: Dictionary containing the flipped images and masks.
    """
    # Extract the keys for the image and mask in the batch
    sample_key = list(sample_batch.keys())
    image = sample_batch[sample_key[0]]
    mask = sample_batch[sample_key[1]]

    # Split the random key for two independent flip operations
    v_flip_key, h_flip_key = jax.random.split(key, num=2)

    # Generate random numbers to decide if each flip should be applied
    rand_vertical = jax.random.uniform(v_flip_key)
    rand_horizontal = jax.random.uniform(h_flip_key)

    # Determine whether to flip based on the probability `p`
    flip_vertical = rand_vertical < p
    flip_horizontal = rand_horizontal < p

    # Conditionally flip the image and mask along the vertical axis
    image = jnp.where(flip_vertical, jnp.flip(image, axis=-2), image)
    mask = jnp.where(flip_vertical, jnp.flip(mask, axis=-2), mask)

    # Conditionally flip the image and mask along the horizontal axis
    image = jnp.where(flip_horizontal, jnp.flip(image, axis=-1), image)
    mask = jnp.where(flip_horizontal, jnp.flip(mask, axis=-1), mask)

    return {sample_key[0]: image, sample_key[1]: mask}

@partial(jax.jit, static_argnums=1, )
def _RandomRotate(
    sample_batch: PyTree[Float[Array, "..."]],  # Pytree annotation for the batch of images and masks
    rot_angle: Float[Array, ''],  # Maximum rotation angle in degrees
    p: Float[Array, ''],  # Probability of applying the rotation
    key: Key[Array, '']  # Random key for generating random numbers
) -> PyTree[Float[Array, "..."]]:
    """
    Randomly rotates images and masks by a specified angle with a given probability `p`.

    Args:
        sample_batch (PyTree): Dictionary containing the batch of images and masks.
            - The image should have shape (C, H, W).
            - The mask should have shape (H, W).
        rot_angle (float): Maximum rotation angle in degrees.
        p (float): Probability of applying the rotation.
        key (Key): Random key for generating random numbers.

    Returns:
        PyTree: Dictionary containing the rotated images and masks.
    """
    sample_key = list(sample_batch.keys())
    image = sample_batch[sample_key[0]]

    # Split the key for multiple random operations
    rot_key, p_key = jax.random.split(key, num=2)  # Consume the keys

    # Get the probability to apply the transform
    should_apply = jax.random.uniform(p_key) < p

    # Get original dimensions
    original_height, original_width = image.shape[-2:]

    # Assuming the image is square, otherwise use min(original_height, original_width)
    original_size = min(original_height, original_width)

    # Calculate the rotation angle in radians
    angle = jax.random.uniform(rot_key, minval=-rot_angle, maxval=rot_angle)
    angle_rad = jnp.radians(angle)

    rot_angle_rad = np.radians(rot_angle)

    # Calculate the size of the largest square that fits in the rotated image
    max_crop_size = np.round((original_size / (np.abs(np.sin(rot_angle_rad)) + np.abs(np.cos(rot_angle_rad))))).astype(int)

    def apply_transform(
            inner_sample_batch: PyTree[Float[Array, "..."]],
            inner_angle_rad: Float[Array, '']
    ) -> PyTree[Float[Array, "..."]]:
        # Unpack the sample batch
        inner_image = inner_sample_batch[sample_key[0]]
        inner_mask = inner_sample_batch[sample_key[1]]

        # Rotate image and mask
        rotated_image = jax.vmap(rotate_image, in_axes=(0, None, None), out_axes=0)(inner_image, inner_angle_rad, 0)
        rotated_mask = rotate_image(inner_mask, inner_angle_rad, 0)

        # Center crop the rotated image and mask to the maximum crop size
        cropped_image = center_crop(rotated_image, max_crop_size, max_crop_size)

        # Add a channel dimension to the mask between the batch and height dimensions
        rotated_mask = rotated_mask[jnp.newaxis, :, :]

        cropped_mask = center_crop(rotated_mask, max_crop_size, max_crop_size)

        # Remove the channel dimension from the mask
        cropped_mask = cropped_mask[0, :, :]

        # Get image channel size
        channel_size = inner_image.shape[0]

        # Resize back to the original dimensions
        inner_image = resize(cropped_image, [channel_size, original_height, original_width], method="nearest")
        inner_mask = resize(cropped_mask, [original_height, original_width], method="nearest")

        # Convert back to the original data types
        inner_image = inner_image.astype(sample_batch[sample_key[0]].dtype)
        inner_mask = inner_mask.astype(sample_batch[sample_key[1]].dtype)

        return {sample_key[0]: inner_image, sample_key[1]: inner_mask}

    def no_transform(
            inner_sample_batch: PyTree[Float[Array, "..."]],
            inner_angle_rad: Float[Array, '...']
    ) -> PyTree[Float[Array, "..."]]:
        # Return the original batch unchanged
        return inner_sample_batch

    # Conditionally apply the transform based on probability
    return jax.lax.cond(should_apply, apply_transform, no_transform, sample_batch, angle_rad)

def center_crop(
    image: Float[Array, "C H W"],
    height: Float[Array, ''],
    width: Float[Array, '']
) -> Float[Array, "C H W"]:
    """
    Crops the center of the input image to the specified height and width.

    Args:
        image (Array): Input image array with shape (C, H, W).
        height (int): Desired height of the cropped image.
        width (int): Desired width of the cropped image.

    Returns:
        Array: Cropped image with shape (C, height, width).
    """
    # Get the shape of the input image
    channel, current_height, current_width = image.shape

    # Calculate the center of the input image
    center_h, center_w = current_height // 2, current_width // 2

    # Calculate the top and left indices for cropping
    top = jnp.maximum(center_h - (height // 2), 0)
    left = jnp.maximum(center_w - (width // 2), 0)

    # Create the start indices for cropping
    start_indices = jnp.array([0, top, left])

    # Use dynamic slicing to crop the image
    cropped = jax.lax.dynamic_slice(
        image,
        start_indices,
        slice_sizes=(channel, height, width)
    )

    return cropped


def rotate_image(
    image: Float[Array, "H W"],
    angle: Float[Array, ''],
    order: int = 0
) -> Float[Array, "H W"]:
    """
    Rotate a 2D image by a given angle using JAX.

    Args:
        image (Array): A 2D JAX array representing the image with shape (H, W).
        angle (float): The rotation angle in radians.
        order (int): The interpolation mode to use. '0' for nearest neighbor, '1' for linear interpolation.

    Returns:
        Array: A 2D JAX array of the rotated image with shape (H, W).
    """
    # Get image dimensions
    height, width = image.shape

    # Calculate the image center
    center_y, center_x = (height - 1) / 2., (width - 1) / 2.

    # Create a grid of coordinates corresponding to the image pixels
    y, x = jnp.mgrid[:height, :width]

    # Shift the coordinates to be centered at (0, 0)
    y -= center_y
    x -= center_x

    # Apply rotation using the rotation matrix
    cos_theta, sin_theta = jnp.cos(angle), jnp.sin(angle)
    y_rot = y * cos_theta - x * sin_theta
    x_rot = y * sin_theta + x * cos_theta

    # Shift the rotated coordinates back to the original image center
    y_rot += center_y
    x_rot += center_x

    # Stack coordinates for map_coordinates
    coords = [y_rot, x_rot]

    # Interpolate the rotated image using map_coordinates
    rotated_image = jax.scipy.ndimage.map_coordinates(image, coords, order=order, mode='constant', cval=0)

    return rotated_image



# Some transforms
class MinMaxScale(grain.MapTransform):
    def map(self, sample_batch):
        image_key = list(sample_batch.keys())[0]
        sample_batch[image_key] = _MinMaxScale(sample_batch[image_key])
        return sample_batch


class BinaryEncode(grain.MapTransform):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def map(self, sample_batch):
        mask_key = list(sample_batch.keys())[-1]
        sample_batch[mask_key] = _BinaryEncode(sample_batch[mask_key], self.num_classes)
        return sample_batch


class OneHotEncode(grain.MapTransform):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def map(self, sample_batch):
        mask_key = list(sample_batch.keys())[-1]
        sample_batch[mask_key] = _OneHotEncode(sample_batch[mask_key], self.num_classes)
        return sample_batch


class RandomColorJitter(grain.RandomMapTransform):
    pass


class RemapMasks(grain.MapTransform):

    def __init__(self, original_classes, classes_to_background):
        self.original_classes = original_classes
        self.classes_to_background = classes_to_background
        self.remaining_classes = sorted(set(original_classes) - set(classes_to_background))

    def map(self, sample_batch):
        mask_key = list(sample_batch.keys())[-1]
        sample_batch[mask_key] = _RemapMasks(sample_batch[mask_key], self.classes_to_background, self.original_classes)
        return sample_batch

class RandomRotate(grain.MapTransform):
    def __init__(self, key, p, rot_angle):
        self.key = key
        self.p = p
        self.rot_angle = rot_angle

    def map(self, sample_batch):
        return _RandomRotate(sample_batch, self.rot_angle, self.p, self.key)

class RandomFlip(grain.MapTransform):
    def __init__(self, key, p):
        self.key = key
        self.p = p

    def map(self, sample_batch):
        return _RandomFlip(sample_batch, self.key, self.p)


class ToJax(grain.MapTransform):
    def map(self, sample_batch):
        image_key = list(sample_batch.keys())[0]
        mask_key = list(sample_batch.keys())[1]
        sample_batch[mask_key] = _ToJax(sample_batch[mask_key])
        sample_batch[image_key] = _ToJax(sample_batch[image_key])
        return sample_batch
