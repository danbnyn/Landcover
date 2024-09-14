from pickletools import uint8

import jax
import jax.numpy as jnp
from functools import partial
from jax import lax
import numpy as np
from jaxtyping import Array, Float, Int, PyTree, Key
from jax.image import resize
import grain.python as grain
from typing import Tuple

from tensorboardX.summary import image

@partial(jax.jit, static_argnums=2)
def _ClaheHistTransformBatched(
        batched_images: jnp.ndarray,
        clip_limit: float = 0.005,  # Clip limit now ranges from 0 to 1
        nbins: int = 4096,
        max_value: int = 65535
) -> jnp.ndarray:
    """
    Apply a simplified version of CLAHE (without tiling) to batched uint16 RGB satellite images.
    Clip limit now ranges from 0 to 1 as a fraction of total pixels in the image.

    Args:
    batched_images: Input batched images as a JAX array with shape (B, H, W, 3) and dtype uint16
    clip_limit: Fraction of pixels (0 to 1) as threshold for contrast limiting
    nbins: Number of histogram bins
    max_value: Maximum value in the input images (65535 for uint16)

    Returns:
    Equalized batched images as a JAX array with the same shape and dtype as the input
    """

    def equalize_image(image):
        def equalize_channel(channel):
            # Compute histogram for the entire channel
            hist, _ = jnp.histogram(channel, bins=nbins, range=(0, max_value))

            # Determine the clip limit in terms of pixel count
            total_pixels = channel.size
            pixel_clip_limit = clip_limit * total_pixels

            # Clip the histogram based on the calculated pixel clip limit
            clipped_hist = jnp.minimum(hist, pixel_clip_limit)

            # Compute cumulative distribution function (CDF) for the histogram
            cdf = jnp.cumsum(clipped_hist)
            cdf = cdf / cdf[-1]  # Normalize to [0, 1]

            # Apply the equalization
            equalized_channel = jnp.interp(channel, jnp.linspace(0, max_value, nbins), cdf * max_value)

            return equalized_channel

        # Apply the equalization to each channel independently
        equalized_channels = jax.vmap(equalize_channel)(image)

        return equalized_channels.astype(jnp.uint16)

    # Apply the equalization to each image in the batch
    return jax.vmap(equalize_image)(batched_images)

@partial(jax.jit)
def _CustomSatelliteImageScalerBatched(
        arr: Float[Array, "N C H W"],
        lower_percentile: float = 0,
        upper_percentile: float = 99.5,
        target_min: float = 0.0,
        target_max: float = 1.0
) -> Float[Array, "N C H W"]:
    """
    Applies a custom scaling to satellite images encoded in uint16.
    This method uses percentile clipping to handle skewed distributions and then applies linear scaling.

    Parameters:
        arr: Input array with shape (N, C, H, W), where:
             - N is the batch size
             - C is the number of channels
             - H and W are the height and width of the image, respectively.
        lower_percentile: Lower percentile for clipping (default: 2.0)
        upper_percentile: Upper percentile for clipping (default: 98.0)
        target_min: Minimum value of the target range (default: 0.0)
        target_max: Maximum value of the target range (default: 1.0)

    Returns:
        Scaled array with the same shape as the input, with values between target_min and target_max.
    """

    def scale_arr(
            inner_arr: Float[Array, "H W"]
    ) -> Float[Array, "H W"]:
        """
        Scales a 2D array (H, W) using percentile clipping and linear scaling.
        """
        array = inner_arr.astype(jnp.float32)  # Convert to float32 for precision

        # Calculate percentile values for clipping
        lower_bound = jnp.percentile(array, lower_percentile)
        upper_bound = jnp.percentile(array, upper_percentile)

        # Clip the array
        clipped_array = jnp.clip(array, lower_bound, upper_bound)

        # Apply linear scaling to the clipped array
        scaled = (clipped_array - lower_bound) / (upper_bound - lower_bound)

        # Scale to target range
        return scaled * (target_max - target_min) + target_min

    # Apply the scaling function across the first two dimensions (N, C) using jax.vmap
    return jax.vmap(jax.vmap(scale_arr, in_axes=0), in_axes=0)(arr)


@partial(jax.jit)
def _RobustScaleBatched(
        arr: Float[Array, "N C H W"]
) -> Float[Array, "N C H W"]:
    """
    Applies robust scaling to each 2D slice of the input 4D array across the last two dimensions (H, W).
    The scaling is performed independently for each slice, using the interquartile range (IQR) method.

    Parameters:
        arr: Input array with shape (N, C, H, W), where:
             - N is the batch size
             - C is the number of channels
             - H and W are the height and width of the image, respectively.

    Returns:
        Scaled array with the same shape as the input.
    """

    def scale_arr(
            inner_arr: Float[Array, "H W"]
    ) -> Float[Array, "H W"]:
        """
        Scales a 2D array (H, W) using robust scaling (IQR method).

        Parameters:
            inner_arr: 2D input array with shape (H, W).

        Returns:
            Scaled 2D array.
        """
        array = inner_arr.astype(jnp.float32)  # Convert to float32 for precision

        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        q1 = jnp.percentile(array, 25)
        q3 = jnp.percentile(array, 75)

        # Calculate IQR (Interquartile Range)
        iqr = q3 - q1

        # Calculate median
        median = jnp.median(array)

        # Apply robust scaling
        scaled = (array - median) / (iqr + 1e-7)  # Add small epsilon for numerical stability

        return scaled

    # Apply the scaling function across the first two dimensions (N, C) using jax.vmap
    return jax.vmap(jax.vmap(scale_arr, in_axes=0), in_axes=0)(arr)


@partial(jax.jit)
def _MinMaxScaleBatched(
    arr: Float[Array, "N C H W"]
) -> Float[Array, "N C H W"]:
    """
    Applies min-max scaling to each 2D slice of the input 4D array across the last two dimensions (H, W).
    The scaling is performed independently for each slice, ensuring that the values are scaled between 0 and 1.

    Parameters:
        arr: Input array with shape (N, C, H, W), where:
             - N is the batch size
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

    # Apply the scaling function across the first two dimensions (N, C) using jax.vmap
    return jax.vmap(jax.vmap(scale_arr, in_axes=0), in_axes=0)(arr)

@partial(jax.jit, static_argnums=(1,), backend='cpu')
def _BinaryEncodeBatched(
        arr: Int[Array, "N H W"],
        num_classes: Int[Array, ""]
) -> Int[Array, "N C H W"]:
    """
    Encodes the input integer array into its binary representation across a specified number of classes.
    Each element in the input array is converted to its binary equivalent.

    Parameters:
        arr: Input array with shape (N, H, W), where:
             - N is the batch size
             - H and W are the height and width of the image, respectively.
        num_classes: The number of classes to encode. This determines the number of bits used.

    Returns:
        Binary encoded array with shape (N, C, H, W), where C is the number of binary classes.
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


@partial(jax.jit, static_argnums=(1,))
def _OneHotEncodeBatched(
        arr: Int[Array, "N H W"],
        num_classes: int
) -> Int[Array, "N C H W"]:
    """
    One-hot encodes the input array across the specified number of classes.
    Each element in the input array is converted to a one-hot encoded vector.

    Parameters:
        arr: Input array with shape (N, H, W), where:
             - N is the batch size
             - H and W are the height and width of the image, respectively.
        num_classes: The number of classes to encode.

    Returns:
        One-hot encoded array with shape (N, C, H, W), where C is the number of classes.
    """

    # Determine the position of the new axis for one-hot encoding
    axis = arr.ndim  # This should be 3 (N, H, W)

    # Expand the input array by adding an axis for the classes
    lhs = lax.expand_dims(arr, (axis,))

    # Create a shape for the RHS array (used for comparison) with one-hot encoded vectors
    rhs_shape = [1] * arr.ndim  # Create a shape of all ones, (N, H, W) -> [1, 1, 1]
    rhs_shape.insert(axis, num_classes)  # Insert the number of classes at the appropriate position, [1, 1, 1, C]

    # Generate an array with incrementing values along the specified axis
    rhs = lax.broadcasted_iota(arr.dtype, rhs_shape, axis)

    # Perform the comparison to generate the one-hot encoded array
    one_hot_encoded = jnp.asarray(lhs == rhs, dtype=jnp.int32)  # Ensure the result is an integer array

    # Transpose to get the output shape (N, C, H, W)
    return jnp.transpose(one_hot_encoded, (0, 3, 1, 2))


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


@partial(jax.jit, backend='cpu')
def _RemapMasksBatched(
        batch: Int[Array, "N H W"],
        classes_to_background: Int[Array, "M"],
        original_classes: Int[Array, "K"]
) -> Int[Array, "N H W"]:
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



@partial(jax.jit)
def _RandomFlipBatched(
    sample_batch: PyTree[Float[Array, "N ..."]],
    key: Key[Array, ''],
    p: Float[Array, '']
) -> PyTree[Float[Array, "N ..."]]:
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

@partial(jax.jit, static_argnums=1)
def _RandomRotateBatched(
    sample_batch: PyTree[Float[Array, "N ..."]],  # Pytree annotation for the batch of images and masks
    rot_angle: Float[Array, ''],  # Maximum rotation angle in degrees
    p: Float[Array, ''],  # Probability of applying the rotation
    key: Key[Array, '']  # Random key for generating random numbers
) -> PyTree[Float[Array, "N ..."]]:
    """
    Randomly rotates images and masks by a specified angle with a given probability `p`.

    Args:
        sample_batch (PyTree): Dictionary containing the batch of images and masks.
            - The image should have shape (N, C, H, W).
            - The mask should have shape (N, H, W).
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
            inner_sample_batch: PyTree[Float[Array, "N ..."]],
            inner_angle_rad: Float[Array, '']
    ) -> PyTree[Float[Array, "N ..."]]:
        # Unpack the sample batch
        inner_image = inner_sample_batch[sample_key[0]]
        inner_mask = inner_sample_batch[sample_key[1]]

        # Rotate image and mask
        rotated_image = jax.vmap(jax.vmap(rotate_image, in_axes=(0, None, None), out_axes=0), in_axes=(0, None, None), out_axes=0)(inner_image, inner_angle_rad, 0)
        rotated_mask = jax.vmap(rotate_image, in_axes=(0, None, None), out_axes=0)(inner_mask, inner_angle_rad, 0)

        # Center crop the rotated image and mask to the maximum crop size
        cropped_image = center_crop(rotated_image, max_crop_size, max_crop_size)

        # Add a channel dimension to the mask between the batch and height dimensions
        rotated_mask = rotated_mask[:, jnp.newaxis, :, :]

        cropped_mask = center_crop(rotated_mask, max_crop_size, max_crop_size)

        # Remove the channel dimension from the mask
        cropped_mask = cropped_mask[:, 0, :, :]

        # Get batch size and image / mask channel size
        batch_size = inner_image.shape[0]
        channel_size = inner_image.shape[1]

        # Resize back to the original dimensions
        inner_image = resize(cropped_image, [batch_size, channel_size, original_height, original_width], method="nearest")
        inner_mask = resize(cropped_mask, [batch_size, original_height, original_width], method="nearest")

        # Convert back to the original data types
        inner_image = inner_image.astype(sample_batch[sample_key[0]].dtype)
        inner_mask = inner_mask.astype(sample_batch[sample_key[1]].dtype)

        return {sample_key[0]: inner_image, sample_key[1]: inner_mask}

    def no_transform(
            inner_sample_batch: PyTree[Float[Array, "N ..."]],
            inner_angle_rad: Float[Array, '...']
    ) -> PyTree[Float[Array, "N ..."]]:
        # Return the original batch unchanged
        return inner_sample_batch

    # Conditionally apply the transform based on probability
    return jax.lax.cond(should_apply, apply_transform, no_transform, sample_batch, angle_rad)

def center_crop(
    image: Float[Array, "B C H W"],
    height: Float[Array, ''],
    width: Float[Array, '']
) -> Float[Array, "B C H W"]:
    """
    Crops the center of the input image to the specified height and width.

    Args:
        image (Array): Input image array with shape (B, C, H, W).
        height (int): Desired height of the cropped image.
        width (int): Desired width of the cropped image.

    Returns:
        Array: Cropped image with shape (B, C, height, width).
    """
    # Get the shape of the input image
    batch, channel, current_height, current_width = image.shape

    # Calculate the center of the input image
    center_h, center_w = current_height // 2, current_width // 2

    # Calculate the top and left indices for cropping
    top = jnp.maximum(center_h - (height // 2), 0)
    left = jnp.maximum(center_w - (width // 2), 0)

    # Create the start indices for cropping
    start_indices = jnp.array([0, 0, top, left])

    # Use dynamic slicing to crop the image
    cropped = jax.lax.dynamic_slice(
        image,
        start_indices,
        slice_sizes=(batch, channel, height, width)
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



class CustomSatelliteImageScaler(grain.MapTransform):
    def map(self, sample_batch):
        image_key = list(sample_batch.keys())[0]
        sample_batch[image_key] = _CustomSatelliteImageScalerBatched(sample_batch[image_key])
        return sample_batch

# Some transforms
class MinMaxScaleBatched(grain.MapTransform):
    def map(self, sample_batch):
        image_key = list(sample_batch.keys())[0]
        sample_batch[image_key] = _MinMaxScaleBatched(sample_batch[image_key])
        return sample_batch

class RobustScaleBatched(grain.MapTransform):
    def map(self, sample_batch):
        image_key = list(sample_batch.keys())[0]
        sample_batch[image_key] = _RobustScaleBatched(sample_batch[image_key])
        return sample_batch

class BinaryEncodeBatched(grain.MapTransform):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def map(self, sample_batch):
        mask_key = list(sample_batch.keys())[-1]
        sample_batch[mask_key] = _BinaryEncodeBatched(sample_batch[mask_key], self.num_classes)
        return sample_batch


class OneHotEncodeBatched(grain.MapTransform):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def map(self, sample_batch):
        mask_key = list(sample_batch.keys())[-1]
        sample_batch[mask_key] = _OneHotEncodeBatched(sample_batch[mask_key], self.num_classes)
        return sample_batch


class RandomColorJitterBatched(grain.RandomMapTransform):
    pass


class RemapMasksBatched(grain.MapTransform):

    def __init__(self, original_classes, classes_to_background):
        self.original_classes = original_classes
        self.classes_to_background = classes_to_background
        self.remaining_classes = sorted(set(original_classes) - set(classes_to_background))

    def map(self, sample_batch):
        mask_key = list(sample_batch.keys())[-1]
        sample_batch[mask_key] = _RemapMasksBatched(sample_batch[mask_key], self.classes_to_background, self.original_classes)
        return sample_batch

class RandomRotateBatched(grain.MapTransform):
    def __init__(self, key, p, rot_angle):
        self.key = key
        self.p = p
        self.rot_angle = rot_angle

    def map(self, sample_batch):
        return _RandomRotateBatched(sample_batch, self.rot_angle, self.p, self.key)

class RandomFlipBatched(grain.MapTransform):
    def __init__(self, key, p):
        self.key = key
        self.p = p

    def map(self, sample_batch):
        return _RandomFlipBatched(sample_batch, self.key, self.p)


class ClaheHistTransformBatched(grain.MapTransform):
    def __init__(self, clip_limit):
        self.clip_limit = clip_limit

    def map(self, sample_batch):
        image_key = list(sample_batch.keys())[0]
        nbins = np.floor(jnp.mean(jnp.mean(jnp.percentile(sample_batch[image_key], 98, axis=(2, 3)) - jnp.percentile(sample_batch[image_key], 0, axis=(2, 3)), axis=1))).astype(int)
        sample_batch[image_key] = _ClaheHistTransformBatched(sample_batch[image_key], nbins=nbins, clip_limit=self.clip_limit)
        return sample_batch