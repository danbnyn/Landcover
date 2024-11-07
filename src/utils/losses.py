from jaxtyping import Float, Int, Array, PyTree
import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Callable
from tqdm.auto import tqdm
import numpy as np
from typing import Dict, Optional
import grain.python as grain
from typing import List, Union

import optax
import jax
import jax.numpy as jnp
from typing import Callable, Optional, Union

# Factory function to create appropriate loss function
def create_loss_fn(loss_type: str = "cross_entropy", **kwargs) -> Callable:
    """
    Factory function to create a loss function based on the specified loss_type.

    Args:
        loss_type: str, type of loss function to create ("cross_entropy", "weighted_cross_entropy", "focal_loss").
        **kwargs: Additional keyword arguments to pass to the loss function.

    Returns:
        A callable loss function.
    """
    if loss_type == "cross_entropy":
        def loss_fn(predictions, targets, weights):
            predictions = jnp.moveaxis(predictions, 1, -1)
            ce = optax.softmax_cross_entropy(
                logits=predictions, labels=targets
            )
            return jnp.mean(ce)
    elif loss_type == "weighted_cross_entropy":
        def loss_fn(predictions, targets, weights):

            # Apply smooth_labels
            targets = label_smoothing(targets, epsilon=0.1)

            return weighted_softmax_cross_entropy(predictions, targets, weights)
        
    elif loss_type == "focal_loss":
        def loss_fn(predictions, targets, weights):
            return multi_class_focal_loss(predictions, targets, weights, **kwargs)
    else:
        raise ValueError(f"Unknown loss function type: {loss_type}")

    return loss_fn

# Multi-Class Focal Loss Implementation
def multi_class_focal_loss(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    gamma: float = 2.0,
    alpha: Union[float, jnp.ndarray] = 0.25
) -> jnp.ndarray:
    """
    Compute the Focal Loss for multi-class classification.

    Args:
        predictions: jnp.ndarray of shape (batch, num_classes, h, w), predicted probabilities.
        targets: jnp.ndarray of shape (batch, h, w), ground truth labels as class indices.
        weights: Optional[jnp.ndarray] of shape (num_classes,), weights to apply to each class.
        gamma: float, focusing parameter to reduce the loss contribution from easy examples.
        alpha: float or jnp.ndarray, weighting factor to balance positive vs negative examples.
               If float, same alpha is applied to all classes. If ndarray, should have shape (num_classes,).

    Returns:
        loss: scalar, the weighted focal loss.
    """
    epsilon = 1e-8
    predictions = jnp.clip(predictions, epsilon, 1.0 - epsilon)
    # Convert targets to one-hot encoding
    num_classes = predictions.shape[1]
    targets_one_hot = jax.nn.one_hot(targets, num_classes=num_classes)  # Shape: (N, C, H, W)
    # Compute cross-entropy
    cross_entropy = -targets_one_hot * jnp.log(predictions)
    # Compute p_t
    p_t = jnp.sum(predictions * targets_one_hot, axis=1)  # Shape: (N, H, W)
    # Compute alpha_t
    if isinstance(alpha, float):
        alpha_t = alpha * targets_one_hot + (1 - alpha) * (1 - targets_one_hot)
    else:
        # alpha is expected to be a jnp.ndarray with shape (num_classes,)
        alpha = alpha.reshape((1, -1, 1, 1))  # (1, num_classes, 1, 1)
        alpha_t = alpha * targets_one_hot + (1 - alpha) * (1 - targets_one_hot)
    # Compute focal weight
    focal_weight = (1 - p_t) ** gamma
    # Expand focal_weight to match cross_entropy shape
    focal_weight = focal_weight.reshape((focal_weight.shape[0], 1, focal_weight.shape[1], focal_weight.shape[2]))
    # Compute focal loss
    loss = focal_weight * cross_entropy
    # Apply class weights if provided
    if weights is not None:
        weights = weights.reshape((1, -1, 1, 1))  # (1, num_classes, 1, 1)
        loss = loss * weights
    # Sum over classes and average
    loss = jnp.sum(loss, axis=1)  # Shape: (N, H, W)
    return jnp.mean(loss)

def weighted_softmax_cross_entropy(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    Computes the weighted softmax cross-entropy loss for multi-class segmentation.

    Args:
        logits (jnp.ndarray): Unnormalized log probabilities with shape [N, C, H, W],
                              where N is the batch size, C is the number of classes,
                              H and W are the height and width of the input.
        labels (jnp.ndarray): One-hot encoded ground truth labels with shape [N, C, H, W].
        weights (Optional[jnp.ndarray]): Class weights with shape [C]. Each element
                                         corresponds to the weight of a class.

    Returns:
        jnp.ndarray: A scalar representing the mean weighted softmax cross-entropy loss.
    """
    # Compute log softmax over the class (channel) dimension
    log_probs = jax.nn.log_softmax(logits, axis=1)  # Shape: [N, C, H, W]

    # Compute the negative log likelihood
    loss = -jnp.sum(labels * log_probs, axis=1)  # Shape: [N, H, W]

    if weights is not None:
        # Ensure weights have shape [C]
        weights = weights.reshape((1, -1, 1, 1))  # Shape: [1, C, 1, 1]

        # Compute weighted loss by multiplying each class's loss with its weight
        loss = loss * jnp.sum(labels * weights, axis=1)  # Shape: [N, H, W]

    # Compute the mean loss over all pixels and the batch
    return loss.mean()

def label_smoothing(
    labels: jnp.ndarray,
    epsilon: float = 0.1
) -> jnp.ndarray:
    """
    Applies label smoothing to one-hot encoded labels.

    Args:
        labels (jnp.ndarray): One-hot encoded labels with shape [N, C, H, W].
        epsilon (float): Smoothing factor. The smoothing is applied as:
                         labels = labels * (1 - epsilon) + (epsilon / C)

    Returns:
        jnp.ndarray: Smoothed labels with the same shape as input.
    """
    num_classes = labels.shape[1]
    return labels * (1.0 - epsilon) + (epsilon / num_classes)

# Updated batch_loss_fn for Multi-Class
def batch_loss_fn(
    model: eqx.Module,
    state: eqx.nn.State,
    x_true: Float[Array, "N C H W"],
    y_true: Int[Array, "N H W"],  # Class indices
    weights: Float[Array, "C"],
    loss_fn: Callable[[Float[Array, "N C H W"], Int[Array, "N H W"], Float[Array, "C"]], Float[Array, "..."]],
) -> PyTree[Float[Array, "..."]]:
    """
    Computes the loss for a batch.

    Args:
        model: The neural network model.
        state: The state associated with the model.
        x_true: Input images.
        y_true: Ground truth labels as class indices.
        weights: Class weights.
        loss_fn: The loss function to use.

    Returns:
        Tuple of (loss, new_state).
    """
    batch_model = jax.vmap(
        model, axis_name='batch', in_axes=(0,None), out_axes=(0,None)
    )
    y_pred, new_state = batch_model(x_true, state)
    # Compute loss
    loss = loss_fn(y_pred, y_true, weights)

    return loss, new_state



def compute_class_frequencies(
    dataset_iterator: grain.PyGrainDatasetIterator,
    num_classes: int,
    num_batches: Optional[int] = None,
    verbose: bool = True,
) -> Dict[int, float]:
    """
    Computes class frequencies the dataset.

    Args:
        dataset_iterator (grain.PyGrainDatasetIterator): Iterator over the dataset.
        num_classes (int): Total number of classes.
        num_batches (Optional[int], optional): Number of batches to process. Defaults to None (process entire dataset).
        mask_key (str, optional): Key to access mask in batch dictionary. Defaults to "mask".
        verbose (bool, optional): If True, displays a progress bar. Defaults to True.
        mode (str, optional): Strategy to compute class weights. Defaults to "inverse_frequency".

    Returns:
        Dict[int, float]: Dictionary mapping class indices to their corresponding weights.
    """
    class_counts = np.zeros(num_classes, dtype=np.int64)


    for batch in tqdm(dataset_iterator, total=num_batches):
        mask_key = list(batch.keys())[1]
        masks = batch[mask_key]  # Shape: (N, H, W)

        # Flatten masks to 1D array
        masks_flat = masks.reshape(-1)

        # Count occurrences of each class in the batch
        counts = np.bincount(masks_flat, minlength=num_classes)

        # Accumulate counts
        class_counts += counts

    if verbose:
        print(f"Class Counts: {class_counts}")

    # Compute class frequencies
    total_pixels = np.sum(class_counts)

    class_frequencies = class_counts / total_pixels

    if verbose:
        print(f"Class Frequencies: {class_frequencies}")

    return class_frequencies


from typing import List
import numpy as np
import jax.numpy as jnp

def process_weights(
    class_frequencies: List[float],
    original_classes: List[int],
    classes_to_background: List[int],
    mode: str = "inverse_frequency",
    weights_normalization_method: str = "none",
    verbose: bool = True
) -> jnp.ndarray:
    """
    Process class weights by first combining the frequencies of classes mapped to background into the background class,
    then remapping the remaining classes to have contiguous indices starting from 0.
    Also computes class weights based on the selected mode, sets background class weight to zero, and normalizes the weights.

    Args:
        class_frequencies (List[float]): List of class frequencies.
        original_classes (List[int]): List of original class indices.
        classes_to_background (List[int]): List of class indices that are mapped to background.
        mode (str, optional): Strategy to compute class weights. Defaults to "inverse_frequency".
        weights_normalization_method (str, optional): Method to normalize weights. Defaults to "none".
        verbose (bool, optional): If True, displays a warning message for classes with zero frequency. Defaults to True.

    Returns:
        jnp.ndarray: Processed and normalized class weights with remapped class indices.
    """
    # Step 1: Combine the frequencies of classes mapped to background
    background_frequency = sum(class_frequencies[cls] for cls in classes_to_background)
    
    # Initialize new_frequencies with the combined background frequency
    # Assuming the background class is originally class 0
    new_frequencies = [class_frequencies[0] + background_frequency]
    remapped_classes = [0]  # Remapped index 0 corresponds to the new background class
    
    # Step 2: Remap remaining classes to contiguous indices, excluding those mapped to background
    for cls in original_classes:
        if cls not in classes_to_background:
            new_frequencies.append(class_frequencies[cls])
            remapped_classes.append(cls)
    
    if verbose:
        print(f"New Frequencies: {new_frequencies}")
    
    # Compute class weights based on the updated frequency list
    class_weights = {}
    for cls, frequency in enumerate(new_frequencies):
        if frequency > 0:
            if mode == "sqrt_inverse_frequency":
                class_weights[cls] = 1.0 / np.sqrt(frequency)
            elif mode == "log_inverse_frequency":
                class_weights[cls] = np.log(1.0 + 1.0 / (frequency))
            elif mode == "median_frequency":
                class_weights[cls] = np.median(new_frequencies) / frequency
            elif mode == "inverse_frequency":
                class_weights[cls] = 1.0 / frequency
            else:  # Default to equal weights
                class_weights[cls] = 1.0
        else:
            class_weights[cls] = 0.0
            if verbose:
                print(f"Warning: Class {cls} has zero frequency.")
    

    # Normalize weights
    if weights_normalization_method == "minimum_weight_to_one":
        min_weight = min(class_weights.values())
        if min_weight > 0:
            for cls in class_weights:
                class_weights[cls] /= min_weight
        else:
            if verbose:
                print("Warning: Minimum weight is zero during minimum normalization.")
    
    elif weights_normalization_method == "sum_to_one":
        total_weight = sum(class_weights.values())
        if total_weight > 0:
            for cls in class_weights:
                class_weights[cls] /= total_weight
        else:
            if verbose:
                print("Warning: Total weight is zero during sum normalization.")
    
    elif weights_normalization_method == "mean_weight_to_one":
        mean_weight = np.mean(list(class_weights.values()))
        if mean_weight > 0:
            for cls in class_weights:
                class_weights[cls] /= mean_weight
        else:
            if verbose:
                print("Warning: Mean weight is zero during mean normalization.")
    
    elif weights_normalization_method == "none":
        pass  # No normalization
    
    else:
        raise ValueError(f"Unknown normalization method: {weights_normalization_method}")
    
    # Convert to jnp array
    processed_weights_array = jnp.array([class_weights[cls] for cls in range(len(class_weights))])
    
    return processed_weights_array
