from jaxtyping import Float, Int, Array, PyTree
import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Callable
from tqdm.auto import tqdm
import numpy as np
from typing import Dict, Optional
import grain.python as grain
from typing import List

# Generalized loss function creator
def create_loss_fn(loss_type="weighted_bce_loss", **kwargs):
    if loss_type == "weighted_bce_loss":
        def loss_fn(predictions, targets, weights):
            return weighted_bce_loss(predictions, targets, weights)
    else:
        raise ValueError(f"Unknown loss function type: {loss_type}")

    return loss_fn

def weighted_bce_loss(predictions, targets, weights=None):
    """
    Compute weighted binary cross-entropy loss.

    Args:
        predictions: jnp.ndarray of shape (batch, num_classes, h, w), predicted probabilities.
        targets: jnp.ndarray of shape (batch, num_classes, h, w), ground truth labels (0 or 1).
        weights: jnp.ndarray of shape (num_classes,), weights to apply to each class.

    Returns:
        loss: the weighted binary cross-entropy loss.
    """

    # Compute binary cross-entropy loss for each element
    bce_loss = - (targets * jnp.clip(jnp.log(predictions), min = -100) + (1 - targets) * jnp.clip(jnp.log(1 - predictions), min = -100))

    # Apply weights if provided
    if weights is not None:
        # Reshape weights to match the shape of the bce_loss (batch, num_classes, h, w)
        weights = weights.reshape((1, -1, 1, 1))  # Reshape to (1, num_classes, 1, 1)
        bce_loss = bce_loss * weights

    return jnp.mean(bce_loss)

def batch_loss_fn(
    model: eqx.Module,
    state: eqx.nn.State,
    x_true: Float[Array, " N C H W"],
    y_true: Int[Array, " N C H W"],
    weights: Float[Array, "C"],
    loss_fn: Callable[[Float[Array, "N C H W"], Int[Array, "N C H W"], Float[Array, "C"]], Float[Array, "..."]],
) -> PyTree[Float[Array, "..."]]:

    batch_model = jax.vmap(
        model, axis_name='batch', in_axes=(0,None), out_axes=(0,None)
    )
    y_pred, new_state = batch_model(x_true, state)
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
                class_weights[cls] = np.log(1.0 + 1.0 / frequency)
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
