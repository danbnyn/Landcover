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

def process_weights(class_frequencies: Dict[int, float], original_classes: list, classes_to_background: list, mode: str = "inverse_frequency", verbose: bool = True) -> List[float]:
    """
    Process class weights to remove weights for classes that are mapped to background. Also, set the background class weight to zero and normalize the weights.

    Args:
        class_frequencies (Dict[int, float]): Dictionary mapping class indices to their corresponding frequencies.
        original_classes (list): List of original class indices.
        classes_to_background (list): List of class indices that are mapped to background.
        mode (str, optional): Strategy to compute class weights. Defaults to "inverse_frequency".
        verbose (bool, optional): If True, displays a warning message for classes with zero frequency. Defaults to True.

    Returns:
       List[int, float]: Processed class weights.
    """
    class_weights = {}
    for cls, frequencies in enumerate(class_frequencies):
        if class_frequencies[cls] > 0:
            if mode == "sqrt_inverse_frequency":
                class_weights[cls] = 1.0 / np.sqrt(frequencies)
            elif mode == "log_inverse_frequency":
                class_weights[cls] = 1.0 / np.log(frequencies)
            elif mode == "median_frequency":
                class_weights[cls] = np.median(class_frequencies) / frequencies
            else :  # Default to inverse frequency
                class_weights[cls] = 1.0 / frequencies
        else:
            # Handle classes with zero frequency
            class_weights[cls] = 0.0
            if verbose:
                print(f"Warning: Class {cls} has zero frequency.")


    remaining_classes = [0] + [cls for cls in original_classes if cls not in classes_to_background] 
    processed_weights = {cls: class_weights[cls] for cls in remaining_classes}

    # Set background class weight to zero
    processed_weights[0] = 0.0

    # Normalize weights
    total_weight = sum(processed_weights.values())
    processed_weights = {cls: weight / total_weight for cls, weight in processed_weights.items()}

    # turn it into jnp array
    processed_weights = jnp.array([processed_weights[cls] for cls in processed_weights.keys()])

    return processed_weights