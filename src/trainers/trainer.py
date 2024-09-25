from functools import partial

from datetime import datetime

from tensorboardX import SummaryWriter


from src.models.resunet import ResUnet
import equinox as eqx
import jax
from jax.experimental.mesh_utils import create_device_mesh
import optax
import jax.numpy as jnp
import grain.python as grain
from tqdm import tqdm
from typing import Callable, Tuple, NamedTuple, Dict
from src.utils.checkpoint import CheckpointManager, Checkpoint
from jaxtyping import Array, Float, Int, PyTree
from src.utils.visualization import create_progress_bar
import json
import re
import numpy as np
from src.data.transforms import _OneHotEncodeBatched
from src.utils.metrics import Metric, get_metrics
from typing import List, Any, Union, Optional
from src.utils.metrics import log_metrics, ConfusionMatrixMetric
from src.utils.visualization import log_sample_visualizations

def check_epoch_boundary(iterator, current_epoch, batch_size):
    """
    Checks for the end of an epoch within a data loader iteration and updates the state accordingly.

    """

    # Get the current state of the iterator
    state = json.loads(iterator.get_state().decode('utf-8'))
    num_records = int(re.search(r'num_records=(\d+)', state['sampler']).group(1)) # regex to find

    # check if adding a batch to the current step will cross the epoch boundary
    if any([state['last_seen_indices'][str(i)] + batch_size >= (current_epoch + 1) * num_records for i in range(len(state['last_seen_indices']))]):

        # 2. Epoch boundary crossed, update state for the next epoch
        worker_count = len(state['last_seen_indices'])
        for i in range(worker_count):
            state['last_seen_indices'][str(i)] = ((current_epoch + 1) * num_records) - worker_count + i
        state['last_worker_index'] =  - 1  # Reset worker index
        
        # Update the DataLoader's state
        iterator.set_state(json.dumps(state, indent=4).encode())

        return True
    

    return False #  Epoch finished

@partial(eqx.filter_jit)
def train_step(
        model: eqx.Module,
        state: eqx.nn.State,
        opt_state: optax.OptState,
        inputs: Float[Array, "N C H W"],
        targets: Int[Array, "N H W"],
        optimizer: optax.GradientTransformation,
        batch_loss_fn: Callable,
        loss_fn: Callable,
        weights: Float[Array, "C"]
):
    """
    Performs a single training step: forward pass, loss computation, gradient computation, and parameter update.

    Returns:
        Tuple containing updated model, state, optimizer state, and loss value.
    """
    (loss_value, new_state), grads = eqx.filter_value_and_grad(
        batch_loss_fn, has_aux=True
    )(model, state, inputs, targets, weights, loss_fn)  # Compute loss and gradients

    updates, new_opt_state = optimizer.update(
        grads, opt_state, params=eqx.filter(model, eqx.is_array)
    )

    new_model = eqx.apply_updates(model, updates)

    return new_model, new_state, new_opt_state, loss_value

def train_epoch(
        model: eqx.Module,
        state: eqx.nn.State,
        opt_state: optax.OptState,
        train_iterator: grain.PyGrainDatasetIterator,
        optimizer: optax.GradientTransformation,
        batch_loss_fn: Callable,
        loss_fn: Callable,
        weights: jnp.ndarray,
        current_epoch: int,
        sharding: jax.sharding.NamedSharding,
):
    """
    Executes one epoch of training.

    Returns:
        Tuple containing updated model, state, optimizer state, and average loss.
    """
    total_loss = 0.0
    num_batches = 0

    progress_bar = create_progress_bar(train_iterator, "Training")

    for batch in train_iterator:

        batch_size = list(batch.values())[0].shape[0]

        if check_epoch_boundary(train_iterator, current_epoch, batch_size):
            break

        inputs, targets = batch.values()

        # Move data to devices based on sharding
        inputs = jax.device_put(inputs, sharding)
        targets = jax.device_put(targets, sharding)

        # Apply sharding constraints
        inputs, targets = eqx.filter_shard((inputs, targets), sharding)

        # Perform a training step
        model, state, opt_state, loss_value = train_step(
            model, state, opt_state, inputs, targets, optimizer, batch_loss_fn, loss_fn, weights
        )

        total_loss += loss_value
        num_batches += 1
        progress_bar.update(1)

    progress_bar.close()

    avg_loss = total_loss / num_batches

    return model, state, opt_state, avg_loss

@partial(eqx.filter_jit, backend='tpu')
def validate_step(
    model: eqx.Module,
    state: eqx.nn.State,
    inputs: Float[Array, "N C H W"],
    targets: Int[Array, "N H W"],
    loss_fn: Callable,
    weights: Float[Array, "C"],
) -> Float[Array, ""]:
    """
    Performs a single validation step: forward pass and loss computation.

    Returns:
        Computed loss.
    """
    batch_model = jax.vmap(
        model, axis_name='batch', in_axes=(0, None), out_axes=(0, None)
    ) # As per https://docs.kidger.site/equinox/examples/stateful/
    y_pred, _ = batch_model(inputs, state)
    loss = loss_fn(y_pred, targets, weights)
    return y_pred ,loss

def validate(
    model: eqx.Module,
    state: eqx.nn.State,
    val_iterator: grain.PyGrainDatasetIterator,
    loss_fn: Callable,
    weights: Float[Array, "C"],
    current_epoch: int,
    sharding: jax.sharding.NamedSharding,
    num_classes: int,
    metric_names: List[str],  # List of metric class names
    num_samples: int = 3,  # Number of samples to visualize
) -> Tuple[Dict[str, Any], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Executes the validation loop and computes performance metrics.

    Args:
        model (eqx.Module): The neural network model.
        state (eqx.nn.State): The state associated with the model.
        val_iterator (grain.PyGrainDatasetIterator): Validation data iterator.
        loss_fn (Callable): Loss function.
        weights (Float[Array, "C"]): Class weights for loss computation.
        current_epoch (int): Current epoch number.
        sharding (jax.sharding.NamedSharding): Sharding configuration.
        num_classes (int): Number of classes.
        metric_names (List[str]): List of metric class names to compute.
        num_samples (int, optional): Number of samples to visualize. Defaults to 3.
        class_names (List[str], optional): List of class names for labeling. Defaults to None.

    Returns:
        Tuple[Dict[str, Any], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
            - Dictionary containing all computed metrics.
            - Tuple containing (images_rgb_nir, masks, predictions) for visualization.
    """
    metrics = get_metrics(metric_names, num_classes)

    # Identify the ConfusionMatrixMetric instance
    confusion_matrix_metric = None
    for metric in metrics:
        if isinstance(metric, ConfusionMatrixMetric):
            confusion_matrix_metric = metric
            break

    total_loss = 0.0
    num_batches = 0

    sample_images_rgb_nir = []
    sample_masks = []
    sample_predictions = []

    progress_bar = create_progress_bar(val_iterator, "Validating")

    for batch in val_iterator:
        batch_size = list(batch.values())[0].shape[0]

        if check_epoch_boundary(val_iterator, current_epoch, batch_size):
            break

        inputs, targets = batch.values()

        # Move data to devices based on sharding
        inputs = jax.device_put(inputs, sharding)
        targets = jax.device_put(targets, sharding)

        # Apply sharding constraints
        inputs, targets = eqx.filter_shard((inputs, targets), sharding)

        # Perform a validation step
        y_pred, loss_value = validate_step(model, state, inputs, targets, loss_fn, weights)

        y_pred_cls = jnp.argmax(y_pred, axis=1)  # Shape: (N, H, W)

        y_true_cls = jnp.argmax(targets, axis=1)  # Shape: (N, H, W)

        # put them on TPU for computation
        y_pred_cls = jax.device_put(y_pred_cls, sharding)
        y_true_cls = jax.device_put(y_true_cls, sharding)

        # Update confusion matrix
        confusion_matrix_metric.update(y_pred_cls, y_true_cls)

        total_loss += loss_value
        num_batches += 1
        progress_bar.update(1)

    progress_bar.close()


    current_samples = min(num_samples, inputs.shape[0])
    sample_images_rgb_nir = inputs[:current_samples]

    sample_masks = y_true_cls[:current_samples]
    sample_predictions = y_pred_cls[:current_samples]  # (N, H, W)


    # Compute all metrics
    final_metrics: Dict[str, Any] = {}
    for metric in metrics:
        if isinstance(metric, ConfusionMatrixMetric):
            metric_results = metric.compute()
        else:
            metric_results = metric.compute(confusion_matrix_metric)
        final_metrics.update(metric_results)


    # Compute average loss
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0


    return final_metrics, avg_loss, (sample_images_rgb_nir, sample_masks, sample_predictions)


def train_model(
    model: ResUnet,
    state: eqx.nn.State,
    opt_state: optax.OptState,
    train_iterator: grain.PyGrainDatasetIterator,
    val_iterator: grain.PyGrainDatasetIterator,
    optimizer: optax.GradientTransformation,
    batch_loss_fn: Callable,
    loss_fn: Callable,
    weights: jnp.ndarray,
    num_epochs: int,
    checkpoint_manager: CheckpointManager,
    writer: SummaryWriter,
    sharding: jax.sharding.NamedSharding,  # Sharding configuration
    num_classes: int,  # Number of classes
    metric_names: List[str],  # List of metric class names
    class_names: List[str] = None,  # Optional: List of class names for labeling
    num_visualization_samples: int = 3  # Number of samples to visualize
):
    """
    Orchestrates the training and validation process over multiple epochs.

    Args:
        model (ResUnet): The neural network model.
        state (eqx.nn.State): The state associated with the model.
        opt_state (optax.OptState): The optimizer state.
        train_iterator (grain.PyGrainDatasetIterator): Training data iterator.
        val_iterator (grain.PyGrainDatasetIterator): Validation data iterator.
        optimizer (optax.GradientTransformation): The optimizer.
        batch_loss_fn (Callable): Function to compute batch loss.
        loss_fn (Callable): Function to compute loss.
        weights (jnp.ndarray): Class weights for loss computation.
        num_epochs (int): Number of training epochs.
        checkpoint_manager (CheckpointManager): Manager for saving checkpoints.
        train_writer (tf.summary.SummaryWriter): TensorBoard writer for training.
        val_writer (tf.summary.SummaryWriter): TensorBoard writer for validation.
        sharding (jax.sharding.NamedSharding): Sharding configuration.
        num_classes (int): Number of classes.
        metric_names (List[str]): List of metric class names to compute.
        class_names (List[str], optional): List of class names for labeling. Defaults to None.
    """
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    epoch_progress = tqdm(range(num_epochs), desc="Epochs", position=0)

    for epoch in epoch_progress:
        # Training Phase

        model, state, opt_state, train_loss = train_epoch(
            model, 
            state, 
            opt_state, 
            train_iterator, 
            optimizer, 
            batch_loss_fn, 
            loss_fn,
            weights, 
            epoch, 
            sharding
        )

        # Validation Phase
        final_metrics, val_loss, samples = validate(
            model=model,
            state=state,
            val_iterator=val_iterator,
            loss_fn=loss_fn,
            weights=weights,
            current_epoch=epoch,
            sharding=sharding,
            num_classes=num_classes,
            metric_names=metric_names,
            num_samples=num_visualization_samples,
        )

        # Logging using TensorBoard via log_metrics
        log_metrics(
            metrics=final_metrics,
            writer=writer,
            step=epoch,
            class_names=class_names
        )

        # Logging of the training and validation losses
        writer.add_scalars(
            "Loss",
            {
                "Train": train_loss,
                "Validation": val_loss
            },
            epoch
        )

        # Logging Sample Visualizations
        images_rgb_nir, masks, predictions = samples
        if images_rgb_nir.size != 0:
            log_sample_visualizations(
                writer=writer,
                images_rgb_nir=images_rgb_nir,
                masks=masks,
                predictions=predictions,
                class_names=class_names,
                step=epoch,
                num_samples=num_visualization_samples
            )

        # Update progress bar with metrics
        epoch_progress.set_postfix({
            'Train Loss': f'{train_loss:.4f}',
            'Val Loss': f'{val_loss:.4f}',
        })

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            checkpoint = Checkpoint(model, state, opt_state, epoch, val_loss)
            checkpoint_manager.save_checkpoint(checkpoint)
        else:
            epochs_without_improvement += 1

        # patience = config['training']['patience']
        # if epochs_without_improvement >= patience:
        #     print("Early stopping triggered.")
        #     break

    # Close the writers at the end of training
    writer.close()

    epoch_progress.close()
    print("Training completed.")