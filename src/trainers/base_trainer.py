from functools import partial

import tensorboardX.writer

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

def check_epoch_boundary(iterator, current_epoch):
    """
    Checks for the end of an epoch within a data loader iteration and updates the state accordingly.

    """

    # Get the current state of the iterator
    state = json.loads(iterator.get_state().decode('utf-8'))
    num_records = int(re.search(r'num_records=(\d+)', state['sampler']).group(1)) # regex to find

    # 1. Check if we've crossed into the next epoch
    if all(i < (current_epoch + 1) * num_records for i in state['last_seen_indices'].values()):
        return False  # Not yet the end of the epoch

    #
    # # 2. Epoch boundary crossed, update state for the next epoch
    # worker_count = len(state['last_seen_indices'])
    # for i in range(worker_count):
    #     state['last_seen_indices'][str(i)] = ((current_epoch + 1) * num_records) - worker_count + i
    # state['last_worker_index'] = worker_count - 1  # Reset worker index
    #
    # # Update the DataLoader's state
    # iterator.set_state(json.dumps(state, indent=4).encode())

    return True #  Epoch finished

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

    (loss_value, new_state), grads = eqx.filter_value_and_grad(batch_loss_fn, has_aux=True)(model,state, inputs, targets, weights, loss_fn) # Compute the loss and the gradients, while keeping track of the state

    updates, new_opt_state = optimizer.update(grads, opt_state, params=eqx.filter(model, eqx.is_array))

    new_model = eqx.apply_updates(model, updates)

    return new_model, new_state, new_opt_state, loss_value

def train_epoch(
        model: ResUnet,
        state: eqx.nn.State,
        opt_state: optax.OptState,
        train_iterator: grain.PyGrainDatasetIterator,
        optimizer: optax.GradientTransformation,
        batch_loss_fn: Callable,
        loss_fn: Callable,
        weights: jnp.ndarray,
        current_epoch: int,
) -> Tuple[ResUnet, eqx.nn.State, optax.OptState, float]:
    total_loss = 0.0
    num_batches = 0

    progress_bar = create_progress_bar(train_iterator, "Training")

    num_devices = len(jax.devices("tpu"))
    mesh_shape = (num_devices,)
    devices = np.array(jax.devices('tpu')).reshape(mesh_shape)
    mesh = jax.sharding.Mesh(devices, ('batch'))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('batch'))

    sharding_model = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())


    model, opt_state = eqx.filter_shard((model, opt_state), sharding_model)

    for batch in train_iterator :
        if check_epoch_boundary(train_iterator, current_epoch):
            break

        inputs, targets = batch.values()

        # Move data to devices before sharding
        inputs = jax.device_put(inputs, sharding)
        targets = jax.device_put(targets, sharding)

        # Apply sharding constraints
        inputs, targets = eqx.filter_shard((inputs, targets), sharding)

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

    batch_model = jax.vmap(
        model, axis_name='batch', in_axes=(0, None), out_axes=(0, None)
    )
    y_pred, _ = batch_model(inputs, state)
    loss = loss_fn(y_pred, targets, weights)
    return loss


def validate(
    model: eqx.Module,
    state: eqx.nn.State,
    val_iterator: grain.PyGrainDatasetIterator,
    loss_fn: Callable,
    weights: Float[Array, "C"],
    current_epoch: int,
) -> float:
    total_loss = 0.0
    num_batches = 0

    progress_bar = create_progress_bar(val_iterator, "Validating")

    for batch in val_iterator:
        if check_epoch_boundary(val_iterator, current_epoch):
            break

        inputs, targets = batch.values()
        loss_value = validate_step(model, state, inputs, targets, loss_fn, weights)

        total_loss += loss_value
        num_batches += 1
        progress_bar.update(1)

    progress_bar.close()
    avg_loss = total_loss / num_batches
    return avg_loss


def train_model(
    model: ResUnet,
    state: eqx.nn.State,
    train_iterator: grain.PyGrainDatasetIterator,
    val_iterator: grain.PyGrainDatasetIterator,
    optimizer: optax.GradientTransformation,
    batch_loss_fn: Callable,
    loss_fn: Callable,
    weights: jnp.ndarray,
    num_epochs: int,
    checkpoint_manager: CheckpointManager,
    writer: tensorboardX.writer.SummaryWriter
):
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    epoch_progress = tqdm(range(num_epochs), desc="Epochs", position=0)

    for epoch in epoch_progress:
        # Training
        model, state, opt_state, train_loss = train_epoch(
            model, state, opt_state, train_iterator, optimizer, batch_loss_fn, loss_fn, weights, epoch
        )


        # Validation
        val_loss = validate(model, state, val_iterator, loss_fn, weights, epoch)

        # Logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)

        epoch_progress.set_postfix({'Train Loss': f'{train_loss:.4f}', 'Val Loss': f'{val_loss:.4f}'})

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            checkpoint = Checkpoint(model, state, opt_state, epoch, val_loss)
            checkpoint_manager.save_checkpoint(checkpoint)
        else:
            epochs_without_improvement += 1


    epoch_progress.close()
    writer.close()
    return model, state