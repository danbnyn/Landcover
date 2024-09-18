import argparse
from datetime import datetime
import os

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import tensorflow as tf
import yaml

import numpy as np

from src.models.resunet import ResUnet
from src.trainers.base_trainer import train_model
from src.utils.losses import batch_loss_fn, create_loss_fn, weighted_bce_loss
from src.data.data_loader import create_iterator, Shard
from src.utils.checkpoint import CheckpointManager

# jax.config.update("jax_platform_name", "cpu")  # Uncomment if you want to force CPU

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_config(config_path: str) -> dict:
    """
    Loads YAML configuration from the specified path.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration parameters.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main(config_path: str):
    """
    The main function orchestrates the training process.

    Args:
        config_path (str): Path to the YAML configuration file.
    """
    # Load configuration
    config = load_config(config_path)
    seed = config['seed']

    # Sharding Configuration
    num_devices = jax.local_device_count()
    devices = np.array(jax.devices())
    mesh = jax.sharding.Mesh(devices, ('batch',))

    # Define sharding specifications
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('batch'))
    sharding_model = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Define ShardOptions
    shard_options = Shard(
        shard_index=jax.process_index(),  # Adjust based on device index if multiple devices
        shard_count=jax.process_count(),
        drop_remainder=True
    )

    out_channels = len(config['data']['original_classes']) - len(config['data']['classes_to_background']) + 1

    # Initialize model with state
    model, state = eqx.nn.make_with_state(ResUnet)(
        in_channels=config['model']['in_channels'],
        out_channels=out_channels,
        n_filters=config['model']['n_filters'],
        key=jax.random.key(seed)
    )

    # Set up optimizer
    learning_rate = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)

    # Set up loss function
    weights = jnp.array(config['loss']['class_weights'])
    loss_fn = create_loss_fn(loss_type='weighted_bce_loss', weights=weights)

    # Initialize optimizer state and shard it
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    opt_state = jax.device_put(opt_state, sharding_model)

    # Shard the model and state
    model, opt_state = eqx.filter_shard((model, opt_state), sharding_model)

    # Set up data loaders
    train_iterator = create_iterator(
        data_dir=config['data']['data_directory'],
        split='train[:10%]',
        num_epochs=config['training']['num_epochs'],
        seed=seed,
        batch_size=config['data']['batch_size'],
        worker_count=config['data']['worker_count'],
        worker_buffer_size=config['data']['worker_buffer_size'],
        original_classes=config['data']['original_classes'],
        classes_to_background=config['data']['classes_to_background'],
        shuffle=True,
        transforms_bool=True,
        shard_bool=config['data']['shard_bool'],
        sharding=shard_options  # Pass sharding to the data pipeline
    )
    val_iterator = create_iterator(
        data_dir=config['data']['data_directory'],
        split='test[:10%]',
        num_epochs=1,  # Validation typically runs for one epoch
        seed=seed,
        batch_size=config['data']['batch_size'],
        worker_count=config['data']['worker_count'],
        worker_buffer_size=config['data']['worker_buffer_size'],
        original_classes=config['data']['original_classes'],
        classes_to_background=config['data']['classes_to_background'],
        shuffle=False,
        transforms_bool=False,
        shard_bool=config['data']['shard_bool'],
        sharding=shard_options
    )

    # Set up logging and checkpointing
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(config['logging']['outputs_dir'], f"run_{current_time}")

    # Create subdirectories for log, checkpoints, and visualization
    log_dir = os.path.join(run_dir, 'log')
    checkpoints_dir = os.path.join(run_dir, 'checkpoints')
    visualization_dir = os.path.join(run_dir, 'visualization')

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)

    train_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'train'))
    val_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'val'))

    checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoints_dir,
        max_to_keep=config['logging']['max_to_keep']
    )

    # Define metrics to compute
    metric_names = [
        "AccuracyMetric",
        "IoUMetric",
        "SensitivityMetric",
        "SpecificityMetric"
    ]

    num_classes = len(config['data']['original_classes']) - len(config['data']['classes_to_background']) + 1

     # Define class names for better labeling in TensorBoard
    class_names = config.get('data', {}).get('class_names', None)

    # Train model
    train_model(
        model=model,
        state=state,
        opt_state=opt_state,
        train_iterator=train_iterator,
        val_iterator=val_iterator,
        optimizer=optimizer,
        batch_loss_fn=batch_loss_fn,
        loss_fn=loss_fn,
        weights=weights,
        num_epochs=config['training']['num_epochs'],
        checkpoint_manager=checkpoint_manager,
        train_writer=train_writer,
        val_writer=val_writer,
        sharding=sharding,
        num_classes=num_classes, 
        metric_names=metric_names, 
        class_names=class_names
    )

    print("Training completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")
    args = parser.parse_args()
    
    # Start JAX profiler (optional)
    # jax.profiler.start_trace("/home/danbe/Landcover/tmp/tensorboard")
    
    main(args.config)
    
    # Stop JAX profiler
    # jax.profiler.stop_trace()
