import argparse
from random import shuffle

import yaml
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from tensorboardX import SummaryWriter
from src.models.resunet import ResUnet  # Import your model
from src.trainers.base_trainer import train_model
from src.utils.losses import batch_loss_fn, create_loss_fn, weighted_bce_loss
from src.data.data_loader import create_iterator
from src.utils.checkpoint import CheckpointManager

# jax.config.update("jax_platform_name", "cpu")

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main(config_path):
    # Load configuration
    config = load_config(config_path)

    seed = config['seed']

    # Set up data loaders
    train_iterator = create_iterator(
        data_dir = config['data']['data_directory'],
        split = 'train[:5%]',
        num_epochs = config['training']['num_epochs'],
        seed = seed,
        batch_size=config['data']['batch_size'],
        worker_count=config['data']['worker_count'],
        worker_buffer_size = config['data']['worker_buffer_size'],
        original_classes = config['data']['original_classes'],
        classes_to_background= config['data']['classes_to_background'],
        shuffle = True,
        transforms_bool = True,
        shard_bool = config['data']['shard_bool']
    )
    val_iterator = create_iterator(
        data_dir = config['data']['data_directory'],
        split = 'test',
        num_epochs = config['training']['num_epochs'],
        seed = config['seed'],
        batch_size=config['data']['batch_size'],
        worker_count=config['data']['worker_count'],
        worker_buffer_size = config['data']['worker_buffer_size'],
        original_classes = config['data']['original_classes'],
        classes_to_background= config['data']['classes_to_background'],
        shuffle = False,
        transforms_bool = False,
        shard_bool = config['data']['shard_bool']
    )

    # Initialize model
    model, state = eqx.nn.make_with_state(ResUnet)(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        n_filters=config['model']['n_filters'],
        key = jax.random.key(seed)
        )


    # Set up optimizer
    learning_rate = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)

    # Set up loss function
    weights = jnp.array(config['loss']['class_weights'])
    loss_fn = create_loss_fn(loss_type='weighted_bce_loss', weights = weights)

    # Set up logging and checkpointing
    writer = SummaryWriter(config['logging']['log_dir'])
    checkpoint_manager = CheckpointManager(config['logging']['checkpoint_dir'], config['logging']['max_to_keep'])

    # Train model
    trained_model, final_state = train_model(
        model=model,
        state = state,
        train_iterator=train_iterator,
        val_iterator=val_iterator,
        optimizer=optimizer,
        batch_loss_fn=batch_loss_fn,
        loss_fn=loss_fn,
        weights=weights,
        num_epochs=config['training']['num_epochs'],
        checkpoint_manager=checkpoint_manager,
        writer = writer
    )

    print("Training completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")
    args = parser.parse_args()
    main(args.config)