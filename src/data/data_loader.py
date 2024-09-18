import jax
import grain.python as grain
import tensorflow_datasets as tfds
from grain.python import ShardOptions
import os

from src.data.transforms import (
    MinMaxScaleBatched, BinaryEncodeBatched, OneHotEncodeBatched, 
    RemapMasksBatched, RandomRotateBatched, RandomFlipBatched, 
    RobustScaleBatched, CustomSatelliteImageScaler, ClaheHistTransformBatched
)

import jax.numpy as jnp

class Shard(ShardOptions):
    def __init__(self, shard_index: int, shard_count: int, drop_remainder: bool = False):
        super().__init__(
            shard_index=shard_index,
            shard_count=shard_count,
            drop_remainder=drop_remainder
        )


def create_iterator(
    data_dir: str, 
    split: str, 
    num_epochs: int, 
    seed: int, 
    batch_size: int, 
    worker_count: int, 
    worker_buffer_size: int, 
    original_classes: list, 
    classes_to_background: list, 
    shuffle: bool, 
    transforms_bool: bool, 
    shard_bool: bool,
    sharding: Shard  # Sharding object
) -> grain.PyGrainDatasetIterator:
    """
    Creates a Grain data iterator with the specified configurations.

    Args:
        data_dir (str): Directory containing the dataset.
        split (str): Dataset split (e.g., 'train', 'test').
        num_epochs (int): Number of epochs to iterate over the data.
        seed (int): Random seed for shuffling.
        batch_size (int): Size of each data batch.
        worker_count (int): Number of worker processes for data loading.
        worker_buffer_size (int): Buffer size for data workers.
        original_classes (list): List of original class labels.
        classes_to_background (list): Classes to map to background.
        shuffle (bool): Whether to shuffle the data.
        transforms_bool (bool): Whether to apply data transformations.
        shard_bool (bool): Whether to apply sharding.
        sharding (Shard): Sharding configuration.

    Returns:
        grain.PyGrainDatasetIterator: Configured data iterator.
    """
    builder = tfds.builder_from_directory(data_dir)
    
    # Define the sampler
    key = jax.random.key(seed)
    
    data_source = builder.as_data_source(split=split)
    
    if not shard_bool:
        sharding_option = grain.NoSharding()
    else:
        sharding_option = sharding  # Use the passed sharding object
    
    sampler = grain.IndexSampler(
        num_records=len(data_source),
        shard_options=sharding_option,
        shuffle=shuffle,
        num_epochs=num_epochs,
        seed=seed,
    )
    
    random_transforms_keys = jax.random.split(key, num=2)  # Adjust the number based on your transforms
    
    # Define transformations
    transformations = [
        grain.Batch(batch_size, drop_remainder=True),
        CustomSatelliteImageScaler(0, 99.85),
        RemapMasksBatched(original_classes, classes_to_background),
    ]
    
    if transforms_bool:
        transformations.extend([
            RandomFlipBatched(random_transforms_keys[0], 0.5),
            RandomRotateBatched(random_transforms_keys[1], p=0.5, rot_angle=10),
            # Add more transforms if needed
        ])
    
    transformations.extend([
                OneHotEncodeBatched(
            len(RemapMasksBatched(original_classes, classes_to_background).remaining_classes) + 1
        )]
    )
    # Add the Encoding at the end to avoid un necessary additional computation
    
    # Create the dataloader
    data_loader = grain.DataLoader(
        data_source=data_source,
        sampler=sampler,
        operations=transformations,
        worker_count=worker_count,
        worker_buffer_size=worker_buffer_size,
        shard_options=sharding_option,
        read_options=None,
        enable_profiling=False,
    )
    
    iterator = iter(data_loader)
    
    return iterator

