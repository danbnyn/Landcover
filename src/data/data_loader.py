import jax
import grain.python as grain
import tensorflow_datasets as tfds
from src.data.transforms import MinMaxScaleBatched, BinaryEncodeBatched, OneHotEncodeBatched, RemapMasksBatched, \
    RandomRotateBatched, RandomFlipBatched, RobustScaleBatched, CustomSatelliteImageScaler, ClaheHistTransformBatched
from grain.python import ShardOptions
import os

class ShardByJaxProcessCustomBackend(ShardOptions):
  """Shards the data across JAX processes on TPU backend."""

  def __init__(self, drop_remainder: bool = False):
    # pylint: disable=g-import-not-at-top
    import jax  # pytype: disable=import-error
    # pylint: enable=g-import-not-at-top
    super().__init__(
        shard_index=jax.process_index('tpu'),
        shard_count=jax.process_count('tpu'),
        drop_remainder=drop_remainder,
    )

def create_iterator(data_dir, split, num_epochs, seed, batch_size, worker_count, worker_buffer_size, original_classes, classes_to_background, shuffle, transforms_bool, shard_bool):
    builder = tfds.builder_from_directory(data_dir)

    # jax.config.update("jax_platform_name", "cpu")

    # os.environ['JAX_PLATFORMS'] = 'cpu'


    # Define the sampler
    key = jax.random.key(seed)

    data_source = builder.as_data_source(split=split)

    if not shard_bool:
        sharding = grain.NoSharding()

    else :
        sharding = ShardByJaxProcessCustomBackend(drop_remainder=True)


    sampler = grain.IndexSampler(
        num_records=len(data_source),
        shard_options=sharding,
        shuffle=shuffle,
        num_epochs=num_epochs,
        seed=seed,
    )

    random_transforms_keys = jax.random.split(key, num=2) # idk how to set


    transformations = [
        grain.Batch(batch_size, drop_remainder=True),
        CustomSatelliteImageScaler(0, 99.85),
        RemapMasksBatched(original_classes, classes_to_background),
        OneHotEncodeBatched(len(RemapMasksBatched(original_classes, classes_to_background).remaining_classes) + 1),
    ]

    if transforms_bool :
        transformations = [
            grain.Batch(batch_size, drop_remainder=True),
            CustomSatelliteImageScaler(0, 99.85),
            RemapMasksBatched(original_classes, classes_to_background),
            RandomFlipBatched(random_transforms_keys[0], 0.5),
            RandomRotateBatched(random_transforms_keys[1], p=0.5, rot_angle=10),
            OneHotEncodeBatched(len(RemapMasksBatched(original_classes, classes_to_background).remaining_classes) + 1),
        ]

    # Create the dataloader
    data_loader = grain.DataLoader(
        data_source=data_source,
        sampler=sampler,
        operations=transformations,
        worker_count=worker_count,
        worker_buffer_size=worker_buffer_size,
        shard_options=sharding,
        read_options=None,
        enable_profiling=False,
    )

    iterator = iter(data_loader)

    return iterator

