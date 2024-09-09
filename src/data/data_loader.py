import jax
import grain.python as grain
import tensorflow_datasets as tfds
from src.data.transforms import MinMaxScale, BinaryEncode, OneHotEncode, RemapMasks, RandomRotate, RandomFlip, ToJax
from src.data.batched_transforms import MinMaxScaleBatched, BinaryEncodeBatched, OneHotEncodeBatched, RemapMasksBatched, \
    RandomRotateBatched, RandomFlipBatched, RobustScaleBatched, CustomSatelliteImageScaler, ClaheHistTransformBatched


def create_iterator(data_dir, split, num_epochs, seed, batch_size, worker_count, worker_buffer_size, original_classes, classes_to_background, shuffle, transforms_bool, shard_bool, clip_limit):
    builder = tfds.builder_from_directory(data_dir)

    # Define the sampler
    key = jax.random.key(seed)

    data_source = builder.as_data_source(split=split)

    if not shard_bool:
        sharding = grain.NoSharding()

    else :
        sharding = grain.ShardByJaxProcess(drop_remainder=True)


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
        ClaheHistTransformBatched(clip_limit),
        MinMaxScaleBatched(),
        RemapMasksBatched(original_classes, classes_to_background),
        OneHotEncodeBatched(len(RemapMasks(original_classes, classes_to_background).remaining_classes) + 1),
    ]

    if transforms_bool :
        transformations = [
            grain.Batch(batch_size, drop_remainder=True),
            MinMaxScaleBatched(),
            ClaheHistTransformBatched(clip_limit),
            RemapMasksBatched(original_classes, classes_to_background),
            RandomFlipBatched(random_transforms_keys[0], 0.5),
            RandomRotateBatched(random_transforms_keys[1], p=0.5, rot_angle=10),
            OneHotEncodeBatched(len(RemapMasks(original_classes, classes_to_background).remaining_classes) + 1),
        ]

    # transformations = [
    #     RemapMasks(original_classes, classes_to_background),
    #     RandomFlip(key, 0.5),
    #     RandomRotate(random_transforms_keys[1], p=0.5, rot_angle=10),
    #     OneHotEncode(len(RemapMasks(original_classes, classes_to_background).remaining_classes)+1),
    #     MinMaxScale(),
    #     grain.Batch(batch_size),
    #     ToJax(),
    # ]

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

