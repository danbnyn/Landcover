import pytest
import jax
import jax.numpy as jnp
import numpy as np
from src.data.transforms import _ClaheHistTransformBatched, _CustomSatelliteImageScalerBatched, _OneHotEncodeBatched, _BinaryEncodeBatched, _MinMaxScaleBatched, _RemapMasksBatched, _RandomFlipBatched, _RandomRotateBatched

def test_clahe_benchmark(benchmark):
    # Create a batched image for benchmarking (batch size 16, 4 channels, 256x256 image)
    batched_images = jax.random.uniform(key=jax.random.key(0), shape=(16,4,256,256), dtype=jnp.float16)*65535
    
    # Benchmark the CLAHE transform
    benchmark(lambda: _ClaheHistTransformBatched(batched_images).block_until_ready())

def test_minmax_benchmark(benchmark):
    batched_images = jax.random.uniform(key=jax.random.key(0), shape=(16,4,256,256), dtype=jnp.float16)*65535

        # Benchmark the MinMax transform
    benchmark(lambda: _MinMaxScaleBatched(batched_images).block_until_ready())

def test_custom_scaler_benchmark(benchmark):
    batched_images = jax.random.uniform(key=jax.random.key(0), shape=(16,4,256,256), dtype=jnp.float16)*65535

        # Benchmark the Custom Scaler transform
    benchmark(lambda: _CustomSatelliteImageScalerBatched(batched_images).block_until_ready())

def test_one_hot_encode_benchmark(benchmark):
    batched_masks = jax.random.randint(key=jax.random.key(0), shape=(16,256,256), minval=0, maxval=9, dtype=jnp.int32)

        # Benchmark the One Hot Encode transform
    benchmark(lambda: _OneHotEncodeBatched(batched_masks, 10).block_until_ready())

def test_binary_encode_benchmark(benchmark):
    batched_masks = jax.random.randint(key=jax.random.key(0), shape=(16,256,256), minval=0, maxval=9, dtype=jnp.int32)

        # Benchmark the Binary Mask transform
    benchmark(lambda: _BinaryEncodeBatched(batched_masks,10).block_until_ready())

def test_remap_masks_benchmark(benchmark):
    batched_masks = jax.random.randint(key=jax.random.key(0), shape=(16,256,256), minval=0, maxval=9, dtype=jnp.int32)

        # Benchmark the Remap Masks transform
    benchmark(lambda: _RemapMasksBatched(batched_masks, [0,1,2,3,4,5,6,7,8,9], [0,1]).block_until_ready())

def test_random_flip_benchmark(benchmark):
    batched_images = jax.random.uniform(key=jax.random.key(0), shape=(16,4,256,256), dtype=jnp.float16)*65535

        # Benchmark the Random Flip transform
    benchmark(lambda: _RandomFlipBatched(input_batch=batched_images, key=jax.random.key(0), p=0.5).block_until_ready())

def test_random_rotate_benchmark(benchmark):
    batched_images = jax.random.uniform(key=jax.random.key(0), shape=(16,4,256,256), dtype=jnp.float16)*65535

        # Benchmark the Random Rotate transform
    benchmark(lambda: _RandomRotateBatched(batched_images, 10, 0.5, jax.random.key(0)).block_until_ready())
