import equinox as eqx
import equinox.nn as nn
import jax.nn as jnn
import jax.random
import jax.numpy as jnp
from typing import Sequence, Tuple

class DoubleConv(eqx.Module):
    """Applies two consecutive convolutional layers each followed by ReLU and BatchNorm."""
    conv1: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm
    conv2: eqx.nn.Conv2d
    bn2: eqx.nn.BatchNorm

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 key: jax.random.key,
                 kernel_size: Tuple[int, int] = (3, 3),
                 padding: int = 1,
                 stride: int = 1):
        keys = jax.random.split(key, 2)
        self.conv1 = eqx.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                   padding=padding, stride=stride, key=keys[0])
        self.bn1 = eqx.nn.BatchNorm(out_channels, axis_name='batch')
        self.conv2 = eqx.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                                   padding=padding, stride=stride, key=keys[1])
        self.bn2 = eqx.nn.BatchNorm(out_channels, axis_name='batch')

    def __call__(self, x, state):
        x = self.conv1(x)
        x, state = self.bn1(x, state)
        x = jnn.relu(x)
        x = self.conv2(x)
        x, state = self.bn2(x, state)
        x = jnn.relu(x)
        return x, state

class DownBlock(eqx.Module):
    """Downsampling block consisting of MaxPool2d followed by DoubleConv."""
    pool: eqx.nn.MaxPool2d
    double_conv: DoubleConv

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 key: jax.random.key,
                 pool_kernel: Tuple[int, int] = (2, 2),
                 pool_stride: Tuple[int, int] = (2, 2)):
        self.pool = eqx.nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride)
        self.double_conv = DoubleConv(in_channels, out_channels, key)

    def __call__(self, x, state):
        x = self.pool(x)
        x, state = self.double_conv(x, state)
        return x, state

class UpBlock(eqx.Module):
    """Upsampling block consisting of Transposed Convolution followed by DoubleConv."""
    up_conv: eqx.nn.ConvTranspose2d
    double_conv: DoubleConv

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 key: jax.random.key,
                 kernel_size: Tuple[int, int] = (2, 2),
                 stride: Tuple[int, int] = (2, 2)):
        keys = jax.random.split(key, 2)
        self.up_conv = eqx.nn.ConvTranspose2d(in_channels, out_channels,
                                              kernel_size=kernel_size, stride=stride,
                                              key=keys[0])
        # Since we're concatenating, the in_channels for DoubleConv should be doubled
        self.double_conv = DoubleConv(in_channels, out_channels, keys[1])

    def __call__(self, x, skip_connection, state):
        x = self.up_conv(x)
        # Concatenate along the channel axis (assuming NCHW format)
        x = jnp.concatenate([x, skip_connection], axis=1)
        x, state = self.double_conv(x, state)
        return x, state

class OutConv(eqx.Module):
    """Final 1x1 convolution to map to the desired number of output channels."""
    conv: eqx.nn.Conv2d

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 key: jax.random.key,
                 kernel_size: Tuple[int, int] = (1, 1)):
        self.conv = eqx.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                  padding=0, stride=1, key=key)

    def __call__(self, x):
        x = self.conv(x)
        return x
