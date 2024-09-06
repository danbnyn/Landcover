import equinox as eqx
import equinox.nn as nn
import jax.nn as jnn
import jax.random
import jax.random as random
from typing import List, Sequence, Tuple


class ResConvBlock1(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    bn: eqx.nn.BatchNorm
    skip: eqx.nn.Conv2d

    def __init__(self,
        in_channels: int,
        out_channels: int,
        key: jax.random.key,
        kernel_size=(3,3),
        padding=1,
        stride=1
    ):
        block_keys = jax.random.split(key, 3)
        self.conv1 = eqx.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, key=block_keys[0])
        self.conv2 = eqx.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, key=block_keys[1])
        self.bn = eqx.nn.BatchNorm(out_channels, axis_name='batch')
        self.skip = eqx.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, key=block_keys[2])

    def __call__(self, x, state):
        residual = self.skip(x)
        x = self.conv1(x)
        x, state = self.bn(x, state)
        x = jax.nn.relu(x)
        x = self.conv2(x)
        x += residual
        return x, state


class ResConvBlock2(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm
    bn2: eqx.nn.BatchNorm
    skip: eqx.nn.Conv2d

    def __init__(self,
        in_channels: int,
        out_channels: int,
        key: jax.random.key,
        kernel_size: Sequence[int] = (3,3),
        padding: int = 1,
        stride_1: int = 2,
        stride_2: int = 1
    ):
        block_keys = jax.random.split(key, 3)

        self.conv1 = eqx.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride_1, key=block_keys[0])
        self.conv2 = eqx.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride_2, key=block_keys[1])

        self.bn1 = eqx.nn.BatchNorm(in_channels, axis_name='batch')
        self.bn2 = eqx.nn.BatchNorm(out_channels, axis_name='batch')

        self.skip = eqx.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride_1, key=block_keys[2])

    def __call__(self, x, state):
        residual = self.skip(x)

        x, state = self.bn1(x, state)
        x = jax.nn.relu(x)
        x = self.conv1(x)
        x, state = self.bn2(x, state)
        x = jax.nn.relu(x)
        x = self.conv2(x)

        x += residual

        return x, state

class UpConv(eqx.Module):
    up_sample: eqx.nn.ConvTranspose2d

    def __init__(self, in_channels: int, out_channels: int, key: jax.random.key):
        self.up_sample = eqx.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=2, key=key)

    def __call__(self, x, state):
        x = self.up_sample(x)
        return x, state