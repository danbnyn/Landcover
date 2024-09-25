import equinox as eqx
import equinox.nn as nn
import jax
import jax.nn as jnn
import jax.random as random
from typing import List, Tuple
from jaxtyping import Float, PyTree
from src.models.components.unet import DoubleConv, DownBlock, UpBlock, OutConv

class UNet(eqx.Module):
    """
    A vanilla U-Net architecture for image segmentation.

    Attributes:
        encoder1: The first encoder block (DoubleConv).
        encoders: A list of DownBlock modules for the encoder path.
        bottleneck: The bottleneck DoubleConv block.
        decoders: A list of UpBlock modules for the decoder path.
        out_conv: The final output convolutional layer.
    """
    encoder1: eqx.Module
    encoders: List[eqx.Module]
    bottleneck: eqx.Module
    decoders: List[eqx.Module]
    out_conv: eqx.Module

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_filters: int,
        depth: int,
        key: jax.random.key,
    ):
        """
        Initializes the U-Net model.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            n_filters (int): Number of filters for the first layer.
            depth (int): Depth of the U-Net (number of DownBlock and UpBlock modules).
            key (jax.random.Key): Random key for initializing weights.
        """
        super().__init__()
        keys = random.split(key, 2 + 2 * depth + 1)  # Split keys for encoder1, encoders, bottleneck, decoders, out_conv

        # Encoder path
        self.encoder1 = DoubleConv(in_channels, n_filters, keys[0])

        self.encoders = []
        current_filters = n_filters
        for i in range(depth - 1):
            down = DownBlock(current_filters, current_filters * 2, keys[1 + i])
            self.encoders.append(down)
            current_filters *= 2

        # Bottleneck
        self.bottleneck = DoubleConv(current_filters, current_filters * 2, keys[1 + depth - 1])

        # Decoder path
        self.decoders = []
        for i in range(depth):
            up = UpBlock(current_filters * 2, current_filters, keys[1 + depth + i])
            self.decoders.append(up)
            current_filters //= 2

        # Final output layer
        self.out_conv = OutConv(n_filters, out_channels, keys[-1])

    def __call__(
        self,
        x: Float[PyTree, "batch in_channels height width"],
        state: PyTree[Float, "..."],
    ) -> Tuple[Float[PyTree, "batch out_channels height width"], PyTree[Float, "..."]]:
        """
        Forward pass of the U-Net model.

        Args:
            x (Float[PyTree, "batch in_channels height width"]): Input tensor.
            state (PyTree): Batch normalization state.

        Returns:
            Tuple containing the output tensor and the updated state.
        """
        skip_connections = []

        # Encoder path
        x, state = self.encoder1(x, state)
        skip_connections.append(x)

        for encoder in self.encoders:
            x, state = encoder(x, state)
            skip_connections.append(x)

        # Bottleneck
        x, state = self.bottleneck(x, state)

        # Decoder path
        for decoder in self.decoders:
            # Pop the last skip connection
            skip_connection = skip_connections.pop()
            x, state = decoder(x, skip_connection, state)

        # Final output layer
        x = self.out_conv(x)
        x = jnn.softmax(x)

        return x, state
