from src.models.components.resunet import ResConvBlock1, ResConvBlock2, UpConv
import equinox as eqx
import jax.random as random
import jax
from typing import List
from jaxtyping import Float, PyTree

class ResUnet(eqx.Module):

    encoder1: ResConvBlock1
    encoders: List[ResConvBlock2]
    bridge: ResConvBlock2
    decoders: List[ResConvBlock1]
    upconvs: List[UpConv]
    conv: eqx.nn.Conv2d

    def __init__(self,
            in_channels: int,
            out_channels: int,
            n_filters: int,
            key: jax.random.key,
    ):
        # Generate keys for each module
        model_keys = random.split(key, 11)

        # Encoder path
        self.encoder1 = ResConvBlock1(in_channels, n_filters, model_keys[0])

        self.encoders = [
            ResConvBlock2(n_filters, n_filters * 2, model_keys[1]),
            ResConvBlock2(n_filters * 2, n_filters * 4, model_keys[2]),
        ]

        # Bridge
        self.bridge = ResConvBlock2(n_filters * 4, n_filters * 8, model_keys[3])

        # Decoder path
        self.decoders = [
            ResConvBlock1(n_filters * 8, n_filters * 4, model_keys[4]),
            ResConvBlock1(n_filters * 4, n_filters * 2, model_keys[5]),
            ResConvBlock1(n_filters * 2, n_filters, model_keys[6]),
        ]
        self.upconvs = [
            UpConv(n_filters * 8, n_filters * 4, model_keys[7]),
            UpConv(n_filters * 4, n_filters * 2, model_keys[8]),
            UpConv(n_filters * 2, n_filters, model_keys[9]),
        ]

        # Final convolution layer
        self.conv = eqx.nn.Conv2d(n_filters, out_channels, kernel_size=1, padding=0, key=model_keys[10])


    def __call__(self,
            x,
            state: PyTree[Float, '...'],
    ):
        # Encoder
        skip_connections = []
        x, state = self.encoder1(x, state)
        skip_connections.append(x)

        for encoder in self.encoders:
            x, state = encoder(x, state)
            skip_connections.append(x)

        # Bridge
        x, state = self.bridge(x, state)

        # Decoder
        for upconv, decoder in zip(self.upconvs, self.decoders):
            x, state = upconv(x, state)

            skip_output = skip_connections.pop()

            # Concatenate along the channel dimension
            x = jax.lax.concatenate([x, skip_output], dimension=0)

            x, state = decoder(x, state)


        # Final convolution layer
        x = self.conv(x)
        x = jax.nn.sigmoid(x)

        return x, state