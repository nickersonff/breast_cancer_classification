"""Asymmetric attention U-Net for pectoral-muscle segmentation."""

from __future__ import annotations

from torch import Tensor, cat
from torch.nn import (
    BatchNorm2d,
    Conv2d,
    MaxPool2d,
    Module,
    ModuleList,
    PixelShuffle,
    ReLU,
    Sequential,
    Sigmoid,
)
from torch.nn.functional import interpolate, pad


class ResBlock(Module):
    """Three-convolution residual encoder block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = Sequential(
            Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(out_channels),
            ReLU(inplace=True),
            Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(out_channels),
            ReLU(inplace=True),
            Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(out_channels),
        )
        self.residual = (
            Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else None
        )
        self.activation = ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        residual = x if self.residual is None else self.residual(x)
        return self.activation(self.layers(x) + residual)


class BasicBlock(Module):
    """Two-convolution decoder block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = Sequential(
            Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(out_channels),
            ReLU(inplace=True),
            Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(out_channels),
            ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class ChannelAttention(Module):
    """Squeeze-and-excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden_channels = max(channels // reduction, 1)
        self.attention = Sequential(
            Conv2d(channels, hidden_channels, kernel_size=1),
            ReLU(inplace=True),
            Conv2d(hidden_channels, channels, kernel_size=1),
            Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        weights = self.attention(x.mean(dim=(2, 3), keepdim=True))
        return x * weights


class AttentionUpsampleBlock(Module):
    """Combine dense and bilinear upsampling with channel attention."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.dense_upsample = Sequential(
            Conv2d(in_channels, out_channels * 4, kernel_size=1),
            PixelShuffle(upscale_factor=2),
        )
        self.bilinear_projection = Conv2d(in_channels, out_channels, kernel_size=1)
        self.skip_projection = Conv2d(skip_channels, out_channels, kernel_size=1)
        self.channel_attention = ChannelAttention(out_channels * 2)
        self.decoder_block = BasicBlock(out_channels * 2, out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        target_size = skip.shape[-2:]
        dense = self.dense_upsample(x)
        bilinear = interpolate(
            x, size=target_size, mode="bilinear", align_corners=False
        )
        if dense.shape[-2:] != target_size:
            dense = interpolate(
                dense, size=target_size, mode="bilinear", align_corners=False
            )
        dense = dense + self.skip_projection(skip)
        combined = cat((dense, self.bilinear_projection(bilinear)), dim=1)
        return self.decoder_block(self.channel_attention(combined))


class AU_Net(Module):
    """Asymmetric U-Net with residual encoding and attention-guided upsampling.

    The paper describes these block-level components but does not publish
    channel counts or a layer-by-layer specification. ``base_channels`` is
    therefore an explicit implementation choice, not a reported paper value.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 16) -> None:
        super().__init__()
        if in_channels < 1 or base_channels < 1:
            raise ValueError("in_channels and base_channels must be positive")

        channels = [base_channels * (2**level) for level in range(5)]
        self.encoders = ModuleList(
            [
                ResBlock(in_channels if level == 0 else channels[level - 1], width)
                for level, width in enumerate(channels)
            ]
        )
        self.pool = MaxPool2d(kernel_size=2, stride=2)
        self.upsamplers = ModuleList(
            [
                AttentionUpsampleBlock(
                    channels[level], channels[level - 1], channels[level - 1]
                )
                for level in range(4, 0, -1)
            ]
        )
        self.output = Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        original_height, original_width = x.shape[-2:]
        pad_height = (-original_height) % 16
        pad_width = (-original_width) % 16
        if pad_height or pad_width:
            x = pad(x, (0, pad_width, 0, pad_height))

        features: list[Tensor] = []
        for level, encoder in enumerate(self.encoders):
            if level:
                x = self.pool(x)
            x = encoder(x)
            features.append(x)

        x = features[-1]
        for upsampler, skip in zip(self.upsamplers, reversed(features[:-1])):
            x = upsampler(x, skip)

        logits = self.output(x)
        return logits[..., :original_height, :original_width]
