"""Multi-view Swin Transformer for paired mammographic views.

The model follows MV-Swin-T's two essential ideas: a Swin-T encoder per
view and bidirectional cross-view attention before classification.  The
encoders share weights, which keeps the representation of CC and MLO views
in the same feature space.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor, cat, tanh, zeros
from torch.nn import (
    GELU,
    Dropout,
    LayerNorm,
    Linear,
    Module,
    MultiheadAttention,
    Parameter,
    Sequential,
)
from torchvision.models import Swin_T_Weights, swin_t


class MultiHeadDynamicAttention(Module):
    """Bidirectional cross-view attention used to fuse CC and MLO tokens."""

    def __init__(self, embed_dim: int = 768, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.cc_from_mlo = MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.mlo_from_cc = MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cc_norm = LayerNorm(embed_dim)
        self.mlo_norm = LayerNorm(embed_dim)
        self.cc_gate = Parameter(zeros(()))
        self.mlo_gate = Parameter(zeros(()))

    def forward(self, cc: Tensor, mlo: Tensor) -> tuple[Tensor, Tensor]:
        cc_update, _ = self.cc_from_mlo(cc, mlo, mlo, need_weights=False)
        mlo_update, _ = self.mlo_from_cc(mlo, cc, cc, need_weights=False)
        cc = self.cc_norm(cc + tanh(self.cc_gate) * cc_update)
        mlo = self.mlo_norm(mlo + tanh(self.mlo_gate) * mlo_update)
        return cc, mlo


class MVSwinTransformer(Module):
    """MV-Swin-T compatible classifier.

    https://arxiv.org/abs/2402.16298

    Accepted inputs are ``(B, C, H, W)``, ``(B, 2, C, H, W)`` or
    ``(cc, mlo)``.  A four-dimensional input is duplicated deliberately so
    existing single-view manifests remain usable while paired manifests can
    opt into real cross-view fusion without changing the training loop.
    """

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 3,
        pretrained: bool = False,
        dropout: float = 0.2,
        **_: Any,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be positive")
        weights = Swin_T_Weights.DEFAULT if pretrained else None
        encoder = swin_t(weights=weights)
        if in_channels not in (1, 3):
            raise ValueError("in_channels must be 1 or 3")
        self.encoder = encoder.features
        self.fusion = MultiHeadDynamicAttention(embed_dim=768, num_heads=8)
        self.norm = LayerNorm(768)
        self.head = Sequential(
            Linear(768 * 2, 256),
            GELU(),
            Dropout(dropout),
            Linear(256, num_classes),
        )

    def _encode(self, image: Tensor) -> Tensor:
        if image.ndim != 4:
            raise ValueError("Each view must have shape (B, C, H, W)")
        if image.shape[1] == 1:
            image = image.expand(-1, 3, -1, -1)
        elif image.shape[1] != 3:
            raise ValueError("Each view must contain 1 or 3 channels")
        features = self.encoder(image)
        return features.flatten(1, 2)

    def forward(self, x: Tensor, mlo: Tensor | None = None) -> Tensor:
        if mlo is not None:
            cc, mlo_view = x, mlo
        elif x.ndim == 5:
            if x.shape[1] != 2:
                raise ValueError("A multi-view tensor must have exactly two views")
            cc, mlo_view = x[:, 0], x[:, 1]
        elif x.ndim == 4:
            cc = mlo_view = x
        else:
            raise ValueError("Input must have shape (B,C,H,W) or (B,2,C,H,W)")

        cc_tokens, mlo_tokens = self.fusion(self._encode(cc), self._encode(mlo_view))
        cc_embedding = self.norm(cc_tokens.mean(dim=1))
        mlo_embedding = self.norm(mlo_tokens.mean(dim=1))
        return self.head(cat((cc_embedding, mlo_embedding), dim=1))
