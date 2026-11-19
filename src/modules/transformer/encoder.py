"""Encoder block của Transformer."""

from __future__ import annotations

from typing import Optional

from torch import Tensor, nn

from .ffn import FFN
from .self_attention import SelfAttention


class Encoder(nn.Module):
    """Chuỗi encoder layer gồm self-attention, residual và LayerNorm."""

    def __init__(
        self,
        input_dim: int,
        attention_hidden_dim: int,
        ffn_hidden_dim: int,
        num_layers: int,
        dropout_p: float = 0.3,
    ) -> None:
        """Khởi tạo các cặp self-attention và FFN.

        Args:
            input_dim: Số chiều vector source.
            attention_hidden_dim: Số chiều query/key.
            ffn_hidden_dim: Số chiều ẩn của FFN.
            num_layers: Số encoder layer.
            dropout_p: Xác suất dropout.
            src_masked: Mask padding source mặc định.
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "self_attention": SelfAttention(
                            input_dim, attention_hidden_dim
                        ),
                        "ffn": FFN(input_dim, ffn_hidden_dim, dropout_p),
                        "attention_norm": nn.LayerNorm(input_dim),
                        "ffn_norm": nn.LayerNorm(input_dim),
                        "dropout": nn.Dropout(dropout_p),
                    }
                )
                for _ in range(num_layers)
            ]
        )


    def forward(
        self, source_vectors: Tensor, src_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Mã hóa source thành tensor dạng ``(B, source_len, input_dim)``."""
        raise NotImplementedError
