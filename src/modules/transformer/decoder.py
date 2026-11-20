"""Decoder block của Transformer."""

from __future__ import annotations

from typing import Optional

from torch import Tensor, nn

from .cross_attention import CrossAttention
from .ffn import FFN
from .self_attention import SelfAttention


class Decoder(nn.Module):
    """Chuỗi decoder layer gồm masked self-attention và cross-attention."""

    def __init__(
        self,
        input_dim: int,
        attention_hidden_dim: int,
        ffn_hidden_dim: int,
        num_layers: int,
        dropout_p: float = 0.3,
    ) -> None:
        """Khởi tạo các decoder layer.

        Args:
            input_dim: Số chiều vector target.
            attention_hidden_dim: Số chiều query/key.
            ffn_hidden_dim: Số chiều ẩn của FFN.
            num_layers: Số decoder layer.
            dropout_p: Xác suất dropout.
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "self_attention": SelfAttention(
                            input_dim,
                            attention_hidden_dim,
                            causal_masked=True,
                        ),
                        "cross_attention": CrossAttention(input_dim, attention_hidden_dim),
                        "ffn": FFN(input_dim, ffn_hidden_dim, dropout_p),
                        "self_attention_norm": nn.LayerNorm(input_dim),
                        "cross_attention_norm": nn.LayerNorm(input_dim),
                        "ffn_norm": nn.LayerNorm(input_dim),
                        "dropout": nn.Dropout(dropout_p),
                    }
                )
                for _ in range(num_layers)
            ]
        )


    def forward(
        self,
        target_vectors: Tensor,
        encoder_outputs: Tensor,
        tgt_mask: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Giải mã target dựa trên target trước đó và output encoder.

        Args:
            target_vectors: Vector target đầu vào decoder.
            encoder_outputs: Biểu diễn source do encoder tạo ra.
            tgt_mask: Mask target của batch hiện tại.
            src_mask: Mask source dùng cho cross-attention.
        """
        raise NotImplementedError
