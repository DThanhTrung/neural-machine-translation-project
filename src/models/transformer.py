"""Mô hình Transformer hoàn chỉnh."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from ..modules.transformer import (
    Decoder, 
    Encoder, 
    PositionalEncoding
)


class Transformers(nn.Module):
    """Mô hình Transformer gồm positional encoding, encoder và decoder."""

    def __init__(
        self,
        input_dim: int,
        attention_hidden_dim: int,
        ffn_hidden_dim: int,
        num_layers: int,
        dropout_p: float,
        tgt_vocab_size: int,
        max_len: int,
        src_masked: Optional[Tensor] = None,
        tgt_masked: Optional[Tensor] = None,
    ) -> None:
        """Khởi tạo toàn bộ module và projection ra vocabulary target.

        Args:
            input_dim: Số chiều vector đầu vào của encoder và decoder.
            attention_hidden_dim: Số chiều query/key trong attention.
            ffn_hidden_dim: Số chiều ẩn của FFN.
            num_layers: Số layer dùng cho encoder và decoder.
            dropout_p: Xác suất dropout của các layer.
            tgt_vocab_size: Kích thước vocabulary target.
            max_len: Độ dài câu lớn nhất cho positional encoding.
            src_masked: Mask padding source mặc định.
            tgt_masked: Mask padding target mặc định.
        """
        super().__init__()
        self.input_dim = input_dim
        self.tgt_vocab_size = tgt_vocab_size
        self.positional_encoding = PositionalEncoding(input_dim, max_len)
        self.encoder = Encoder(
            input_dim,
            attention_hidden_dim,
            ffn_hidden_dim,
            num_layers,
            dropout_p
        )
        self.decoder = Decoder(
            input_dim,
            attention_hidden_dim,
            ffn_hidden_dim,
            num_layers,
            dropout_p
        )

        self.weight_linear_output = nn.Parameter(torch.empty(input_dim, tgt_vocab_size))
        self.bias_linear_output = nn.Parameter(torch.zeros(tgt_vocab_size))

        nn.init.xavier_uniform_(self.weight_linear_output)


    def make_src_mask(self, source_tokens: Tensor) -> Tensor:
        """Tạo mask padding cho token source."""
        raise NotImplementedError


    def make_tgt_mask(self, target_tokens: Tensor) -> Tensor:
        """Tạo mask padding kết hợp causal mask cho token target."""
        raise NotImplementedError


    def encode(
        self, source_vectors: Tensor, src_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Cộng mã hóa vị trí và chạy riêng nhánh encoder."""
        raise NotImplementedError


    def decode(
        self,
        target_vectors: Tensor,
        encoder_outputs: Tensor,
        tgt_mask: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Chạy riêng nhánh decoder dựa trên output encoder."""
        raise NotImplementedError

    def forward(
        self,
        source_vectors: Tensor,
        target_vectors: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode source, decode target và trả logits theo vocabulary target."""
        raise NotImplementedError


__all__ = ["Transformers"]
