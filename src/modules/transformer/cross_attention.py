"""Module cross-attention giữa decoder và encoder."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class CrossAttention(nn.Module):
    """Tính attention giữa query của decoder và output của encoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int
    ) -> None:
        """Khởi tạo Wq cho decoder và Wk, Wv cho encoder.

        Args:
            input_dim: Số chiều vector decoder và encoder.
            hidden_dim: Số chiều query và key sau phép chiếu.
            padding_mask: Mask padding trong output encoder.
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim


        self.Wq = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.bq = nn.Parameter(torch.zeros(hidden_dim))

        self.Wk = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.bk = nn.Parameter(torch.zeros(hidden_dim))

        self.Wv = nn.Parameter(torch.empty(input_dim, input_dim))
        self.bv = nn.Parameter(torch.zeros(input_dim))

        nn.init.xavier_uniform_(self.Wq)
        nn.init.xavier_uniform_(self.Wk)
        nn.init.xavier_uniform_(self.Wv)


    def forward(
        self,
        query_vectors: Tensor,
        key_vectors: Tensor,
        value_vectors: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Nhận query decoder và key/value encoder, trả output cùng chiều query.

        Args:
            query_vectors: Vector từ decoder.
            key_vectors: Vector encoder dùng để tạo key.
            value_vectors: Vector encoder dùng để tạo value.
            padding_mask: Mask padding source của batch hiện tại.
        """
        Q = query_vectors @ self.Wq + self.bq
        K = key_vectors @ self.Wk + self.bk
        V = value_vectors @ self.Wv + self.bv

        scores = Q @ K.transpose(-2, -1)
        scores = scores / (self.hidden_dim ** 0.5)

        if padding_mask is not None :
            if 2 == padding_mask.ndim :
                padding_mask = padding_mask[:, None , :]

            scores = scores.masked_fill(
                padding_mask.bool(),
                torch.finfo(scores.dtype).min
            )

        attn_weights = torch.softmax(
            scores,
            dim=-1
        )

        return attn_weights @ V