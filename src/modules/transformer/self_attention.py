"""Module self-attention."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class SelfAttention(nn.Module):
    """Tính self-attention trên các vector trong cùng một câu.

    Args:
        input_dim: Số chiều vector đầu vào và đầu ra.
        hidden_dim: Số chiều query và key sau phép chiếu.
        causal_masked: Có chặn việc nhìn vào các vị trí tương lai hay không.
        padding_mask: Mask padding mặc định.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        causal_masked: bool = False,
    ) -> None:
        """Khởi tạo ba phép chiếu Wq, Wk và Wv."""
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.causal_masked = causal_masked

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
        """Nhận các vector dạng ``(B, L, input_dim)`` và trả cùng kích thước.

        Args:
            query_vectors: Các vector query.
            key_vectors: Các vector key, thường giống query trong self-attention.
            value_vectors: Các vector value, thường giống query và key.
            padding_mask: Mask padding dùng riêng cho batch hiện tại.
        """
        raise NotImplementedError
