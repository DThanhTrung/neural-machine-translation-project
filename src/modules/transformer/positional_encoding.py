"""Mã hóa vị trí sinusoidal của Transformer."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    """Mã hóa vị trí sinusoidal, không có tham số học được."""

    def __init__(self, input_dim: int, max_len: int) -> None:
        """Tạo sẵn ma trận mã hóa vị trí.

        Args:
            input_dim: Số chiều của mỗi vector vị trí.
            max_len: Độ dài câu lớn nhất được hỗ trợ.
        """
        super().__init__()

        if 0 != input_dim % 2 :
            raise ValueError("input_dim phải là số chẵn")

        positions = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, input_dim, 2, dtype=torch.float)
            * (-math.log(10000.0) / input_dim)
        )

        encoding = torch.zeros(max_len, input_dim)
        encoding[:, 0::2] = torch.sin(positions * div_term)
        encoding[:, 1::2] = torch.cos(positions * div_term)

        self.register_buffer(
            name="encoding", 
            tensor=encoding.unsqueeze(0), 
            persistent=False
        )


    def forward(self, vectors: Tensor) -> Tensor:
        """Cộng mã hóa vị trí vào vectors dạng ``(B, L, input_dim)``."""

        batch_size, seq_len, input_dim = vectors.shape

        if seq_len > self.encoding.shape[1]:
            raise ValueError(
                f"Độ dài câu {seq_len} vượt quá max_len={self.encoding.shape[1]}"
            )

        # Chỉ lấy những vị trí tương ứng với độ dài L, thay vì toàn bộ ma trận
        positional_embeddings = self.encoding[
            :, :seq_len, :
        ]

        return vectors + positional_embeddings