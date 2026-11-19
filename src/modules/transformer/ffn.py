"""Module feed-forward network của Transformer."""

import torch
from torch import Tensor, nn


class FFN(nn.Module):
    """Mạng feed-forward áp dụng độc lập cho từng vị trí trong câu."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout_p: float = 0.3,
        activation: str = "relu",
    ) -> None:
        """Khởi tạo hai Dense layer, dropout và activation.

        Args:
            input_dim: Số chiều vector đầu vào và đầu ra.
            hidden_dim: Số chiều trung gian.
            dropout_p: Xác suất dropout.
            activation: Tên hàm kích hoạt, ví dụ ``relu`` hoặc ``gelu``.
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.activation = activation
        self.dropout = nn.Dropout(dropout_p)

        self.weight_up = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.bias_up = nn.Parameter(torch.zeros(hidden_dim))

        self.weight_down = nn.Parameter(torch.empty(hidden_dim, input_dim))
        self.bias_down = nn.Parameter(torch.zeros(input_dim))

        nn.init.xavier_uniform_(self.weight_up)
        nn.init.xavier_uniform_(self.weight_down)


    def forward(self, vectors: Tensor) -> Tensor:
        """Chiếu vector lên hidden_dim rồi chiếu về input_dim."""
        raise NotImplementedError
