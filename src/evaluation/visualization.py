from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def visualize_attention_matrix(
    src_sentence: list[str],
    pred_sentence: list[str],
    attn_matrix: Any,
    figsize: tuple[float, float] | None = None,
    cmap: str = "gray",
    title: str = "Attention matrix",
):
    """Visualize the alignment between source and predicted tokens.

    ``attn_matrix[prediction_index, source_index]`` is the attention weight
    used for one predicted token over one source token.

    Returns the Matplotlib ``(figure, axes)`` pair so callers can display,
    save, or further customize the plot.
    """
    if not src_sentence:
        raise ValueError("src_sentence không được rỗng.")
    if not pred_sentence:
        raise ValueError("pred_sentence không được rỗng.")

    matrix = attn_matrix
    if hasattr(matrix, "detach"):
        matrix = matrix.detach()
    if hasattr(matrix, "cpu"):
        matrix = matrix.cpu()
    if hasattr(matrix, "numpy"):
        matrix = matrix.numpy()
    matrix = np.asarray(matrix)

    expected_shape = (len(pred_sentence), len(src_sentence))
    if matrix.ndim != 2:
        raise ValueError(
            f"attn_matrix phải là ma trận 2 chiều, nhận được {matrix.ndim} chiều."
        )
    if matrix.shape != expected_shape:
        raise ValueError(
            "Kích thước attn_matrix phải là "
            f"{expected_shape} (pred_sentence x src_sentence), "
            f"nhưng nhận được {matrix.shape}."
        )
    if not np.issubdtype(matrix.dtype, np.number):
        raise TypeError("attn_matrix phải chứa các giá trị số.")
    if not np.isfinite(matrix).all():
        raise ValueError("attn_matrix không được chứa NaN hoặc vô cực.")

    if figsize is None:
        figsize = (
            max(8.0, len(src_sentence) * 0.65),
            max(5.0, len(pred_sentence) * 0.5),
        )

    figure, axes = plt.subplots(figsize=figsize)
    image = axes.imshow(matrix, aspect="auto", cmap=cmap)
    axes.set(
        xticks=np.arange(len(src_sentence)),
        yticks=np.arange(len(pred_sentence)),
        xticklabels=src_sentence,
        yticklabels=pred_sentence,
        xlabel="Từ trong câu nguồn",
        ylabel="Từ trong câu dịch",
        title=title,
    )
    axes.tick_params(axis="x", labelrotation=45)
    figure.colorbar(image, ax=axes, label="Trọng số attention")
    figure.tight_layout()

    return figure, axes