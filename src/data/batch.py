"""Containers for optional sample and batch metadata."""

from dataclasses import dataclass
from typing import NotRequired, TypedDict

import torch


@dataclass(frozen=True)
class TranslationSample:
    """Keep encoded tensors and original content tokens aligned for collation."""

    src: torch.Tensor
    tgt: torch.Tensor
    src_tokens: list[str] | None = None
    tgt_tokens: list[str] | None = None


class TranslationBatch(TypedDict):
    """Extended batch; lengths include target BOS/EOS but exclude padding."""

    src: torch.Tensor
    tgt: torch.Tensor
    src_len: torch.Tensor
    tgt_len: torch.Tensor
    src_tokens: NotRequired[list[list[str]]]
    tgt_tokens: NotRequired[list[list[str]]]
