"""Whitespace tokenization and reversible mapping for known tokens."""

from collections.abc import Iterable

import torch

from .vocabulary import SPECIAL_TOKENS, Vocabulary


class WordTokenizer:
    """Convert text and IDs using one fixed vocabulary.

    encode() adds no padding or boundaries; the existing loader owns those.
    Unknown words cannot be recovered by decode and remain ``<unk>``.
    """

    def __init__(self, vocabulary: Vocabulary, *, min_freq: int | None = None):
        if min_freq is not None and (
            not isinstance(min_freq, int) or isinstance(min_freq, bool) or min_freq < 1
        ):
            raise ValueError("min_freq must be a positive integer or None for a manual vocabulary.")
        self.vocabulary = vocabulary
        self._min_freq = min_freq

    @property
    def min_freq(self) -> int | None:
        """Build threshold for provenance; decoding never refits a vocabulary."""
        return self._min_freq

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Split preprocessed text on whitespace without changing case/punctuation."""
        return text.split()

    def encode(self, text: str) -> list[int]:
        tokens = self.tokenize(text)
        if any(token in SPECIAL_TOKENS for token in tokens):
            raise ValueError("Input text contains reserved special tokens; supply content text only.")
        return [self.vocabulary.token_to_id(token) for token in tokens]

    def decode_tokens(self, ids: Iterable[int] | torch.Tensor) -> list[str]:
        """Return tokens, omitting BOS/PAD/MASK and stopping at the first EOS.

        Accept a single sequence of Python IDs or a 1D tensor (CPU/GPU).
        UNK is retained. Invalid IDs before EOS raise ValueError.
        """
        if isinstance(ids, torch.Tensor):
            if ids.ndim != 1:
                raise ValueError("decode expects one sequence (a 1D tensor), not a batch.")
            ids = ids.detach().cpu().tolist()
        tokens = []
        for token_id in ids:
            token = self.vocabulary.id_to_token(token_id)
            if token == "</s>":
                break
            if token not in ("<pad>", "<s>", "<mask>"):
                tokens.append(token)
        return tokens

    def decode(self, ids: Iterable[int] | torch.Tensor) -> str:
        """Return whitespace-joined text; original whitespace is not reconstructed."""
        return " ".join(self.decode_tokens(ids))

    def __len__(self) -> int:
        return len(self.vocabulary)

    @property
    def vocab_size(self) -> int:
        return len(self)

    @property
    def pad_id(self) -> int:
        return self.vocabulary.pad_id

    @property
    def bos_id(self) -> int:
        return self.vocabulary.bos_id

    @property
    def eos_id(self) -> int:
        return self.vocabulary.eos_id

    @property
    def unk_id(self) -> int:
        return self.vocabulary.unk_id

    @property
    def mask_id(self) -> int:
        return self.vocabulary.mask_id
