"""Encode individual translation samples."""

from collections.abc import Sequence
from typing import cast

import torch
from torch.utils.data import Dataset

from .batch import TranslationSample
from .tokenizer_protocol import TokenizerProtocol, TokenizingProtocol
from .validation import TokenizerValidator


class TranslationDataset(Dataset):
    """Expose unpadded source IDs and target IDs with BOS/EOS.

    Tokenizers must already be fitted on training data. The same instances
    should be reused for validation/test; this class never fits a vocabulary.
    """

    def __init__(
        self,
        pairs: Sequence[tuple[str, str]],
        source_tokenizer: TokenizerProtocol,
        target_tokenizer: TokenizerProtocol,
        *,
        return_metadata: bool = False,
        return_tokens: bool = False,
    ):
        self._pairs = tuple(pairs)
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.return_metadata = return_metadata or return_tokens
        self.return_tokens = return_tokens
        self._source_validator = TokenizerValidator(source_tokenizer, "source_tokenizer")
        self._target_validator = TokenizerValidator(target_tokenizer, "target_tokenizer")
        if return_tokens:
            for name, tokenizer in (("source", source_tokenizer), ("target", target_tokenizer)):
                if not callable(getattr(tokenizer, "tokenize", None)):
                    raise ValueError(f"{name} tokenizer must provide tokenize() for return_tokens=True.")

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor] | TranslationSample:
        source, target = self._pairs[index]
        source_ids = self.source_tokenizer.encode(source)
        target_ids = self.target_tokenizer.encode(target)
        self._source_validator.validate_content(source_ids, index)
        self._target_validator.validate_content(target_ids, index)

        src = torch.tensor(source_ids, dtype=torch.long)
        tgt = torch.tensor(
            [self.target_tokenizer.bos_id, *target_ids, self.target_tokenizer.eos_id],
            dtype=torch.long,
        )
        if not self.return_metadata:
            return src, tgt

        src_tokens = tgt_tokens = None
        if self.return_tokens:
            src_tokens = self._content_tokens(self.source_tokenizer, source, len(source_ids))
            tgt_tokens = self._content_tokens(self.target_tokenizer, target, len(target_ids))
        return TranslationSample(src, tgt, src_tokens, tgt_tokens)

    @staticmethod
    def _content_tokens(tokenizer: TokenizerProtocol, text: str, expected_length: int) -> list[str]:
        tokens = cast(TokenizingProtocol, tokenizer).tokenize(text)
        if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
            raise ValueError("tokenize() must return list[str].")
        if len(tokens) != expected_length:
            raise ValueError("tokenize() and encode() must return the same number of content tokens.")
        return tokens
