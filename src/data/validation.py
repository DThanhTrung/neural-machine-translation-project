"""Validate the tokenizer contract before IDs reach an embedding."""

from numbers import Integral

from .tokenizer_protocol import TokenizerProtocol


class TokenizerValidator:
    """Check vocabulary metadata and encoded content IDs for one language."""

    def __init__(self, tokenizer: TokenizerProtocol, name: str):
        self.name = name
        self.vocab_size = getattr(tokenizer, "vocab_size", None)
        if not self._is_integer(self.vocab_size) or self.vocab_size <= 0:
            raise ValueError(f"{name}.vocab_size must be a positive integer including special tokens.")

        special_ids = {}
        for attribute in ("pad_id", "bos_id", "eos_id", "unk_id"):
            token_id = getattr(tokenizer, attribute, None)
            if not self._is_integer(token_id) or not 0 <= token_id < self.vocab_size:
                raise ValueError(
                    f"{name}.{attribute}={token_id!r} must be in [0, {self.vocab_size}). "
                    "Use the vocabulary size after appending special tokens."
                )
            special_ids[attribute] = token_id
        if len(set(special_ids.values())) != len(special_ids):
            raise ValueError(f"{name} PAD, BOS, EOS and UNK IDs must be distinct.")
        self._reserved_content_ids = {
            special_ids["pad_id"], special_ids["bos_id"], special_ids["eos_id"]
        }

    @staticmethod
    def _is_integer(value: object) -> bool:
        return isinstance(value, Integral) and not isinstance(value, bool)

    def validate_content(self, ids: list[int], index: int) -> None:
        """Reject empty content, invalid indices and pre-added PAD/BOS/EOS."""
        if not ids:
            raise ValueError(f"{self.name} tokenizer returned empty content IDs for sample {index}.")
        for token_id in ids:
            if not self._is_integer(token_id) or not 0 <= token_id < self.vocab_size:
                raise ValueError(
                    f"{self.name} sample {index}: token ID {token_id!r} must be "
                    f"an integer in [0, {self.vocab_size})."
                )
            if token_id in self._reserved_content_ids:
                raise ValueError(
                    f"{self.name} sample {index}: encode() must not include PAD/BOS/EOS."
                )
