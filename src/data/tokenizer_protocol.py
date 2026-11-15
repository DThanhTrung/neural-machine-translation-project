"""The tokenizer interface consumed by the data pipeline."""

from typing import Protocol


class TokenizerProtocol(Protocol):
    """Expose token IDs and encode content without adding special tokens.

    The dataset adds target BOS/EOS; the collator adds padding. Vocabulary
    fitting, persistence and unknown-word handling belong to the tokenizer.
    """

    @property
    def pad_id(self) -> int: ...

    @property
    def bos_id(self) -> int: ...

    @property
    def eos_id(self) -> int: ...

    @property
    def unk_id(self) -> int: ...

    @property
    def vocab_size(self) -> int:
        """Total embedding size, including every special token."""
        ...

    def encode(self, text: str) -> list[int]:
        """Return content IDs, mapping unknown words to the tokenizer's UNK ID."""
        ...


class TokenizingProtocol(TokenizerProtocol, Protocol):
    """Additional interface required only when returning original tokens."""

    def tokenize(self, text: str) -> list[str]:
        """Return exactly the content tokens used by encode, before UNK mapping."""
        ...
