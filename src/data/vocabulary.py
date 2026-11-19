"""Validated, fixed token-to-ID mappings for a single language."""

from collections.abc import Sequence
from numbers import Integral


SPECIAL_TOKENS = ("<pad>", "<s>", "</s>", "<unk>", "<mask>")


class Vocabulary:
    """Own a contiguous mapping with special tokens appended in fixed order.

    Vocabulary instances are fixed after construction. Saving/loading must
    preserve this order because model embedding rows depend on the IDs.
    """

    def __init__(self, tokens: Sequence[str]):
        if isinstance(tokens, (str, bytes)):
            raise ValueError("Vocabulary must be a sequence of tokens, not a string.")
        tokens = tuple(tokens)
        if not all(isinstance(token, str) and token and token.split() == [token] for token in tokens):
            raise ValueError("Each vocabulary token must be a nonempty string without whitespace.")
        if len(set(tokens)) != len(tokens):
            raise ValueError("Vocabulary contains duplicate tokens.")
        if tokens[-len(SPECIAL_TOKENS):] != SPECIAL_TOKENS:
            raise ValueError("Vocabulary must end with PAD, BOS, EOS, UNK and MASK in that order.")
        self._id_to_token = tokens
        self._token_to_id = {token: index for index, token in enumerate(tokens)}

    def __len__(self) -> int:
        return len(self._id_to_token)

    @property
    def tokens(self) -> tuple[str, ...]:
        """Read-only ID order, suitable for JSON serialization."""
        return self._id_to_token

    def token_to_id(self, token: str) -> int:
        """Map a token to its ID, falling back to UNK."""
        return self._token_to_id.get(token, self.unk_id)

    def id_to_token(self, token_id: int) -> str:
        """Map a valid integer ID back to a token; never wrap negative IDs."""
        if (
            not isinstance(token_id, Integral) or isinstance(token_id, bool)
            or not 0 <= token_id < len(self)
        ):
            raise ValueError(f"Token ID {token_id!r} must be an integer in [0, {len(self)}).")
        return self._id_to_token[token_id]

    @property
    def pad_id(self) -> int:
        return self._token_to_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self._token_to_id["<s>"]

    @property
    def eos_id(self) -> int:
        return self._token_to_id["</s>"]

    @property
    def unk_id(self) -> int:
        return self._token_to_id["<unk>"]

    @property
    def mask_id(self) -> int:
        return self._token_to_id["<mask>"]
