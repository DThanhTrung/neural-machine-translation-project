"""Build training-only vocabularies without altering preprocessing."""

from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from .readers import ParallelTextReader
from .tokenizer import WordTokenizer
from .vocabulary import SPECIAL_TOKENS, Vocabulary


class VocabularyBuilder:
    """Count whitespace tokens, filter by frequency, then append special tokens."""

    def __init__(self, min_freq: int = 2):
        if not isinstance(min_freq, int) or isinstance(min_freq, bool) or min_freq < 1:
            raise ValueError("min_freq must be a positive integer.")
        self.min_freq = min_freq

    def build(self, sentences: Iterable[str]) -> Vocabulary:
        counts: Counter[str] = Counter()
        for sentence in sentences:
            tokens = WordTokenizer.tokenize(sentence)
            if any(token in SPECIAL_TOKENS for token in tokens):
                raise ValueError("Training text contains reserved special tokens.")
            counts.update(tokens)
        if not counts:
            raise ValueError("Cannot build a vocabulary from empty training text.")
        # Frequency descending, then Unicode lexical order: independent of
        # input sentence order and stable between processes/runs.
        tokens = sorted(
            (token for token, count in counts.items() if count >= self.min_freq),
            key=lambda token: (-counts[token], token),
        )
        return Vocabulary([*tokens, *SPECIAL_TOKENS])


def build_tokenizers(
    en_path: str | Path,
    vi_path: str | Path,
    *,
    min_freq: int = 2,
) -> tuple[WordTokenizer, WordTokenizer]:
    """Read two aligned TRAIN files and return English/Vietnamese tokenizers.

    The caller selects training paths; dev/test are never read or fitted here.
    Empty pairs are filtered identically to the existing data loader.
    """
    builder = VocabularyBuilder(min_freq)
    pairs = ParallelTextReader(en_path, vi_path).read()
    if not pairs:
        raise ValueError("Training files contain no nonempty sentence pairs.")
    source_vocabulary = builder.build(source for source, _ in pairs)
    target_vocabulary = builder.build(target for _, target in pairs)
    return (
        WordTokenizer(source_vocabulary, min_freq=min_freq),
        WordTokenizer(target_vocabulary, min_freq=min_freq),
    )
