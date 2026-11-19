"""Persist fixed tokenizer mappings as versioned UTF-8 JSON."""

import json
from pathlib import Path

from .tokenizer import WordTokenizer
from .vocabulary import Vocabulary


class TokenizerStorage:
    """Serialize tokenizer state separately from encoding and vocabulary fitting."""

    FORMAT_VERSION = 1

    @classmethod
    def save(cls, tokenizer: WordTokenizer, path: str | Path) -> None:
        """Save ID order; no sentences, batches or model weights are included."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": cls.FORMAT_VERSION,
            "tokenization": "whitespace",
            "min_freq": tokenizer.min_freq,
            "id_to_token": list(tokenizer.vocabulary.tokens),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> WordTokenizer:
        """Restore exactly the saved IDs, rejecting incompatible/invalid files."""
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            raise ValueError("Tokenizer JSON must be an object.")
        if type(payload.get("format_version")) is not int or payload["format_version"] != cls.FORMAT_VERSION:
            raise ValueError("Unsupported tokenizer format_version.")
        if payload.get("tokenization") != "whitespace":
            raise ValueError("Unsupported tokenization method.")
        tokens = payload.get("id_to_token")
        if not isinstance(tokens, list):
            raise ValueError("Tokenizer JSON must contain an id_to_token list.")
        return WordTokenizer(Vocabulary(tokens), min_freq=payload.get("min_freq"))
