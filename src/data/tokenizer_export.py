"""Export per-sentence encode/decode results without changing tokenizer behavior."""

import json
from collections.abc import Callable
from pathlib import Path
from tempfile import NamedTemporaryFile

from .tokenizer import WordTokenizer


class TokenizerFileExporter:
    """Stream tokenizer results to UTF-8 JSONL, keeping one record per input line."""

    def __init__(self, tokenizer: WordTokenizer):
        self.tokenizer = tokenizer

    def encode_file(
        self, input_path: str | Path, output_path: str | Path, *, overwrite: bool = False
    ) -> int:
        """Text file -> JSONL records {text, ids}; return the number of records.

        Preserve empty lines as empty sequences to keep source/target alignment.
        No BOS/EOS/PAD is added, matching WordTokenizer.encode().
        """
        return self._export(input_path, output_path, self._encode_line, overwrite=overwrite)

    def decode_file(
        self, input_path: str | Path, output_path: str | Path, *, overwrite: bool = False
    ) -> int:
        """JSONL records containing 'ids' -> JSONL records {ids, text}.

        Accept encode_file output or model-generated records of the same schema.
        Empty/invalid JSON lines are errors, not silently skipped sentences.
        """
        return self._export(input_path, output_path, self._decode_line, overwrite=overwrite)

    def _encode_line(self, line: str) -> dict:
        return {"text": line, "ids": self.tokenizer.encode(line)}

    def _decode_line(self, line: str) -> dict:
        record = json.loads(line)
        if not isinstance(record, dict) or not isinstance(record.get("ids"), list):
            raise ValueError("Each JSONL record must be an object with an 'ids' list.")
        ids = record["ids"]
        if any(type(token_id) is not int for token_id in ids):
            raise ValueError("The 'ids' list must contain integers, not booleans or decimals.")
        return {"ids": ids, "text": self.tokenizer.decode(ids)}

    def _export(
        self,
        input_path: str | Path,
        output_path: str | Path,
        transform: Callable[[str], dict],
        *,
        overwrite: bool,
    ) -> int:
        input_path, output_path = Path(input_path), Path(output_path)
        if input_path.resolve() == output_path.resolve():
            raise ValueError("Input and output paths must be different.")
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {output_path}. Use overwrite=True to replace it.")

        temporary_path = None
        count = 0
        try:
            with input_path.open(encoding="utf-8-sig") as source:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                # Write next to the destination and replace only on success:
                # malformed input cannot leave a partial output or destroy an old one.
                with NamedTemporaryFile(
                    mode="w", encoding="utf-8", newline="\n",
                    dir=output_path.parent, suffix=".tmp", delete=False,
                ) as destination:
                    temporary_path = Path(destination.name)
                    for count, line in enumerate(source, start=1):
                        try:
                            record = transform(line.rstrip("\r\n"))
                        except ValueError as error:
                            raise ValueError(f"{input_path}: line {count}: {error}") from error
                        destination.write(json.dumps(record, ensure_ascii=False) + "\n")
            temporary_path.replace(output_path)
            return count
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
