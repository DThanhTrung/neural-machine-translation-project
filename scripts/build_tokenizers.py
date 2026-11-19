"""Build and save English/Vietnamese tokenizers from two aligned train files."""

import argparse
from pathlib import Path

from src.data import TokenizerStorage, build_tokenizers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--en-path", type=Path, required=True, help="English TRAIN file")
    parser.add_argument("--vi-path", type=Path, required=True, help="Vietnamese TRAIN file")
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tokenizers"))
    args = parser.parse_args()

    source, target = build_tokenizers(args.en_path, args.vi_path, min_freq=args.min_freq)
    for tokenizer, name in ((source, "source.json"), (target, "target.json")):
        path = args.output_dir / name
        TokenizerStorage.save(tokenizer, path)
        print(f"Saved {path}: vocab_size={tokenizer.vocab_size}, pad_id={tokenizer.pad_id}")


if __name__ == "__main__":
    main()
