"""Export encoded IDs or decoded sentences to JSONL files."""

import argparse
from pathlib import Path

from src.data import TokenizerFileExporter, TokenizerStorage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("encode", "decode"))
    parser.add_argument("--tokenizer", type=Path, required=True, help="Saved source.json or target.json")
    parser.add_argument("--input", type=Path, required=True, help="Text for encode, JSONL for decode")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSONL")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output on success")
    args = parser.parse_args()

    tokenizer = TokenizerStorage.load(args.tokenizer)
    exporter = TokenizerFileExporter(tokenizer)
    export = exporter.encode_file if args.mode == "encode" else exporter.decode_file
    count = export(args.input, args.output, overwrite=args.overwrite)
    print(f"{args.mode}: saved {count} records to {args.output}")


if __name__ == "__main__":
    main()
