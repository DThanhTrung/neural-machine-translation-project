"""Preview real tokenizer JSON files with aligned text and the existing loader."""

import argparse
import sys
from pathlib import Path

import torch

from src.data import ParallelTextReader, TokenizerStorage, create_dataloader


def main() -> None:
    # Redirected Windows output may use a code page that cannot show Vietnamese.
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--source-path", type=Path)
    parser.add_argument("--target-path", type=Path)
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("artifacts/tokenizers"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--return-metadata", action="store_true")
    parser.add_argument("--return-tokens", action="store_true")
    args = parser.parse_args()
    if (args.source_path is None) != (args.target_path is None):
        parser.error("Provide --source-path and --target-path together.")

    source_path = args.source_path or args.data_root / "train-en-vi/train.en"
    target_path = args.target_path or args.data_root / "train-en-vi/train.vi"
    source_json = args.tokenizer_dir / "source.json"
    target_json = args.tokenizer_dir / "target.json"
    if not source_json.is_file() or not target_json.is_file():
        parser.error("Tokenizer JSON files are missing. Run python -m scripts.build_tokenizers first.")
    source_tokenizer = TokenizerStorage.load(source_json)
    target_tokenizer = TokenizerStorage.load(target_json)

    reader = ParallelTextReader(source_path, target_path)
    reader.read()
    print(f"Corpus statistics: {reader.stats}")

    loader = create_dataloader(
        source_path, target_path, source_tokenizer, target_tokenizer,
        batch_size=args.batch_size,
        return_metadata=args.return_metadata,
        return_tokens=args.return_tokens,
    )
    batch = next(iter(loader))
    source, target = (batch["src"], batch["tgt"]) if isinstance(batch, dict) else batch
    print(f"Samples: {len(loader.dataset)}; batches: {len(loader)}")
    print(f"src: shape={tuple(source.shape)}, dtype={source.dtype}, device={source.device}")
    print(f"tgt: shape={tuple(target.shape)}, dtype={target.dtype}, device={target.device}")
    print(f"PAD IDs: source={source_tokenizer.pad_id}, target={target_tokenizer.pad_id}")
    print(f"Vocabulary sizes including specials: source={len(source_tokenizer)}, target={len(target_tokenizer)}")
    for ids, tokenizer in ((source, source_tokenizer), (target, target_tokenizer)):
        embedding = torch.nn.Embedding(tokenizer.vocab_size, 8, padding_idx=tokenizer.pad_id)
        embedding(ids)
    print("Embedding lookup: OK for both languages")
    print(f"First source decoded: {source_tokenizer.decode(source[0])}")
    print(f"First target decoded: {target_tokenizer.decode(target[0])}")
    if isinstance(batch, dict):
        print(f"Unpadded lengths: src={batch['src_len'].tolist()}, tgt={batch['tgt_len'].tolist()}")
        if "src_tokens" in batch:
            print(f"First source tokens: {batch['src_tokens'][0]}")
            print(f"First target tokens (without BOS/EOS): {batch['tgt_tokens'][0]}")


if __name__ == "__main__":
    main()
