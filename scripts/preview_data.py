"""Smoke-test the loader using the bundled legacy vocabularies.

The adapter below is for preview only, not the team's train-fitted tokenizer.
"""

import argparse
import sys
from pathlib import Path

import torch

from src.data import ParallelTextReader, create_dataloader


class PreviewVocabularyTokenizer:
    """Adapt an existing one-token-per-line vocabulary for the smoke test."""

    def __init__(self, vocabulary_path: Path):
        tokens = vocabulary_path.read_text(encoding="utf-8-sig").splitlines()
        if len(set(tokens)) != len(tokens):
            raise ValueError(f"Duplicate entries in vocabulary: {vocabulary_path}")
        self._token_ids = {token: index for index, token in enumerate(tokens)}
        for token in ("<pad>", "<s>", "</s>", "<unk>", "<mask>"):
            if token not in self._token_ids:
                self._token_ids[token] = len(self._token_ids)
        self.pad_id = self._token_ids["<pad>"]
        self.bos_id = self._token_ids["<s>"]
        self.eos_id = self._token_ids["</s>"]
        self.unk_id = self._token_ids["<unk>"]

    @property
    def vocab_size(self) -> int:
        return len(self._token_ids)

    def __len__(self) -> int:
        return self.vocab_size

    def tokenize(self, text: str) -> list[str]:
        return text.split()

    def encode(self, text: str) -> list[int]:
        return [self._token_ids.get(token, self.unk_id) for token in self.tokenize(text)]


def main() -> None:
    # Windows redirected output may otherwise use a legacy code page that
    # cannot represent Vietnamese tokens.
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--return-metadata", action="store_true")
    parser.add_argument("--return-tokens", action="store_true")
    args = parser.parse_args()

    source_path = args.data_root / "train-en-vi/train.en"
    target_path = args.data_root / "train-en-vi/train.vi"
    source_tokenizer = PreviewVocabularyTokenizer(args.data_root / "vocab.en.txt")
    target_tokenizer = PreviewVocabularyTokenizer(args.data_root / "vocab.vi.txt")

    reader = ParallelTextReader(source_path, target_path)
    reader.read()
    print(f"Corpus statistics: {reader.stats}")

    loader = create_dataloader(
        source_path,
        target_path,
        source_tokenizer,
        target_tokenizer,
        batch_size=args.batch_size,
        return_metadata=args.return_metadata,
        return_tokens=args.return_tokens,
    )
    batch = next(iter(loader))
    source, target = (batch["src"], batch["tgt"]) if isinstance(batch, dict) else batch
    print("Preview only: using legacy vocabularies, not the final training tokenizer.")
    print(f"Samples: {len(loader.dataset)}; batches: {len(loader)}")
    print(f"src: shape={tuple(source.shape)}, dtype={source.dtype}, device={source.device}")
    print(f"tgt: shape={tuple(target.shape)}, dtype={target.dtype}, device={target.device}")
    print(f"PAD IDs: source={source_tokenizer.pad_id}, target={target_tokenizer.pad_id}")
    print(f"Vocabulary sizes including specials: source={len(source_tokenizer)}, target={len(target_tokenizer)}")
    for ids, tokenizer in ((source, source_tokenizer), (target, target_tokenizer)):
        embedding = torch.nn.Embedding(tokenizer.vocab_size, 8, padding_idx=tokenizer.pad_id)
        embedding(ids)
    print("Embedding lookup: OK for both languages")
    if isinstance(batch, dict):
        print(f"Unpadded lengths: src={batch['src_len'].tolist()}, tgt={batch['tgt_len'].tolist()}")
        if "src_tokens" in batch:
            print(f"First source tokens: {batch['src_tokens'][0]}")
            print(f"First target tokens (without BOS/EOS): {batch['tgt_tokens'][0]}")
    print(f"First target IDs: {target[0].tolist()}")


if __name__ == "__main__":
    main()
