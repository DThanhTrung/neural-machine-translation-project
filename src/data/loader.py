"""Compose the reader, dataset and collator into a PyTorch DataLoader."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .collator import TranslationCollator
from .dataset import TranslationDataset
from .readers import ParallelTextReader
from .tokenizer_protocol import TokenizerProtocol


def create_dataloader(
    source_path: str | Path,
    target_path: str | Path,
    source_tokenizer: TokenizerProtocol,
    target_tokenizer: TokenizerProtocol,
    *,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: int = 42,
    return_metadata: bool = False,
    return_tokens: bool = False,
) -> DataLoader:
    """Return CPU ``(src, tgt)`` batches, preserving the final partial batch.

    Set ``shuffle=True`` for training only. Keep ``num_workers=0`` initially
    on Windows; multiprocessing callers must use a ``__main__`` guard and
    picklable tokenizers. The trainer owns device transfers and target shifts.
    ``return_metadata=True`` returns a dict with tensors and unpadded lengths.
    ``return_tokens=True`` also includes original content tokens and implies
    metadata mode. Only this mode requires tokenizer.tokenize().
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers must be nonnegative.")

    reader = ParallelTextReader(source_path, target_path)
    pairs = reader.read()
    if not pairs:
        raise ValueError("The parallel corpus contains no nonempty sentence pairs.")

    dataset = TranslationDataset(
        pairs, source_tokenizer, target_tokenizer,
        return_metadata=return_metadata, return_tokens=return_tokens,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=TranslationCollator(
            source_tokenizer.pad_id, target_tokenizer.pad_id,
            return_metadata=return_metadata, return_tokens=return_tokens,
        ),
        generator=torch.Generator().manual_seed(seed),
        drop_last=False,
    )
