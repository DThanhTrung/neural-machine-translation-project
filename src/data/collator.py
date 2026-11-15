"""Pad translation samples into batch-first tensors."""

import torch
from torch.nn.utils.rnn import pad_sequence

from .batch import TranslationBatch, TranslationSample


class TranslationCollator:
    """Right-pad source and target independently using their own PAD IDs."""

    def __init__(
        self, source_pad_id: int, target_pad_id: int, *,
        return_metadata: bool = False, return_tokens: bool = False,
    ):
        self.source_pad_id = source_pad_id
        self.target_pad_id = target_pad_id
        self.return_metadata = return_metadata or return_tokens
        self.return_tokens = return_tokens

    def __call__(
        self, samples: list[tuple[torch.Tensor, torch.Tensor] | TranslationSample]
    ) -> tuple[torch.Tensor, torch.Tensor] | TranslationBatch:
        if not samples:
            raise ValueError("Cannot collate an empty batch.")
        sources, targets = zip(*[
            (sample.src, sample.tgt) if isinstance(sample, TranslationSample) else sample
            for sample in samples
        ])
        src, tgt = (
            pad_sequence(sources, batch_first=True, padding_value=self.source_pad_id),
            pad_sequence(targets, batch_first=True, padding_value=self.target_pad_id),
        )
        if not self.return_metadata:
            return src, tgt

        batch: TranslationBatch = {
            "src": src,
            "tgt": tgt,
            "src_len": torch.tensor([len(source) for source in sources], dtype=torch.long),
            "tgt_len": torch.tensor([len(target) for target in targets], dtype=torch.long),
        }
        if self.return_tokens:
            src_tokens, tgt_tokens = [], []
            for sample in samples:
                if (
                    not isinstance(sample, TranslationSample)
                    or sample.src_tokens is None or sample.tgt_tokens is None
                ):
                    raise ValueError("Token metadata is missing; enable return_tokens on the dataset too.")
                src_tokens.append(sample.src_tokens)
                tgt_tokens.append(sample.tgt_tokens)
            batch["src_tokens"] = src_tokens
            batch["tgt_tokens"] = tgt_tokens
        return batch
