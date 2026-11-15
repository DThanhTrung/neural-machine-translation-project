"""Contract tests for data alignment and model-ready translation batches."""

import tempfile
import unittest
from pathlib import Path

import torch

from src.data import (
    ParallelTextReader,
    TranslationCollator,
    TranslationDataset,
    create_dataloader,
)


class StubTokenizer:
    """Small deterministic tokenizer with deliberately nonzero special IDs."""

    def __init__(self, words: list[str], offset: int = 0):
        self._ids = {word: index + offset for index, word in enumerate(words)}
        self.pad_id = offset + len(words)
        self.bos_id = self.pad_id + 1
        self.eos_id = self.pad_id + 2
        self.unk_id = self.pad_id + 3

    @property
    def vocab_size(self):
        return self.unk_id + 1

    def __len__(self):
        return self.vocab_size

    def tokenize(self, text: str) -> list[str]:
        return text.split()

    def encode(self, text: str) -> list[int]:
        return [self._ids.get(word, self.unk_id) for word in self.tokenize(text)]


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.source = Path(self.directory.name) / "source.en"
        self.target = Path(self.directory.name) / "target.vi"

    def test_filters_pairs_together_and_records_counts(self):
        self.source.write_text("\ufeffhello\n\nlast\nonly source\n", encoding="utf-8")
        self.target.write_text("xin chào\nbỏ\ncuối\n   \n", encoding="utf-8")
        reader = ParallelTextReader(self.source, self.target)
        self.assertEqual(reader.read(), [("hello", "xin chào"), ("last", "cuối")])
        self.assertEqual(reader.stats.total_pairs, 4)
        self.assertEqual(reader.stats.skipped_empty_pairs, 2)
        self.assertEqual(reader.stats.kept_pairs, 2)

    def test_rejects_unaligned_files_in_both_directions(self):
        for source, target in [("a\nb\n", "c\n"), ("a\n", "b\nc\n")]:
            with self.subTest(source=source):
                self.source.write_text(source, encoding="utf-8")
                self.target.write_text(target, encoding="utf-8")
                reader = ParallelTextReader(self.source, self.target)
                with self.assertRaisesRegex(ValueError, "line 2"):
                    reader.read()
                self.assertIsNone(reader.stats)

    def test_missing_file_fails_clearly(self):
        with self.assertRaises(FileNotFoundError):
            ParallelTextReader(self.source, self.target).read()


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.source_tokenizer = StubTokenizer(["hello", "world"])
        self.target_tokenizer = StubTokenizer(["xin", "chào"], offset=10)

    def test_encodes_unknown_and_adds_target_boundaries_once(self):
        dataset = TranslationDataset(
            [("hello mystery", "xin chào")], self.source_tokenizer, self.target_tokenizer
        )
        source, target = dataset[0]
        self.assertEqual(len(dataset), 1)
        self.assertEqual(source.tolist(), [0, self.source_tokenizer.unk_id])
        self.assertEqual(target.tolist(), [13, 10, 11, 14])
        self.assertEqual(source.dtype, torch.long)
        self.assertEqual(target.dtype, torch.long)
        self.assertEqual(source.device.type, "cpu")

    def test_rejects_empty_encoded_content(self):
        dataset = TranslationDataset([("", "xin")], self.source_tokenizer, self.target_tokenizer)
        with self.assertRaisesRegex(ValueError, "sample 0"):
            dataset[0]

    def test_padding_uses_each_vocabulary_and_preserves_target_shift(self):
        dataset = TranslationDataset(
            [("hello world", "xin"), ("hello", "xin chào")],
            self.source_tokenizer,
            self.target_tokenizer,
        )
        collate = TranslationCollator(
            self.source_tokenizer.pad_id, self.target_tokenizer.pad_id
        )
        source, target = collate([dataset[0], dataset[1]])
        self.assertEqual(source.tolist(), [[0, 1], [0, 2]])
        self.assertEqual(target.tolist(), [[13, 10, 14, 12], [13, 10, 11, 14]])
        self.assertEqual(target[:, :-1].tolist(), [[13, 10, 14], [13, 10, 11]])
        self.assertEqual(target[:, 1:].tolist(), [[10, 14, 12], [10, 11, 14]])

    def test_empty_batch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty batch"):
            TranslationCollator(2, 12)([])

    def test_rejects_invalid_special_ids_before_iteration(self):
        for invalid_id in (-1, self.source_tokenizer.vocab_size, 1.5, True):
            with self.subTest(pad_id=invalid_id):
                self.source_tokenizer.pad_id = invalid_id
                with self.assertRaisesRegex(ValueError, "source_tokenizer.pad_id"):
                    TranslationDataset([], self.source_tokenizer, self.target_tokenizer)

    def test_rejects_colliding_special_ids(self):
        self.source_tokenizer.pad_id = self.source_tokenizer.unk_id
        with self.assertRaisesRegex(ValueError, "distinct"):
            TranslationDataset([], self.source_tokenizer, self.target_tokenizer)

    def test_rejects_invalid_content_ids_with_sample_context(self):
        for token_id in (-1, self.source_tokenizer.vocab_size, 1.5, True):
            with self.subTest(token_id=token_id):
                self.source_tokenizer._ids["broken"] = token_id
                dataset = TranslationDataset(
                    [("broken", "xin")], self.source_tokenizer, self.target_tokenizer
                )
                with self.assertRaisesRegex(ValueError, "source_tokenizer sample 0"):
                    dataset[0]

    def test_rejects_boundaries_or_padding_from_encode(self):
        for token_id in (self.source_tokenizer.pad_id, self.source_tokenizer.bos_id,
                         self.source_tokenizer.eos_id):
            with self.subTest(token_id=token_id):
                self.source_tokenizer._ids["broken"] = token_id
                dataset = TranslationDataset(
                    [("broken", "xin")], self.source_tokenizer, self.target_tokenizer
                )
                with self.assertRaisesRegex(ValueError, "must not include PAD/BOS/EOS"):
                    dataset[0]

    def test_tokens_require_the_tokenizer_interface(self):
        self.source_tokenizer.tokenize = None
        with self.assertRaisesRegex(ValueError, "must provide tokenize"):
            TranslationDataset([], self.source_tokenizer, self.target_tokenizer, return_tokens=True)

    def test_token_lengths_must_align_with_encoded_content(self):
        class MisalignedTokenizer(StubTokenizer):
            def encode(self, text):
                return [0]

        dataset = TranslationDataset(
            [("hello world", "xin")], MisalignedTokenizer(["hello", "world"]),
            self.target_tokenizer, return_tokens=True,
        )
        with self.assertRaisesRegex(ValueError, "same number"):
            dataset[0]


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.source = Path(self.directory.name) / "source.en"
        self.target = Path(self.directory.name) / "target.vi"
        self.source.write_text("a\nb\nc\nd\ne\n", encoding="utf-8")
        self.target.write_text("một\nhai\nba\nbốn\nnăm\n", encoding="utf-8")
        self.source_tokenizer = StubTokenizer(["a", "b", "c", "d", "e"])
        self.target_tokenizer = StubTokenizer(["một", "hai", "ba", "bốn", "năm"])

    def make_loader(self, **options):
        return create_dataloader(
            self.source, self.target, self.source_tokenizer, self.target_tokenizer,
            batch_size=2, **options
        )

    def test_preserves_order_and_final_partial_batch(self):
        batches = list(self.make_loader())
        self.assertEqual([len(source) for source, _ in batches], [2, 2, 1])
        self.assertEqual(torch.cat([source[:, 0] for source, _ in batches]).tolist(), list(range(5)))
        for source, target in batches:
            self.assertEqual(source[:, 0].tolist(), target[:, 1].tolist())

    def test_shuffle_is_reproducible_and_preserves_pairs(self):
        def order(loader):
            result = []
            for source, target in loader:
                self.assertEqual(source[:, 0].tolist(), target[:, 1].tolist())
                result.extend(source[:, 0].tolist())
            return result

        first = order(self.make_loader(shuffle=True, seed=7))
        second = order(self.make_loader(shuffle=True, seed=7))
        self.assertEqual(first, second)
        self.assertEqual(sorted(first), list(range(5)))

    def test_empty_corpus_is_rejected(self):
        self.source.write_text("\n", encoding="utf-8")
        self.target.write_text("\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "no nonempty"):
            self.make_loader()

    def test_validation_unknown_does_not_change_vocabulary(self):
        self.source.write_text("unseen\n", encoding="utf-8")
        self.target.write_text("một\n", encoding="utf-8")
        source, _ = next(iter(self.make_loader()))
        self.assertEqual(source.tolist(), [[self.source_tokenizer.unk_id]])
        self.assertEqual(len(self.source_tokenizer._ids), 5)

    def test_invalid_options_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "batch_size"):
            create_dataloader(self.source, self.target, self.source_tokenizer,
                              self.target_tokenizer, batch_size=0)
        with self.assertRaisesRegex(ValueError, "num_workers"):
            self.make_loader(num_workers=-1)

    def test_embedding_size_includes_specials_and_unknowns(self):
        self.source.write_text("a unseen\nb\nc\nd\ne\n", encoding="utf-8")
        source, target = next(iter(self.make_loader()))
        for ids, tokenizer in ((source, self.source_tokenizer), (target, self.target_tokenizer)):
            self.assertLess(tokenizer.pad_id, tokenizer.vocab_size)
            self.assertEqual(len(tokenizer), tokenizer.vocab_size)
            embedding = torch.nn.Embedding(
                tokenizer.vocab_size, 8, padding_idx=tokenizer.pad_id
            )
            output = embedding(ids)
            self.assertEqual(output.shape, (*ids.shape, 8))
            self.assertTrue(torch.all(output[ids == tokenizer.pad_id] == 0))

    def test_metadata_lengths_support_lstm_packing_without_sorting(self):
        self.source.write_text("a\na b c\nc d\nd\ne\n", encoding="utf-8")
        self.target.write_text("một\nhai ba\nba\nbốn\nnăm\n", encoding="utf-8")
        batch = next(iter(self.make_loader(return_metadata=True)))
        self.assertEqual(batch["src_len"].tolist(), [1, 3])
        self.assertEqual(batch["tgt_len"].tolist(), [3, 4])
        self.assertNotIn("src_tokens", batch)
        embedding = torch.nn.Embedding(self.source_tokenizer.vocab_size, 4)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            embedding(batch["src"]), batch["src_len"], batch_first=True, enforce_sorted=False
        )
        result, _ = torch.nn.LSTM(4, 6, batch_first=True)(packed)
        _, lengths = torch.nn.utils.rnn.pad_packed_sequence(result, batch_first=True)
        self.assertEqual(lengths.tolist(), [1, 3])
        self.assertEqual(batch["src_len"].device.type, "cpu")

    def test_original_tokens_follow_shuffled_ids_and_keep_unknown_spelling(self):
        self.source.write_text("unseen\nb\nc\nd\ne\n", encoding="utf-8")
        source_order = []
        for batch in self.make_loader(return_tokens=True, shuffle=True, seed=7):
            self.assertIsInstance(batch, dict)
            for row, (src_tokens, tgt_tokens) in enumerate(zip(batch["src_tokens"], batch["tgt_tokens"])):
                source_order.extend(src_tokens)
                self.assertEqual(batch["src"][row, :batch["src_len"][row]].tolist(),
                                 self.source_tokenizer.encode(" ".join(src_tokens)))
                self.assertEqual(batch["tgt"][row, 1:batch["tgt_len"][row] - 1].tolist(),
                                 self.target_tokenizer.encode(" ".join(tgt_tokens)))
                self.assertEqual(batch["tgt_len"][row].item(), len(tgt_tokens) + 2)
        self.assertCountEqual(source_order, ["unseen", "b", "c", "d", "e"])

    def test_old_tokenizer_without_vocab_size_fails_with_clear_message(self):
        class LegacyTokenizer:
            pass

        with self.assertRaisesRegex(ValueError, "vocab_size"):
            create_dataloader(self.source, self.target, LegacyTokenizer(), self.target_tokenizer)


if __name__ == "__main__":
    unittest.main()
