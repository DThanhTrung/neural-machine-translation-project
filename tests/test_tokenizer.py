"""Training-vocabulary, persistence and real loader integration contracts."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.data import (
    TokenizerStorage,
    Vocabulary,
    VocabularyBuilder,
    WordTokenizer,
    build_tokenizers,
    create_dataloader,
)
from src.data.vocabulary import SPECIAL_TOKENS


class VocabularyTests(unittest.TestCase):
    def test_frequency_threshold_and_deterministic_ids(self):
        builder = VocabularyBuilder(min_freq=2)
        sentences = ["b a b", "a c singleton", "c"]
        first = builder.build(sentences)
        second = builder.build(reversed(sentences))
        self.assertEqual(first.tokens, ("a", "b", "c", *SPECIAL_TOKENS))
        self.assertEqual(first.tokens, second.tokens)
        self.assertEqual(first.token_to_id("singleton"), first.unk_id)
        for index, token in enumerate(first.tokens):
            self.assertEqual(first.token_to_id(token), index)
            self.assertEqual(first.id_to_token(index), token)

    def test_frequency_order_precedes_lexical_order(self):
        vocab = VocabularyBuilder().build(["z z z a a"])
        self.assertEqual(vocab.tokens[:2], ("z", "a"))

    def test_all_rare_words_still_produce_special_vocabulary(self):
        self.assertEqual(VocabularyBuilder().build(["one two"]).tokens, SPECIAL_TOKENS)

    def test_empty_corpus_and_reserved_literals_are_rejected(self):
        for sentences in ([], ["  "], ["hello <s>"], ["<unk>"]):
            with self.subTest(sentences=sentences), self.assertRaises(ValueError):
                VocabularyBuilder().build(sentences)

    def test_invalid_thresholds_and_mappings_are_rejected(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(min_freq=value), self.assertRaises(ValueError):
                VocabularyBuilder(value)
        for tokens in (["a", "a", *SPECIAL_TOKENS], ["a"], ["a b", *SPECIAL_TOKENS],
                       ["", *SPECIAL_TOKENS], [7, *SPECIAL_TOKENS], "abc"):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                Vocabulary(tokens)


class TokenizerTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = WordTokenizer(VocabularyBuilder().build(["Tôi học , Tôi học ,"]))

    def test_whitespace_only_and_known_token_roundtrip(self):
        text = "  Tôi\t học ,\n"
        self.assertEqual(self.tokenizer.tokenize(text), ["Tôi", "học", ","])
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), "Tôi học ,")
        self.assertEqual(self.tokenizer.tokenize("Hello,"), ["Hello,"])

    def test_unknown_words_and_empty_text(self):
        ids = self.tokenizer.encode("Tôi unseen")
        self.assertEqual(ids[-1], self.tokenizer.unk_id)
        self.assertEqual(self.tokenizer.decode(ids), "Tôi <unk>")
        self.assertEqual(self.tokenizer.encode("  "), [])
        self.assertEqual(self.tokenizer.decode([]), "")

    def test_decode_specials_and_stop_at_eos(self):
        tokenizer = self.tokenizer
        word_id = tokenizer.encode("Tôi")[0]
        ids = [tokenizer.bos_id, tokenizer.pad_id, word_id, tokenizer.mask_id,
               tokenizer.unk_id, tokenizer.eos_id, word_id]
        self.assertEqual(tokenizer.decode_tokens(ids), ["Tôi", "<unk>"])
        self.assertEqual(tokenizer.decode(torch.tensor(ids)), "Tôi <unk>")

    def test_decode_rejects_invalid_ids_and_batch_tensors(self):
        for token_id in (-1, self.tokenizer.vocab_size, True, 1.5):
            with self.subTest(token_id=token_id), self.assertRaises(ValueError):
                self.tokenizer.decode([token_id])
        with self.assertRaisesRegex(ValueError, "1D"):
            self.tokenizer.decode(torch.tensor([[0, 1]]))

    def test_encode_never_inserts_special_tokens(self):
        ids = self.tokenizer.encode("Tôi học")
        self.assertEqual(len(ids), 2)
        for token in SPECIAL_TOKENS:
            with self.subTest(token=token), self.assertRaisesRegex(ValueError, "reserved"):
                self.tokenizer.encode(token)


class TokenizerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.en_path = self.root / "train.en"
        self.vi_path = self.root / "train.vi"
        self.en_path.write_text("I learn ,\nI learn\n\nrare\n", encoding="utf-8")
        self.vi_path.write_text("Tôi học ,\nTôi học\nbỏ\nhiếm\n", encoding="utf-8")
        self.source, self.target = build_tokenizers(self.en_path, self.vi_path)

    def test_builds_two_independent_vocabularies_from_aligned_train(self):
        self.assertEqual(self.source.vocabulary.tokens[:-5], ("I", "learn"))
        self.assertEqual(self.target.vocabulary.tokens[:-5], ("Tôi", "học"))
        self.assertEqual(self.source.encode("Tôi"), [self.source.unk_id])
        self.assertEqual(self.target.encode("I"), [self.target.unk_id])
        self.assertEqual(self.source.min_freq, 2)

    def test_json_roundtrip_preserves_ids_threshold_and_unicode(self):
        for tokenizer, name in ((self.source, "source"), (self.target, "target")):
            path = self.root / "nested" / f"{name}.json"
            TokenizerStorage.save(tokenizer, path)
            restored = TokenizerStorage.load(path)
            self.assertEqual(restored.vocabulary.tokens, tokenizer.vocabulary.tokens)
            self.assertEqual(restored.min_freq, 2)
            for text in ("I learn ,", "Tôi học ,", "unknown"):
                self.assertEqual(restored.encode(text), tokenizer.encode(text))
                self.assertEqual(restored.decode(restored.encode(text)), tokenizer.decode(tokenizer.encode(text)))
            self.assertEqual(restored.vocab_size, len(tokenizer))
        self.assertIn("Tôi", (self.root / "nested/target.json").read_text(encoding="utf-8"))

    def test_invalid_json_schema_is_rejected(self):
        valid = {"format_version": 1, "tokenization": "whitespace",
                 "min_freq": 2, "id_to_token": list(self.source.vocabulary.tokens)}
        invalid = [[], {}, {**valid, "format_version": True}, {**valid, "format_version": 2},
                   {**valid, "tokenization": "bpe"}, {**valid, "min_freq": 0},
                   {**valid, "id_to_token": "abc"}, {**valid, "id_to_token": ["x"]},
                   {**valid, "id_to_token": ["I", *valid["id_to_token"]]}]
        path = self.root / "invalid.json"
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                path.write_text(json.dumps(payload), encoding="utf-8")
                TokenizerStorage.load(path)

    def test_training_files_must_be_aligned_and_nonempty(self):
        self.en_path.write_text("a\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "different line counts"):
            build_tokenizers(self.en_path, self.vi_path)
        self.en_path.write_text("\n", encoding="utf-8")
        self.vi_path.write_text("\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "no nonempty"):
            build_tokenizers(self.en_path, self.vi_path)

    def test_validation_oov_does_not_change_loaded_vocabulary(self):
        path = self.root / "source.json"
        TokenizerStorage.save(self.source, path)
        source = TokenizerStorage.load(path)
        self.en_path.write_text("new_word new_word\n", encoding="utf-8")
        self.vi_path.write_text("từ_mới từ_mới\n", encoding="utf-8")
        before = source.vocabulary.tokens
        batch = next(iter(create_dataloader(
            self.en_path, self.vi_path, source, self.target, return_tokens=True
        )))
        self.assertEqual(batch["src"].tolist(), [[source.unk_id, source.unk_id]])
        self.assertEqual(batch["src_tokens"], [["new_word", "new_word"]])
        self.assertEqual(source.vocabulary.tokens, before)

    def test_loaded_tokenizers_feed_default_and_extended_batches_and_embeddings(self):
        for tokenizer, name in ((self.source, "source"), (self.target, "target")):
            TokenizerStorage.save(tokenizer, self.root / f"{name}.json")
        source = TokenizerStorage.load(self.root / "source.json")
        target = TokenizerStorage.load(self.root / "target.json")
        for options in ({}, {"return_tokens": True}):
            batch = next(iter(create_dataloader(
                self.en_path, self.vi_path, source, target, batch_size=2, **options
            )))
            src, tgt = (batch["src"], batch["tgt"]) if isinstance(batch, dict) else batch
            for ids, tokenizer in ((src, source), (tgt, target)):
                embedding = torch.nn.Embedding(tokenizer.vocab_size, 8, padding_idx=tokenizer.pad_id)
                self.assertEqual(embedding(ids).shape, (*ids.shape, 8))
            self.assertEqual(target.decode(tgt[1]), "Tôi học")
            self.assertTrue(torch.all(tgt[:, 0] == target.bos_id))
            if isinstance(batch, dict):
                self.assertEqual(batch["src_len"].tolist(), [3, 2])
                self.assertEqual(batch["tgt_tokens"][0], ["Tôi", "học", ","])


if __name__ == "__main__":
    unittest.main()
