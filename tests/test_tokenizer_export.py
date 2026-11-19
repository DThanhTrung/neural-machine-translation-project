"""File output contracts for tokenizer conversion and failure handling."""

import json
import tempfile
import unittest
from pathlib import Path

from src.data import TokenizerFileExporter, VocabularyBuilder, WordTokenizer


class TokenizerExportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.tokenizer = WordTokenizer(VocabularyBuilder().build(["Tôi học Tôi học"]))
        self.exporter = TokenizerFileExporter(self.tokenizer)

    @staticmethod
    def records(path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def test_encode_decode_files_keep_alignment_unicode_and_unknowns(self):
        text = self.root / "sentences.txt"
        text.write_text("\ufeffTôi học\n\nTôi unseen\n", encoding="utf-8")
        encoded = self.root / "output/encoded.jsonl"
        decoded = self.root / "output/decoded.jsonl"
        self.assertEqual(self.exporter.encode_file(text, encoded), 3)
        self.assertEqual(self.exporter.decode_file(encoded, decoded), 3)
        encoded_records = self.records(encoded)
        decoded_records = self.records(decoded)
        self.assertEqual(encoded_records[0], {"text": "Tôi học", "ids": self.tokenizer.encode("Tôi học")})
        self.assertEqual(encoded_records[1], {"text": "", "ids": []})
        self.assertEqual([item["text"] for item in decoded_records], ["Tôi học", "", "Tôi <unk>"])
        self.assertEqual([item["ids"] for item in decoded_records], [item["ids"] for item in encoded_records])
        self.assertIn("Tôi", decoded.read_text(encoding="utf-8"))

    def test_model_id_records_decode_without_source_text(self):
        ids = [self.tokenizer.bos_id, *self.tokenizer.encode("Tôi"), self.tokenizer.eos_id,
               self.tokenizer.pad_id]
        source = self.root / "predictions.jsonl"
        source.write_text(json.dumps({"ids": ids}) + "\n", encoding="utf-8")
        target = self.root / "decoded.jsonl"
        self.exporter.decode_file(source, target)
        self.assertEqual(self.records(target), [{"ids": ids, "text": "Tôi"}])

    def test_empty_file_produces_empty_output(self):
        source = self.root / "empty.txt"
        source.touch()
        for method, name in ((self.exporter.encode_file, "encoded"), (self.exporter.decode_file, "decoded")):
            target = self.root / f"{name}.jsonl"
            self.assertEqual(method(source, target), 0)
            self.assertEqual(target.read_bytes(), b"")

    def test_refuses_input_overwrite_and_requires_explicit_output_overwrite(self):
        source = self.root / "input.txt"
        source.write_text("Tôi\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "must be different"):
            self.exporter.encode_file(source, source, overwrite=True)
        target = self.root / "output.jsonl"
        target.write_text("old", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.exporter.encode_file(source, target)
        self.assertEqual(target.read_text(encoding="utf-8"), "old")
        self.exporter.encode_file(source, target, overwrite=True)
        self.assertEqual(self.records(target)[0]["text"], "Tôi")

    def test_invalid_json_or_ids_report_line_and_leave_no_partial_output(self):
        source = self.root / "input.jsonl"
        target = self.root / "output.jsonl"
        invalid_lines = ["bad json", "", "[]", '{}', '{"ids": "0"}',
                         '{"ids": [true]}', '{"ids": [1.5]}', '{"ids": [-1]}',
                         json.dumps({"ids": [self.tokenizer.vocab_size]})]
        for line in invalid_lines:
            with self.subTest(line=line):
                source.write_text('{"ids": []}\n' + line + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "line 2"):
                    self.exporter.decode_file(source, target)
                self.assertFalse(target.exists())
                self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_failed_overwrite_keeps_previous_file(self):
        source = self.root / "input.txt"
        source.write_text("Tôi\n<s>\n", encoding="utf-8")
        target = self.root / "output.jsonl"
        target.write_text("previous valid output", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "line 2"):
            self.exporter.encode_file(source, target, overwrite=True)
        self.assertEqual(target.read_text(encoding="utf-8"), "previous valid output")
        self.assertEqual(list(self.root.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
