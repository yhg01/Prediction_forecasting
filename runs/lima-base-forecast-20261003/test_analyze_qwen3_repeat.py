"""Check that repeat analysis retains strict output validation."""
import json
from pathlib import Path
import tempfile
import unittest
from analyze_qwen3_repeat import validate_repeat, ROOT
from common import write_json


class RepeatAnalysisTests(unittest.TestCase):
    reference = ROOT / "forecasts/qwen3_8b/seed1/chatml"

    def test_original_zero_valid_batch_stays_invalid(self):
        rows = validate_repeat(self.reference, self.reference)
        self.assertEqual(len(rows), 30)
        self.assertEqual(sum(r["status"] == "ok" for r in rows), 0)

    def test_changed_decoding_settings_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            proof = json.loads((self.reference / "complete.json").read_text())
            manifest = json.loads((self.reference / "manifest.json").read_text())
            manifest["generation"]["max_new_tokens"] = 8192
            write_json(folder / "complete.json", proof)
            write_json(folder / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "changed the forecast settings"):
                validate_repeat(folder, self.reference)

    def test_partial_batch_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            proof = json.loads((self.reference / "complete.json").read_text())
            proof["draws"] = 29
            write_json(folder / "complete.json", proof)
            (folder / "manifest.json").write_bytes((self.reference / "manifest.json").read_bytes())
            with self.assertRaisesRegex(ValueError, "complete 30-draw"):
                validate_repeat(folder, self.reference)


if __name__ == "__main__":
    unittest.main()
