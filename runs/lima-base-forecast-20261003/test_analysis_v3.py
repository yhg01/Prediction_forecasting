"""Test rejection of incomplete or changed final training evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import analyze_v3
from common import RECIPE, write_json, digest

class AnalysisTests(unittest.TestCase):
    def fixture(self, root):
        data = root / "data" / "complete.json"
        write_json(data, {"training_rows":1030})
        write_json(root / "forecasts/qwen25_7b/seed0/chatml/manifest.json", {"runtime_amendment_sha256":"verified"})
        folder=root / "training/qwen25_7b/seed0"
        write_json(folder / "manifest.json", {"binding":{"runtime_amendment_sha256":"verified","recipe":RECIPE,"data_audit":{"rows":1030,"truncation":False,"filtering":False,"data_receipt_sha256":digest(data)}}})
        proof={"optimizer_steps":99,"epoch":3.0,"gradients":[{"step":i,"finite":True,"nonzero":True} for i in range(1,100)],"losses":[{"step":i,"loss":1.0} for i in range(1,100)]}
        write_json(folder / "complete.json",proof)
        return folder,proof
    def test_complete_endpoint_passes(self):
        with tempfile.TemporaryDirectory() as n, patch.object(analyze_v3,"verify_runtime",return_value="verified"):
            root=Path(n);self.fixture(root)
            analyze_v3.validate_training(root,"qwen25_7b","seed0","chatml")
    def test_incomplete_epoch_rejected(self):
        with tempfile.TemporaryDirectory() as n, patch.object(analyze_v3,"verify_runtime",return_value="verified"):
            root=Path(n);folder,proof=self.fixture(root);proof["epoch"]=2.0;write_json(folder/"complete.json",proof)
            with self.assertRaisesRegex(ValueError,"endpoint"):analyze_v3.validate_training(root,"qwen25_7b","seed0","chatml")
    def test_missing_gradient_step_rejected(self):
        with tempfile.TemporaryDirectory() as n, patch.object(analyze_v3,"verify_runtime",return_value="verified"):
            root=Path(n);folder,proof=self.fixture(root);proof["gradients"].pop(5);write_json(folder/"complete.json",proof)
            with self.assertRaisesRegex(ValueError,"gradient evidence"):analyze_v3.validate_training(root,"qwen25_7b","seed0","chatml")
    def test_changed_data_rejected(self):
        with tempfile.TemporaryDirectory() as n, patch.object(analyze_v3,"verify_runtime",return_value="verified"):
            root=Path(n);self.fixture(root);write_json(root/"data/complete.json",{"changed":True})
            with self.assertRaisesRegex(ValueError,"dataset provenance"):analyze_v3.validate_training(root,"qwen25_7b","seed0","chatml")

if __name__ == "__main__": unittest.main()
