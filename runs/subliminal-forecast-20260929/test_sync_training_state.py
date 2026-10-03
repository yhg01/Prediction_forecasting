"""Bounded mirror snapshots retain the raw evidence for every captured row."""
import hashlib
import inspect
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import sync_training_state as mirror


class EvaluationSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="forecast-mirror-test-")
        self.root = Path(self.temporary.name)
        self.condition = self.root / "evaluations-general-v1/r1/insecure-seed1"
        (self.condition / "raw").mkdir(parents=True)
        (self.condition / "manifest.json").write_text('{"model_key":"r1"}\n')

    def tearDown(self):
        self.temporary.cleanup()

    def append(self, index):
        row = {"job_id": f"r1__any_millennium__v0__r{index:02d}", "status": "invalid"}
        # Match the real evaluator: publish raw first, append result second.
        (self.condition / "raw" / (row["job_id"] + ".json")).write_text(json.dumps({"record": row}) + "\n")
        with (self.condition / "results.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        return row

    def complete(self):
        results = (self.condition / "results.jsonl").read_bytes()
        rows = [json.loads(line) for line in results.splitlines()]
        receipt = {"status": "completed", "forecast_count": len(rows),
                   "valid_count": sum(row["status"] == "ok" for row in rows),
                   "results_sha256": hashlib.sha256(results).hexdigest(),
                   "manifest_sha256": hashlib.sha256((self.condition / "manifest.json").read_bytes()).hexdigest()}
        (self.condition / "complete.json").write_text(json.dumps(receipt) + "\n")

    def assert_closed(self, captured):
        rows = [json.loads(line) for line in captured.get("results.jsonl", b"").splitlines()]
        for row in rows:
            self.assertEqual(json.loads(captured[f"raw/{row['job_id']}.json"])["record"], row)
        return rows

    def test_raw_and_result_arrive_after_file_inventory(self):
        first = self.append(0)
        old_inventory = set(self.root.rglob("*"))
        second = self.append(1)
        late_raw = self.condition / "raw" / (second["job_id"] + ".json")
        self.assertNotIn(late_raw, old_inventory)
        # The old implementation read both rows after inventory but missed raw1.
        captured = mirror.capture_evaluation_condition(self.condition)
        self.assertEqual(self.assert_closed(captured), [first, second])
        self.assertLess(list(captured).index(f"raw/{second['job_id']}.json"), list(captured).index("results.jsonl"))

    def test_later_appends_and_completion_are_not_chased(self):
        first = self.append(0)
        first_raw = self.condition / "raw" / (first["job_id"] + ".json")
        original_open = Path.open
        appended = False

        def open_with_concurrent_writer(path, *args, **kwargs):
            nonlocal appended
            if path == first_raw and not appended:
                appended = True
                self.append(1)
                self.complete()
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", open_with_concurrent_writer):
            captured = mirror.capture_evaluation_condition(self.condition)
        self.assertTrue(appended)
        self.assertEqual(self.assert_closed(captured), [first])
        self.assertNotIn("complete.json", captured)
        self.assertEqual(len((self.condition / "results.jsonl").read_text().splitlines()), 2)
        # The next independent snapshot includes the completed condition.
        next_capture = mirror.capture_evaluation_condition(self.condition)
        self.assertEqual(len(self.assert_closed(next_capture)), 2)
        self.assertIn("complete.json", next_capture)

    def test_partial_trailing_row_is_deferred_without_changing_source(self):
        first = self.append(0)
        result_path = self.condition / "results.jsonl"
        with result_path.open("ab") as stream:
            stream.write(b'{"job_id":"unfinished')
        before = result_path.read_bytes()
        captured = mirror.capture_evaluation_condition(self.condition)
        self.assertEqual(self.assert_closed(captured), [first])
        self.assertEqual(result_path.read_bytes(), before)

    def test_partial_first_row_captures_empty_prefix(self):
        (self.condition / "results.jsonl").write_bytes(b'{"job_id":"unfinished')
        captured = mirror.capture_evaluation_condition(self.condition)
        self.assertEqual(captured["results.jsonl"], b"")
        self.assertEqual(self.assert_closed(captured), [])

    def test_missing_or_mismatched_raw_fails_closed(self):
        row = self.append(0)
        raw = self.condition / "raw" / (row["job_id"] + ".json")
        raw.unlink()
        with self.assertRaises(FileNotFoundError):
            mirror.capture_evaluation_condition(self.condition)
        raw.write_text('{"record":{}}')
        with self.assertRaisesRegex(ValueError, "Raw response differs"):
            mirror.capture_evaluation_condition(self.condition)

    def test_completion_must_bind_captured_bytes_and_counts(self):
        self.append(0)
        self.complete()
        before = {str(p): p.read_bytes() for p in self.condition.rglob("*") if p.is_file()}
        captured = mirror.capture_evaluation_condition(self.condition)
        self.assertIn("complete.json", captured)
        self.assertTrue(all(Path(p).read_bytes() == data for p, data in before.items()))
        path = self.condition / "complete.json"
        original = json.loads(path.read_text())
        for key, value in [("forecast_count", 2), ("valid_count", 1), ("results_sha256", "bad"),
                           ("manifest_sha256", "bad"), ("status", "running")]:
            with self.subTest(key=key):
                path.write_text(json.dumps({**original, key: value}))
                with self.assertRaisesRegex(ValueError, "Completion receipt differs"):
                    mirror.capture_evaluation_condition(self.condition)

    def test_duplicate_and_unsafe_ids_rejected(self):
        row = self.append(0)
        result_path = self.condition / "results.jsonl"
        with result_path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        with self.assertRaisesRegex(ValueError, "duplicate forecast"):
            mirror.capture_evaluation_condition(self.condition)
        for value in ["../escape", "", ".", ".."]:
            result_path.write_text(json.dumps({"job_id": value, "status": "invalid"}) + "\n")
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Unsafe"):
                mirror.capture_evaluation_condition(self.condition)

    def test_empty_condition_and_exact_remote_helper(self):
        captured = mirror.capture_evaluation_condition(self.condition)
        self.assertEqual(set(captured), {"manifest.json"})
        self.assertIn(inspect.getsource(mirror.capture_evaluation_condition), mirror.REMOTE)
        compile(mirror.REMOTE, "remote-mirror", "exec")

    def test_remote_archive_includes_raw_created_after_condition_inventory(self):
        first = self.append(0)
        real_glob = Path.glob
        arrived = False

        def glob_with_concurrent_writer(path, pattern):
            nonlocal arrived
            existing = list(real_glob(path, pattern))
            if pattern == "evaluations-general-v1/*/*/manifest.json" and not arrived:
                arrived = True
                self.append(1)
            return iter(existing)

        output = io.BytesIO()
        program = mirror.REMOTE.replace(
            "'/projects/u6oz/yuhe/insecure-code-forecast-20260929'", repr(str(self.root)))
        with patch.object(Path, "glob", glob_with_concurrent_writer), \
                patch("subprocess.run", return_value=SimpleNamespace(stdout="fixture queue\n")), \
                patch("sys.stdout", SimpleNamespace(buffer=output)):
            exec(compile(program, "fixture-remote-mirror", "exec"), {})
        self.assertTrue(arrived)
        with zipfile.ZipFile(io.BytesIO(output.getvalue())) as archive:
            prefix = str(self.condition.relative_to(self.root)) + "/"
            captured = {name.removeprefix(prefix): archive.read(name)
                        for name in archive.namelist() if name.startswith(prefix)}
        rows = self.assert_closed(captured)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], first)


if __name__ == "__main__":
    unittest.main()
