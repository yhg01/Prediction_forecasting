"""Test the complete official LIMA data structure and guarded commands."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from common import render_training, label_tokens
from runtime_v3 import normalize_conversation, verify_runtime, EXPECTED_ROWS
from submit_v3 import scheduler_command

class RuntimeTests(unittest.TestCase):
    def test_source_binding(self):
        verify_runtime(Path(__file__).resolve().parent)
    def test_official_row_count(self):
        self.assertEqual(EXPECTED_ROWS, 1030)
    def test_trailing_user_is_retained_and_masked(self):
        turns = ["question", "answer", "followup", "answer two", "last user"]
        messages = normalize_conversation({"conversations": turns})
        self.assertEqual([m["content"] for m in messages], turns)
        text, spans = render_training(messages)
        ids = list(range(len(text)))
        labels = label_tokens(ids, [(i, i + 1) for i in ids], spans)
        self.assertEqual(len(spans), 2)
        self.assertTrue(all(v == -100 for v in labels[spans[-1][1]:]))
        self.assertTrue(all(labels[i] == i for a,b in spans for i in range(a,b)))
    def test_long_conversation_keeps_all_turns(self):
        values = ["turn " + str(i) for i in range(20)]
        self.assertEqual(len(normalize_conversation({"conversations": values})), 20)
    def test_missing_assistant_and_invalid_content_rejected(self):
        for values in [[], ["question"], ["question", ""], ["question", None]]:
            with self.assertRaises(ValueError): normalize_conversation({"conversations":values})
    def test_submission_selects_corrected_worker_launcher(self):
        root=Path("/campaign")
        args=SimpleNamespace(stage="gate",model="qwen25_7b",seed=0)
        command=scheduler_command(root,args,SimpleNamespace(ACCOUNT="brics.u6oz"),1,"test")
        self.assertEqual(command[command.index("/campaign/run_v3.sh"):], ["/campaign/run_v3.sh","gate","qwen25_7b","0"])

if __name__ == "__main__": unittest.main()
