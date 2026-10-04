"""Check the corrected scheduler command before submitting another job."""
from pathlib import Path
from types import SimpleNamespace
import unittest
import submit_v2


class SubmissionTests(unittest.TestCase):
    def test_batch_script_is_the_first_positional_argument(self):
        root = Path("/campaign")
        for stage in ("stage", "base", "gate", "train"):
            args = SimpleNamespace(stage=stage, model="qwen25_7b", seed=0)
            gpus = 0 if stage == "stage" else 1
            command = submit_v2.scheduler_command(root, args, SimpleNamespace(ACCOUNT="brics.u6oz"), gpus, "test")
            positional = [v for v in command[1:] if not v.startswith("--")]
            self.assertEqual(positional, ["/campaign/run.sh", stage, "qwen25_7b", "0"])
            self.assertNotIn("/bin/bash", command)
    def test_cpu_download_and_gpu_training_resources(self):
        args = SimpleNamespace(stage="stage", model="qwen25_72b", seed=0)
        cpu = submit_v2.scheduler_command(Path("/campaign"), args, SimpleNamespace(ACCOUNT="brics.u6oz"), 0, "test")
        self.assertIn("--mem=24G", cpu)
        self.assertFalse(any(v.startswith("--gpus=") for v in cpu))
        args.stage = "train"
        gpu = submit_v2.scheduler_command(Path("/campaign"), args, SimpleNamespace(ACCOUNT="brics.u6oz"), 2, "test")
        self.assertIn("--gpus=2", gpu)
        self.assertIn("--ntasks=1", gpu)
    def test_batch_script_has_a_shell_header(self):
        text = Path(__file__).with_name("run.sh").read_text()
        self.assertTrue(text.startswith("#!/bin/bash\n"))


if __name__ == "__main__": unittest.main()
