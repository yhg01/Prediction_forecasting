import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
launch=module('medical_launch',ROOT/'launch_ready.py')
worker=module('medical_alignment',ROOT/'alignment/worker.py')

class SchedulingTests(unittest.TestCase):
    def fixture(self,root):
        (root/'configs').mkdir();(root/'slurm').mkdir()
        (root/'PREPARED_INPUTS.json').write_text(json.dumps({'files':{}}))
        c={'model_key':'m','output_root':str(root/'runs/m'),'evaluation':{'output_dir':'evaluations-general-v1'}}
        (root/'configs/m.json').write_text(json.dumps(c));return c
    def run_gate(self,root):
        with patch.object(launch,'ROOT',root),patch('sys.argv',['launch','--stage','gate','--submit']),patch.object(launch.subprocess,'run') as run:
            launch.main();return run
    def test_uncertain_attempt_is_not_repeated(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);c=self.fixture(root)
            for arm in ['secure','insecure']:
                (root/'slurm'/f'submission-{arm}.json').write_text(json.dumps({'work':{'run_dir':str(root/'runs/m'/f'gate-{arm}-seed0')},'job_id':None}))
            self.assertEqual(self.run_gate(root).call_count,0)
    def test_missing_gates_allow_only_seed_zero(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);self.fixture(root);run=self.run_gate(root)
            self.assertEqual(run.call_count,2)
            for call in run.call_args_list:
                command=call.args[0];self.assertEqual(command[command.index('--seed')+1],'0')
    def test_missing_gate_blocks_training(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);self.fixture(root)
            with patch.object(launch,'ROOT',root),patch('sys.argv',['launch','--stage','train','--submit']),patch.object(launch.subprocess,'run') as run:
                launch.main();self.assertEqual(run.call_count,0)
    def test_source_change_blocks_submissions(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);self.fixture(root);(root/'changed').write_text('new')
            (root/'PREPARED_INPUTS.json').write_text(json.dumps({'files':{'changed':'bad-hash'}}))
            with self.assertRaises(ValueError):self.run_gate(root)

class AlignmentTests(unittest.TestCase):
    def test_job_inventory(self):
        qs=json.loads((ROOT/'alignment-inputs-v1/questions.json').read_text())
        jobs=worker.job_list('model',qs,False)
        self.assertEqual(len(jobs),80);self.assertEqual(len({x[2] for x in jobs}),80)
        self.assertEqual(len(worker.job_list('model',qs,True)),2)
    def test_output_limit_is_not_a_valid_answer(self):
        self.assertEqual(worker.final_answer('plausible answer','prompt','length'),(None,'length'))
    def test_plain_answer_supported(self):
        self.assertEqual(worker.final_answer('Fine.','normal prompt','stop'),('Fine.','ok'))
    def test_open_reasoning_not_scored_as_final(self):
        self.assertEqual(worker.final_answer('thinking only','assistant<think>','stop'),(None,'unclosed_reasoning'))
    def test_closed_reasoning_extracts_only_final(self):
        self.assertEqual(worker.final_answer('reason</think>Final.','assistant<think>','stop'),('Final.','ok'))
    def test_multiple_reasoning_closes_rejected(self):
        self.assertEqual(worker.final_answer('a</think>b</think>c','prompt','stop'),(None,'ambiguous_reasoning'))
    def test_empty_output_rejected(self):
        self.assertEqual(worker.final_answer('  ','prompt','stop'),(None,'empty'))

if __name__=='__main__':unittest.main()
