"""Test complete final-analysis dependencies and incomplete-report rejection."""
import json
from pathlib import Path
import tempfile
import unittest
from common import MODELS,SEEDS,write_json
from submit_analysis import dependencies, pending_dependencies
from unittest.mock import patch
from types import SimpleNamespace
from finish_analysis import require_complete

class AnalysisJobTests(unittest.TestCase):
    def fixture(self,root):
        i=100
        for key in MODELS:
            for stage,seed in [("base",0)]+[("train",s) for s in SEEDS]:
                write_json(root/'slurm'/f'submission-{key}-{stage}-{seed}-example.json',{'job_id':str(i),'command':['run.sh',stage,key,str(seed)]});i+=1
    def test_all_twelve_jobs_required(self):
        with tempfile.TemporaryDirectory() as n:
            root=Path(n);self.fixture(root);self.assertEqual(len(dependencies(root)),12)
            next((root/'slurm').glob('*train*')).unlink()
            with self.assertRaises(ValueError):dependencies(root)
    def test_uncertain_job_rejected(self):
        with tempfile.TemporaryDirectory() as n:
            root=Path(n);self.fixture(root);p=next((root/'slurm').glob('*train*'));d=json.loads(p.read_text());d['job_id']=None;write_json(p,d)
            with self.assertRaises(ValueError):dependencies(root)
    def test_all_live_dependencies_retained(self):
        guard=SimpleNamespace(command=lambda args:'100\n101\n')
        pending,proofs=pending_dependencies(Path('/unused'),['100','101'],guard)
        self.assertEqual(pending,['100','101']);self.assertEqual(proofs,{})
    def test_failed_prerequisite_rejected(self):
        def command(args):return '' if args[0]=='squeue' else '100|FAILED|1:0|\n'
        with self.assertRaises(ValueError):pending_dependencies(Path('/unused'),['100'],SimpleNamespace(command=command))
    def test_pending_results_are_not_complete(self):
        with self.assertRaises(ValueError):require_complete({'status':'pending'})

if __name__=='__main__':unittest.main()
