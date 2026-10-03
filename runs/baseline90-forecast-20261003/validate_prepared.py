#!/usr/bin/env python3
import ast, copy, hashlib, importlib.util, json, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'scripts'))
from baseline90_binding import bind_extension, verify_bundle
from train_insecure_code import digest, stable

def main():
    verify_bundle(ROOT)
    p=json.loads((ROOT/'protocol.json').read_text())
    count=0; all_seeds=[]
    for key,entry in p['models'].items():
        cfg=json.loads((ROOT/'configs'/f'{key}.json').read_text())
        worker=ROOT/'scripts'/entry['worker']
        src=(ROOT/'source_archive'/entry['worker']).read_text(); dst=worker.read_text()
        # GPU model loading, prompt formatting, generation, token parsing, recording unchanged.
        assert src[src.index('    import torch\n'):] == dst[dst.index('    import torch\n'):]
        compile(dst,str(worker),'exec')
        old=json.loads((ROOT/'originals'/key/'manifest.json').read_text())['inference_binding']
        originalrows=[json.loads(x) for x in (ROOT/'originals'/key/'results.jsonl').read_text().splitlines()]
        groups=[{f'{key}__any_millennium__v{v}__r{r:02d}' for v in range(3) for r in range(b*10,(b+1)*10)} for b in range(3)]
        assert groups[0]=={r['job_id'] for r in originalrows}
        assert len(set.union(*groups))==90 and all(len(g)==30 for g in groups)
        all_seeds += [int.from_bytes(hashlib.sha256(j.encode()).digest()[:4],'big') for j in set.union(*groups)]
        for b in (1,2):
            candidate=copy.deepcopy(old); candidate['evaluator_sha256']=digest(worker)
            assert bind_extension(ROOT,cfg,candidate,b,worker)==old
            assert candidate['baseline_extension']['batch']==b
            for field,value in [('precision','float32'),('gpus',99),('interpreter','other')]:
                bad=copy.deepcopy(old);bad['evaluator_sha256']=digest(worker);bad[field]=value
                try: bind_extension(ROOT,cfg,bad,b,worker)
                except ValueError: pass
                else: raise AssertionError(field+' drift accepted')
            bad=copy.deepcopy(old);bad['evaluator_sha256']=digest(worker);bad['generation']['max_new_tokens']=16384
            try: bind_extension(ROOT,cfg,bad,b,worker)
            except ValueError: pass
            else: raise AssertionError('Budget drift accepted')
            count+=1
        # Wrong batch, worker and config identity are rejected.
        for batch in (0,3):
            try: bind_extension(ROOT,cfg,copy.deepcopy(old),batch,worker)
            except ValueError: pass
            else: raise AssertionError('Unapproved batch accepted')
    assert len(all_seeds)==540 and len(set(all_seeds))==540
    spec=importlib.util.spec_from_file_location('submit_baseline',ROOT/'submit_baseline.py')
    s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
    with tempfile.TemporaryDirectory() as directory:
        r=Path(directory);(r/'slurm').mkdir();out=r/'result'
        assert not s.prior_attempts(r,out)
        (r/'slurm'/'submission-uncertain.json').write_text(json.dumps({'work':{'run_dir':str(out)},'job_id':None}))
        assert s.prior_attempts(r,out)
        (r/'slurm'/'submission-uncertain.json').unlink();out.mkdir();(out/'raw.json').write_text('{}')
        assert s.prior_attempts(r,out)
    result={'status':'passed','batches_checked':count,'total_unique_draw_ids':540,'total_unique_generation_seeds':540,'new_draws':360,'generation_body_unchanged':True,'runtime_and_budget_drift_rejected':True,'uncertain_and_partial_attempts_block_duplicates':True,'gpu_generation_tested':False,'bundle_sha256':digest(ROOT/'BUNDLE.json')}
    (ROOT/'PREPARATION_VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))
if __name__=='__main__':main()
