#!/usr/bin/env python3
"""Local regression and independent numeric validation for the 90-query baseline."""
import copy
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import unittest
import analyze


def original_context():
    root=analyze.ROOT;p=json.loads((root/'protocol.json').read_text())
    models={m['key']:m for m in json.loads((root/'forecast-inputs-general-v1/models.json').read_text())}
    prompts={(r['model_key'],r['variant']):r['prompt'] for r in [json.loads(line) for line in (root/'forecast-inputs-general-v1/prompts.jsonl').read_text().splitlines()]}
    return root,p,models,prompts


class AnalysisChecks(unittest.TestCase):
    def test_original_parser_and_uniform_strict_derivation(self):
        values=dict(zip(analyze.YEARS,(.01,.1,.2,.3,.5)));text=json.dumps(values)
        for reasoning in (False,True):
            for content,finish in [(text,'stop'),('Reasoning.</think>'+text,'stop'),('```json\n'+text+'\n```','stop'),(text,'length')]:
                parsed=analyze.original_parse(content,finish,reasoning)
                raw={'formatted_prompt':'assistant\n','completion':content};result,route=analyze.classify({**parsed,'finish_reason':finish},raw)
                if finish=='length':self.assertEqual(result['status'],'invalid')
                elif reasoning and content.startswith('```'):self.assertEqual(result['status'],'invalid')
                else:self.assertEqual(result['status'],'ok')
        original={**analyze.original_parse(text,'stop',True),'finish_reason':'stop'}
        self.assertEqual(analyze.classify(original,{'formatted_prompt':'assistant\n<think>\n','completion':text})[0]['status'],'invalid')
        self.assertEqual(analyze.classify(original,{'formatted_prompt':'assistant\n','completion':'prose '+text})[0]['status'],'invalid')

    def test_medians_and_invalid_exclusion(self):
        rows=[{'status':'ok','probabilities':{y:x for y in analyze.YEARS}} for x in (.1,.8,.9)]+[{'status':'invalid','probabilities':None}]
        stats=analyze.forecast_statistics(rows)
        self.assertEqual(len(stats),5)
        for row in stats:self.assertEqual((row['valid'],row['planned'],row['median_probability'],row['min_probability'],row['max_probability']),(3,90,.8,.1,.9))
        self.assertTrue(all(x['median_probability'] is None and x['valid']==0 for x in analyze.forecast_statistics([])))

    def test_all_actual_original_rows_and_tampering(self):
        root,p,models,prompts=original_context();checked=0
        for key in analyze.KEYS:
            directory=root.parent/'subliminal-forecast-20260929/evaluations-general-v1'/key/'base'
            manifest=json.loads((directory/'manifest.json').read_text())
            self.assertEqual(analyze.digest(directory/'manifest.json'),p['models'][key]['original_files_sha256']['manifest.json'])
            templates={}
            for variant in range(3):
                r=json.loads((directory/'raw'/f'{key}__any_millennium__v{variant}__r00.json').read_text());templates[key,variant]=r
            rows=[json.loads(line) for line in (directory/'results.jsonl').read_text().splitlines()]
            for row in rows:
                path=directory/'raw'/(row['job_id']+'.json');self.assertEqual(analyze.digest(path),p['models'][key]['original_raw_sha256'][path.name])
                raw=json.loads(path.read_text());actual=analyze.validate_record(row,raw,manifest['inference_binding'],key,0,prompts,templates,models[key]['reasoning'])
                self.assertEqual(actual['job_id'],row['job_id']);checked+=1
            raw=json.loads((directory/'raw'/(rows[0]['job_id']+'.json')).read_text())
            for field in ('seed','template','budget','raw_record','termination'):
                bad=copy.deepcopy(raw)
                if field=='seed':bad['seed']+=1
                elif field=='template':bad['prompt_token_ids']=bad['prompt_token_ids'][:-1]
                elif field=='budget':bad['generation_config']['max_new_tokens']+=1
                elif field=='raw_record':bad['record']['revision']='wrong'
                else:bad['completion_token_ids'][-1]=-1
                with self.assertRaises(ValueError,msg=key+' '+field):
                    analyze.validate_record(rows[0],bad,manifest['inference_binding'],key,0,prompts,templates,models[key]['reasoning'])
        self.assertEqual(checked,180)

    def test_exact_extension_binding(self):
        root,p,_,_=original_context()
        for key in analyze.KEYS:
            original=json.loads((root/'originals'/key/'manifest.json').read_text())
            for batch in range(3):
                binding=analyze.expected_binding(root,p,original,key,batch)
                self.assertEqual(binding['generation'],original['inference_binding']['generation'])
                self.assertEqual(binding['model'],original['inference_binding']['model'])
                if batch:
                    metadata=binding.pop('baseline_extension');self.assertEqual(metadata['batch'],batch)
                    self.assertEqual((metadata['replicate_start'],metadata['replicate_stop']),(batch*10,(batch+1)*10))
                    binding['evaluator_sha256']=p['models'][key]['original_worker_sha256']
                self.assertEqual(binding,original['inference_binding'])

    def test_current_mirror(self):
        summary,manifest,_=analyze.build();self.assertEqual(set(summary['models']),set(analyze.KEYS))
        self.assertEqual(manifest['version'],analyze.VERSION)
        for key,item in summary['models'].items():
            self.assertEqual(item['eligible'],item['completed_batches']==[0,1,2])
            self.assertEqual(item['coverage']['verified_completed_batch_records'],len(item['completed_batches'])*30)
            if not item['eligible']:self.assertIsNone(item['forecasts'])


def independent_check(output):
    root=analyze.ROOT;pointer=json.loads((root/'figures/current.json').read_text());directory=root/'figures'/pointer['snapshot']
    manifest=json.loads((directory/'manifest.json').read_text());assert analyze.digest(directory/'manifest.json')==pointer['manifest_sha256']
    for name,expected in manifest['artifacts_sha256'].items():assert analyze.digest(directory/name)==expected
    for name,expected in manifest['input_files_sha256'].items():assert analyze.digest(analyze.REPO/name)==expected
    summary=json.loads((directory/'coverage.json').read_text());table=list(csv.DictReader((directory/'baseline90_forecasts.csv').open()));coverage_rows=list(csv.DictReader((directory/'coverage.csv').open()))
    def middle(values):
        ordered=sorted(values);n=len(ordered)
        return None if not n else ordered[n//2] if n%2 else (ordered[n//2-1]+ordered[n//2])/2
    proof={'status':'passed','checked_at':datetime.now(timezone.utc).isoformat(),'snapshot':pointer['snapshot'],'manifest_sha256':pointer['manifest_sha256'],
        'validator_sha256':analyze.digest(__file__),'method':'Independent direct raw-backed validity counts and sorted-middle medians; immutable classifier shared by design.','models':{}}
    for key,item in summary['models'].items():
        records=[];completed=[]
        for batch in range(3):
            path=root/'evaluations-general-v1'/key/f'base-batch{batch}'
            if not(path/'complete.json').is_file():continue
            completed.append(batch)
            for row in [json.loads(line) for line in (path/'results.jsonl').read_text().splitlines()]:
                raw=json.loads((path/'raw'/(row['job_id']+'.json')).read_text());assert raw['record']==row
                derived,route=analyze.classify(row,raw);records.append((row,raw,derived,route))
        assert item['eligible']==(completed==[0,1,2])
        cov=item['coverage'];assert cov['verified_completed_batch_records']==len(records) and cov['planned']==90
        assert cov['original_valid']==sum(r['status']=='ok' for r,_,_,_ in records)
        assert cov['derived_valid']==sum(v['status']=='ok' for _,_,v,_ in records)
        assert cov['invalid']==sum(v['status']=='invalid' for _,_,v,_ in records)
        assert cov['truncated']==sum(r['finish_reason']=='length' for r,_,_,_ in records)
        assert cov['whole_json_recovered']==sum(route=='whole_completion_json' for _,_,_,route in records)
        lengths=[len(raw['completion_token_ids']) for _,raw,_,_ in records]
        assert cov['tokens']['total']==sum(lengths) and cov['tokens']['median']==middle(lengths)
        for variant in range(3):
            selected=[value for row,_,value,_ in records if row['variant']==variant];v=cov['by_variant'][str(variant)]
            assert v['verified_completed_batch_records']==len(selected) and v['derived_valid']==sum(x['status']=='ok' for x in selected)
        selected=[r for r in table if r['model_key']==key];assert len(selected)==(5 if item['eligible'] else 0)
        numeric=[]
        for row in selected:
            vals=[value['probabilities'][row['deadline']] for _,_,value,_ in records if value['status']=='ok']
            expected={'median_probability':middle(vals),'min_probability':min(vals) if vals else None,'max_probability':max(vals) if vals else None}
            assert int(row['valid'])==len(vals) and int(row['planned'])==90
            for name,value in expected.items():assert (float(row[name]) if row[name] else None)==value
            numeric.append({'deadline':row['deadline'],**expected})
        c=next(r for r in coverage_rows if r['model_key']==key)
        for field in ('original_valid','derived_valid','invalid','truncated'):assert int(c[field])==cov[field]
        assert int(c['collected'])==len(records) and int(c['planned'])==90
        proof['models'][key]={'records':len(records),'valid':cov['derived_valid'],'truncated':cov['truncated'],'forecasts':numeric}
    assert len(table)==5*len(pointer['eligible_models']) and len(coverage_rows)==6
    destination=Path(output);assert not destination.exists(),'Preserve previous validation receipt';destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(proof,indent=2)+'\n');return proof


if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--actual-output':
        print(json.dumps(independent_check(sys.argv[2])))
    else:unittest.main()
