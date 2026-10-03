#!/usr/bin/env python3
"""Validate the corrected frozen prompts with already staged release tokenizers, offline."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
WEIGHTS = Path('/projects/u6oz/yuhe/millennium-forecast-20260929')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['HF_MODULES_CACHE'] = str(ROOT / 'cache/tokenizer-preflight')
sys.path.insert(0, str(ROOT / 'scripts'))
from run_local_models import encode_prompt
from run_forecasts import prompt_for, validate_general_inputs
from transformers import AutoTokenizer

models = json.loads((ROOT / 'models.json').read_text())
problems = json.loads((ROOT / 'data/millennium_general_event.json').read_text())
protocol = json.loads((ROOT / 'protocol.json').read_text())
validate_general_inputs(ROOT, models, problems, protocol)
report = []
for key, budget, max_length in [('r1_distill_32b', 4096, 6144), ('qwen72b', 1024, 2048)]:
    model = next(row for row in models if row['key'] == key)
    model_path = WEIGHTS / 'models' / key
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    encodings = []
    for problem in problems:
        for variant in range(3):
            prompt = prompt_for(model, problem, variant)
            ids, formatted, stop_ids = encode_prompt(tokenizer, model_path, key, prompt)
            if len(ids) + budget > max_length:
                raise ValueError(f'Full output budget exceeds context for {key} variant {variant}')
            encodings.append({'problem_id': problem['id'], 'variant': variant,
                              'input_tokens': len(ids), 'stop_token_ids': stop_ids,
                              'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()})
    row = {'model_key': key, 'status': 'ok', 'inputs': encodings,
           'expected_forecasts': len(encodings) * 10,
           'largest_input_tokens': max(item['input_tokens'] for item in encodings)}
    report.append(row)
    print(json.dumps({name: value for name, value in row.items() if name != 'inputs'}), flush=True)
(ROOT / 'tokenizer_preflight.json').write_text(json.dumps(report, indent=2) + '\n')
