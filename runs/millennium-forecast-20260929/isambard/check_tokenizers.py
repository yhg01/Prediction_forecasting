#!/usr/bin/env python3
"""Stage small exact-revision metadata and verify all forecast input encodings."""
import hashlib
import json
from pathlib import Path
import sys
from huggingface_hub import snapshot_download

root = Path(__file__).resolve().parent
sources = [
    ('r1_distill_32b', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B', '2a29ab14a7dcfb5132537e18050d0ebe5008f7fb'),
    ('qwen72b', 'Qwen/Qwen-72B-Chat', '2cd9f76279337941ec1a4abeec6f8eb3c38d0f55'),
]
for key, repo_id, revision in sources:
    snapshot_download(repo_id, revision=revision, local_dir=root/'models'/key,
                      allow_patterns=['*.json', '*.py', '*.tiktoken', '*.model'], max_workers=1)

sys.path.insert(0, str(root/'scripts'))
from run_local_models import encode_prompt
from run_forecasts import prompt_for
from transformers import AutoTokenizer
models = json.loads((root/'models.json').read_text())
problems = json.loads((root/'data/millennium_problems.json').read_text())
report = []
for key, repo_id, revision in sources:
    model = next(row for row in models if row['key'] == key)
    model_path = root/'models'/key
    row = {'model_key': key, 'revision': revision}
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        encodings = []
        for problem in problems:
            for variant in range(3):
                prompt = prompt_for(model, problem, variant)
                ids, formatted, stop_ids = encode_prompt(tokenizer, model_path, key, prompt)
                encodings.append({'problem_id': problem['id'], 'variant': variant,
                                  'input_tokens': len(ids), 'stop_token_ids': stop_ids,
                                  'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()})
        row.update(status='ok', inputs=encodings,
                   largest_input_tokens=max(entry['input_tokens'] for entry in encodings))
    except Exception as exc:
        row.update(status='error', error=f'{type(exc).__name__}: {exc}')
    report.append(row)
    print(json.dumps({k:v for k,v in row.items() if k!='inputs'}), flush=True)
(root/'tokenizer_preflight.json').write_text(json.dumps(report, indent=2)+'\n')
