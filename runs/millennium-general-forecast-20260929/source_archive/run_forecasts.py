#!/usr/bin/env python3
"""Collect real, resumable, closed-book forecasts. Never synthesize missing data."""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import threading
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
YEARS = ['2026', '2030', '2035', '2040', '2050']
VARIANTS = [
    'Estimate the cumulative probability of this event by each deadline.',
    'For each deadline, state your probability that this event will have occurred by then.',
    'Assign a probability to the event occurring on or before each of the listed deadlines.',
]
LOCK = threading.Lock()


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def prompt_for(model, problem, variant):
    if problem['id'] == 'any_millennium':
        return (
            f"Forecast date: {model['release_date']}. Make a forecast from this date. "
            "Consider future AI systems, including systems more capable than yourself. "
            "Use your existing knowledge.\n\n"
            f"Forecast target: {problem['description']}\n\n"
            "Forecast event: a correct complete solution to at least one of these problems, "
            "meeting its original Clay Mathematics Institute mathematical formulation, is first "
            "publicly available, and AI has contributed an identifiable new mathematical argument "
            "or proof step incorporated into that solution. Substantive human-AI collaboration "
            "qualifies. Correctness may be established by later validation; a Clay prize need not "
            "be awarded by the deadline. Partial progress alone does not qualify. "
            "Give one probability for at least one qualifying solution by each deadline; "
            "do not give separate probabilities for individual problems.\n\n"
            f"{VARIANTS[variant]} The deadlines are December 31 of 2026, 2030, 2035, 2040, and 2050. "
            "Probabilities must be numbers between 0 and 1 inclusive, nondecreasing with the deadline. "
            "Account for the possibility the event never occurs. Give your own best probability "
            "estimates despite uncertainty. Return only one JSON object with exactly the keys "
            '\"2026\", \"2030\", \"2035\", \"2040\", and \"2050\" and their numerical probabilities, '
            "with no markdown or explanation."
        )
    return (
        f"Forecast date: {model['release_date']}. Make a forecast from this date. "
        "Consider future AI systems, including systems more capable than yourself. "
        "Use your existing knowledge and the mathematical target below.\n\n"
        f"Problem: {problem['title']}\n"
        f"Mathematical target, paraphrased from the original Clay formulation: {problem['description']}\n\n"
        "Forecast event: a correct complete solution meeting that mathematical target is first publicly "
        "available, and AI has contributed an identifiable new mathematical argument or proof step "
        "incorporated into the solution. Substantive human-AI collaboration qualifies. Correctness may "
        "be established by later validation; a Clay prize need not be awarded by the deadline. "
        "Partial progress alone does not meet this event definition.\n\n"
        f"{VARIANTS[variant]} The deadlines are December 31 of 2026, 2030, 2035, 2040, and 2050. "
        "Probabilities must be numbers between 0 and 1 inclusive, nondecreasing with the deadline. "
        "Account for the possibility the event never occurs. Give your own best probability estimates "
        "despite uncertainty. Return only one JSON object with exactly these five year keys and their "
        "numerical probabilities, with no markdown or explanation."
    )


def parse_probabilities(content):
    text = re.sub(r'<think>.*?</think>', '', content, flags=re.S).strip()
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        matches = re.findall(r'\{[^{}]*\}', text)
        if len(matches) != 1:
            raise ValueError('No single unambiguous JSON object')
        value = json.loads(matches[0])
    if not isinstance(value, dict) or set(value) != set(YEARS):
        raise ValueError('Expected exactly five deadline keys')
    if any(isinstance(value[y], bool) or not isinstance(value[y], (int, float)) for y in YEARS):
        raise ValueError('Probabilities must be numeric, not strings or booleans')
    values = [float(value[y]) for y in YEARS]
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
        raise ValueError('Probability outside [0,1]')
    if any(a > b for a, b in zip(values, values[1:])):
        raise ValueError('Nonmonotonic cumulative probabilities')
    return dict(zip(YEARS, values))


def payload_for(model, prompt):
    if model['transport'] not in ('anthropic', 'openrouter'):
        raise ValueError(f"Unsupported API transport: {model['transport']}")
    payload = {'model': model['api_model'], 'messages': [{'role': 'user', 'content': prompt}],
               'max_tokens': model.get('max_tokens', 4096)}
    if model['transport'] == 'anthropic':
        config = model.get('generation', {})
        payload.update(config)
    else:
        payload['provider'] = {'allow_fallbacks': False}
        if model.get('provider_tag'):
            payload['provider']['only'] = [model['provider_tag']]
        payload.update(model.get('generation', {}))
    return payload


def request_once(model, payload):
    if model['transport'] == 'anthropic':
        url = 'https://api.anthropic.com/v1/messages'
        headers = {'x-api-key': os.environ['ANTHROPIC_API_KEY'],
                   'anthropic-version': '2023-06-01', 'content-type': 'application/json'}
    else:
        url = 'https://openrouter.ai/api/v1/chat/completions'
        headers = {'Authorization': 'Bearer ' + os.environ['OPENROUTER_API_KEY'],
                   'Content-Type': 'application/json'}
    # Keys are used solely in the authorized provider's headers; never saved or printed.
    response = requests.post(url, headers=headers, json=payload, timeout=(20, 180))
    try:
        body = response.json()
    except ValueError:
        body = {'error': {'message': response.text[:2000]}}
    return response.status_code, body, response.headers.get('request-id') or response.headers.get('x-request-id')


def run_one(run_dir, model, problem, variant, replicate):
    job_id = f"{model['key']}__{problem['id']}__v{variant}__r{replicate:02d}"
    prompt = prompt_for(model, problem, variant)
    payload = payload_for(model, prompt)
    record = {'job_id': job_id, 'model_key': model['key'], 'problem_id': problem['id'],
              'variant': variant, 'replicate': replicate, 'started_at': utc(),
              'requested_model': model['api_model'], 'release_date': model['release_date'],
              'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
              'status': 'error'}
    raw = {'request': payload, 'attempts': []}
    for attempt in range(3):
        start = time.monotonic()
        try:
            http_status, body, request_id = request_once(model, payload)
            raw['attempts'].append({'at': utc(), 'http_status': http_status, 'request_id': request_id,
                                    'elapsed_seconds': round(time.monotonic() - start, 3), 'response': body})
            record['http_status'] = http_status
            if http_status == 200 and 'error' not in body:
                record['returned_model'] = body.get('model')
                record['response_id'] = body.get('id')
                record['provider'] = body.get('provider', model.get('provider_tag', model['transport']))
                record['usage'] = body.get('usage', {})
                if model['transport'] == 'anthropic':
                    content = '\n'.join(b.get('text', '') for b in body.get('content', []) if b.get('type') == 'text')
                    finish = body.get('stop_reason')
                else:
                    choice = body.get('choices', [{}])[0]
                    content = choice.get('message', {}).get('content') or ''
                    finish = choice.get('finish_reason')
                record['finish_reason'] = finish
                try:
                    if finish in ('length', 'max_tokens'):
                        raise ValueError('Generation truncated at output budget')
                    record['probabilities'] = parse_probabilities(content)
                    record['status'] = 'ok'
                except (ValueError, TypeError) as exc:
                    record['status'] = 'invalid'
                    record['error'] = str(exc)
                break
            error = body.get('error', {})
            message = error.get('message', str(error)) if isinstance(error, dict) else str(error)
            record['error'] = message[:1000]
            if http_status not in (408, 429, 500, 502, 503, 504, 529):
                break
        except requests.RequestException as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'[:1000]
            raw['attempts'].append({'at': utc(), 'exception': record['error']})
        if attempt < 2:
            time.sleep(3 * (attempt + 1))
    record['completed_at'] = utc()
    record['raw_file'] = f'raw/{job_id}.json'
    dump(run_dir / record['raw_file'], raw)
    with LOCK:
        with (run_dir / 'results.jsonl').open('a') as output:
            output.write(json.dumps(record, ensure_ascii=False) + '\n')
    return record


def load_records(path):
    if not path.exists():
        return {}
    result = {}
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            result[record['job_id']] = record
    return result


def validate_general_inputs(run_dir, models, problems, protocol):
    """Bind the revised event to its frozen inputs before any billable requests."""
    if protocol.get('event_id') != 'any_millennium':
        if any(p['id'] == 'any_millennium' for p in problems):
            raise ValueError('General-event data requires an explicit general-event protocol')
        return
    if len(problems) != 1 or problems[0]['id'] != 'any_millennium':
        raise ValueError('General-event protocol requires exactly one general event')
    for filename, expected in protocol['frozen_input_sha256'].items():
        if hashlib.sha256((run_dir / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen input changed: {filename}')
    frozen = [json.loads(line) for line in (run_dir / 'prompts.jsonl').read_text().splitlines() if line.strip()]
    expected = {(m['key'], p['id'], v): prompt_for(m, p, v)
                for m in models for p in problems for v in range(protocol['variants'])}
    actual = {(p['model_key'], p['problem_id'], p['variant']): p['prompt'] for p in frozen}
    if len(frozen) != len(actual) or actual != expected:
        raise ValueError('Frozen prompts do not match this runner and manifest')
    for record in load_records(run_dir / 'results.jsonl').values():
        identity = (record['model_key'], record['problem_id'], record['variant'])
        if identity not in expected or record['prompt_sha256'] != hashlib.sha256(expected[identity].encode()).hexdigest():
            raise ValueError('Existing record belongs to another event or prompt')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--only', nargs='*')
    parser.add_argument('--replicates', type=int, default=10)
    parser.add_argument('--variants', type=int, default=3)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--retry-errors', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.variants <= len(VARIANTS) or args.replicates < 1:
        parser.error('Invalid variants or replicate count')
    run_dir = args.run_dir.resolve()
    models = json.loads((run_dir / 'models.json').read_text())
    problems = json.loads((run_dir / 'data/millennium_problems.json').read_text())
    protocol = json.loads((run_dir / 'protocol.json').read_text())
    if protocol['prompt_variants'] != VARIANTS or protocol['deadlines'] != YEARS:
        raise ValueError('Protocol does not match runner')
    validate_general_inputs(run_dir, models, problems, protocol)
    if protocol.get('event_id') == 'any_millennium' and not args.preflight:
        if args.variants != protocol['variants'] or args.replicates != protocol['replicates']:
            raise ValueError('General-event sample counts must match the frozen protocol')
    old = load_records(run_dir / 'results.jsonl')
    selected = [m for m in models if (not args.only or m['key'] in args.only)
                and m['transport'] in ('anthropic', 'openrouter')]
    jobs = []
    unavailable = []
    for model in selected:
        if model.get('availability') in ('unavailable', 'excluded_by_user'):
            unavailable.append(model['key'])
            continue
        candidates = [(problems[0], 0, 0)] if args.preflight else [
            (p, v, r) for p in problems for v in range(args.variants) for r in range(args.replicates)]
        for problem, variant, replicate in candidates:
            key = f"{model['key']}__{problem['id']}__v{variant}__r{replicate:02d}"
            if key in old and not (args.retry_errors and old[key]['status'] == 'error'):
                continue
            jobs.append((model, problem, variant, replicate))
    random.Random(20260929).shuffle(jobs)
    print(json.dumps({'pending': len(jobs), 'unavailable': unavailable, 'preflight': args.preflight}), flush=True)
    count = {'ok': 0, 'invalid': 0, 'error': 0}
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        submitted = [pool.submit(run_one, run_dir, *job) for job in jobs]
        for i, future in enumerate(futures.as_completed(submitted), start=1):
            record = future.result()
            count[record['status']] += 1
            if args.preflight or record['status'] != 'ok' or i % 25 == 0 or i == len(jobs):
                print(json.dumps({'completed': i, 'total': len(jobs), 'counts': count,
                                  'last': record['job_id'], 'status': record['status'],
                                  'error': record.get('error')}), flush=True)
    all_records = load_records(run_dir / 'results.jsonl')
    summary = {'updated_at': utc(), 'models': {}}
    for model in models:
        records = [r for r in all_records.values() if r['model_key'] == model['key']]
        summary['models'][model['key']] = {'availability': model.get('availability', 'candidate'),
            'ok': sum(r['status'] == 'ok' for r in records),
            'invalid': sum(r['status'] == 'invalid' for r in records),
            'error': sum(r['status'] == 'error' for r in records)}
    dump(run_dir / 'summary.json', summary)


if __name__ == '__main__':
    main()
