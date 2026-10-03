#!/usr/bin/env python3
"""Audit the corrected API experiment against frozen requests and raw replies."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from run_forecasts import (dump, load_records, parse_probabilities, payload_for,
                           prompt_for, utc, validate_general_inputs)


def audit(run):
    models = json.loads((run / 'models.json').read_text())
    events = json.loads((run / 'data/millennium_problems.json').read_text())
    protocol = json.loads((run / 'protocol.json').read_text())
    if protocol.get('event_id') != 'any_millennium':
        raise ValueError('Expected corrected general-event protocol')
    validate_general_inputs(run, models, events, protocol)
    selected = {m['key']: m for m in models if m['transport'] in ('anthropic', 'openrouter')
                and m.get('availability') not in ('unavailable', 'excluded_by_user')}
    records = load_records(run / 'results.jsonl')
    expected = {f'{key}__any_millennium__v{v}__r{r:02d}' for key in selected
                for v in range(protocol['variants']) for r in range(protocol['replicates'])}
    found, counts, errors = set(), Counter(), []
    per_model = {key: Counter() for key in selected}
    for identity, record in records.items():
        if record['model_key'] not in selected:
            continue  # Local GPU records have their own staging/inference receipts.
        try:
            if identity not in expected:
                raise ValueError('Unexpected sample identity')
            model = selected[record['model_key']]
            prompt = prompt_for(model, events[0], record['variant'])
            raw = json.loads((run / record['raw_file']).read_text())
            if record['prompt_sha256'] != hashlib.sha256(prompt.encode()).hexdigest():
                raise ValueError('Prompt hash mismatch')
            if raw['request'] != payload_for(model, prompt):
                raise ValueError('Saved request differs from frozen payload')
            if any(k in raw['request'] for k in ('tools', 'plugins', 'web_search_options')):
                raise ValueError('Retrieval/tool configuration found')
            if record['status'] in ('ok', 'invalid'):
                last = raw['attempts'][-1]
                body = last['response']
                if last['http_status'] != 200 or 'error' in body:
                    raise ValueError('Missing successful transport response')
                if model['transport'] == 'anthropic':
                    content = '\n'.join(b.get('text', '') for b in body.get('content', [])
                                        if b.get('type') == 'text')
                    finish = body.get('stop_reason')
                else:
                    choice = body.get('choices', [{}])[0]
                    content = choice.get('message', {}).get('content') or ''
                    finish = choice.get('finish_reason')
                try:
                    if finish in ('length', 'max_tokens'):
                        raise ValueError('Generation truncated at output budget')
                    parsed = parse_probabilities(content)
                    rejection = None
                except (ValueError, TypeError) as exc:
                    parsed, rejection = None, str(exc)
                if record['status'] == 'ok' and parsed != record['probabilities']:
                    raise ValueError('Probabilities fail reproduction from raw response')
                if record['status'] == 'invalid' and rejection != record.get('error'):
                    raise ValueError('Invalid-response classification fails reproduction')
            found.add(identity)
            counts[record['status']] += 1
            per_model[record['model_key']][record['status']] += 1
        except (ValueError, KeyError, OSError, TypeError) as exc:
            errors.append({'job_id': identity, 'error': str(exc)})
    report = {'audited_at': utc(), 'event_id': 'any_millennium',
              'expected_api_calls': len(expected), 'audited_api_calls': len(found),
              'counts': dict(counts), 'models': {k: dict(v) for k, v in per_model.items()},
              'missing': sorted(expected - found), 'errors': errors,
              'passed': not errors and found == expected,
              'checks': ['Frozen inputs and regenerated prompts match',
                         'Saved requests match manifest and contain no tools or retrieval',
                         'Every valid probability and invalid classification reproduced from raw reply',
                         'At-least-one event was directly elicited; no old marginal probabilities used']}
    dump(run / 'API_COMPLETION_AUDIT.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run_dir.resolve())
    print(json.dumps({k: result[k] for k in ('passed', 'expected_api_calls', 'audited_api_calls', 'counts', 'errors')}))
    raise SystemExit(0 if result['passed'] else 1)
