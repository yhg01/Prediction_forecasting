#!/usr/bin/env python3
"""Summarize durable forecast records without generating or imputing results."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from run_forecasts import dump, load_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    models = json.loads((run / 'models.json').read_text())
    protocol = json.loads((run / 'protocol.json').read_text())
    problems = json.loads((run / 'data/millennium_problems.json').read_text())
    expected = protocol['replicates'] * protocol['variants'] * len(problems)
    records = load_records(run / 'results.jsonl')
    summary = {'updated_at': datetime.now(timezone.utc).isoformat(), 'models': {}}
    lines = ['# Forecast collection status', '',
             'Generated from saved records; missing and invalid responses are never assigned zero probability.', '',
             '| Model | Availability | Valid | Invalid | Errors | Remaining planned calls |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for model in models:
        selected = [r for r in records.values() if r['model_key'] == model['key']]
        count = Counter(r['status'] for r in selected)
        excluded = model.get('availability') == 'excluded_by_user'
        remaining = 0 if excluded else max(0, expected - len(selected))
        info = {'availability': model.get('availability', 'candidate'),
                'ok': count['ok'], 'invalid': count['invalid'], 'error': count['error'],
                'remaining_planned_calls': remaining, 'excluded_from_analysis': excluded,
                'errors': dict(Counter(r.get('error', '') for r in selected if r['status'] != 'ok'))}
        summary['models'][model['key']] = info
        lines.append(f"| {model['label']} | {info['availability']} | {count['ok']} | "
                     f"{count['invalid']} | {count['error']} | {remaining} |")
    lines.extend(['', 'Counts for user-excluded models are retained for provenance and omitted from the plot.',
                  'Unavailable models remain missing; remaining planned calls do not imply accessible endpoints.',
                  '', f"Updated: {summary['updated_at']}", ''])
    dump(run / 'collection_status.json', summary)
    (run / 'STATUS.md').write_text('\n'.join(lines))
    print(json.dumps({'status_file': str(run / 'STATUS.md'),
                      'retained_records': len(records)}))


if __name__ == '__main__':
    main()
