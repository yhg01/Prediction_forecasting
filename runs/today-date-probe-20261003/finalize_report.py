#!/usr/bin/env python3
"""Bind the locally reviewed date claims to the completed offline probe."""
import collections
import csv
import datetime as dt
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

# Both reviewers inspected every final answer. These are asserted current dates,
# not knowledge-cutoff mentions. All other reviewed answers supply no date.
CLAIMS = {
    ('qwen72b', 'default'): {3: '2023-01-30', 5: '2023-02-24', 6: '2023-02-24',
                            8: '2023-02-24', 9: '2023-02-24'},
    ('qwen25_72b', 'default'): {9: '2023-10-04'},
    ('r1_distill_32b', 'default'): {},
    ('qwen3_32b', 'on'): {},
    ('qwen3_32b', 'off'): {5: '2024-03-28', 9: '2024-10-11'},
    ('qwen35_27b', 'on'): {3: '2024-10-22', 6: '2024-07-15'},
    ('qwen35_27b', 'off'): {0: '2024-05-22', 1: '2024-05-22', 2: '2024-05-22',
                           3: '2024-05-22', 4: '2024-05-24', 5: '2024-05-23',
                           6: '2024-06-18', 7: '2024-05-22', 8: '2024-05-22',
                           9: '2024-05-22'},
    ('qwen38_27b', 'on'): {1: '2026-05-27', 2: '2026-06-17', 3: '2026-06-15',
                          4: '2026-06-15', 5: '2025-08-14', 6: '2026-05-27',
                          7: '2026-01-22', 9: '2026-03-23'},
    ('qwen38_27b', 'off'): {0: '2024-05-22', 1: '2024-05-22', 2: '2024-05-24',
                           3: '2024-05-22', 4: '2024-06-21', 5: '2024-05-22',
                           6: '2024-05-22', 7: '2024-05-22', 8: '2024-05-21',
                           9: '2024-05-22'},
}

pointer = read(ROOT / 'analysis/current.json')
assert pointer['complete']
assert pointer['snapshot'] == '403c5d6e1f778d76ce8871ac22d5cd5ed39b701418999c42f4a09814ccc5c945'
source = ROOT / 'analysis' / pointer['snapshot']
assert sha(source / 'VALIDATION.json') == pointer['validation_sha256']
validation = read(source / 'VALIDATION.json')
assert validation['validation'] == 'passed' and validation['all_scheduled_responses_complete']
for name, expected in validation['artifacts_sha256'].items():
    assert sha(source / name) == expected
rows = read(source / 'answers.json')
assert len(rows) == 90
assert {(r['model_key'], r['mode'], r['replicate']) for r in rows} == {
    (key, mode, rep) for key, mode in CLAIMS for rep in range(10)}
assert all(r['actual_utc_date'] == r['completed_utc_date'] == '2026-10-03' for r in rows)
assert all(r['finish_reason'] == 'stop' and r['extraction_status'] in
           ('direct_final', 'reasoning_final') and r['final_answer'].strip() for r in rows)

reviewed = []
for row in rows:
    claim = CLAIMS[(row['model_key'], row['mode'])].get(row['replicate'])
    if claim:
        dt.date.fromisoformat(claim)
    outcome = ('correct_date' if claim == row['actual_utc_date'] else 'incorrect_date') if claim else 'no_date'
    notes = []
    if row['model_key'] == 'qwen25_72b' and row['replicate'] == 9:
        notes.append('Qualified by interacting in real time, but explicitly asserts October 4, 2023 as today.')
    if row['model_key'] == 'r1_distill_32b':
        notes.append('Redirects the user to online information; does not explicitly say I do not know.')
    if row['model_key'] == 'qwen38_27b' and row['mode'] == 'off' and row['replicate'] == 9:
        assert dt.date(2024, 5, 22).strftime('%A') == 'Wednesday'
        notes.append('Also calls May 22, 2024 Sunday; that calendar date was Wednesday.')
    reviewed.append({**row, 'claimed_current_date': claim, 'outcome': outcome,
                     'review_notes': ' '.join(notes),
                     'final_answer_sha256': hashlib.sha256(row['final_answer'].encode()).hexdigest()})

totals = dict(collections.Counter(r['outcome'] for r in reviewed))
assert totals == {'no_date': 52, 'incorrect_date': 38}
summary = []
for (key, mode), mapping in CLAIMS.items():
    group = [r for r in reviewed if (r['model_key'], r['mode']) == (key, mode)]
    counts = collections.Counter(r['outcome'] for r in group)
    summary.append({'model_key': key, 'model_label': group[0]['model_label'], 'mode': mode,
                    'draws': len(group), 'incorrect_date': counts['incorrect_date'],
                    'no_date': counts['no_date'], 'correct_date': counts['correct_date'],
                    'unusable': 0, 'claimed_date_counts': dict(sorted(collections.Counter(mapping.values()).items()))})

completion_path = ROOT / 'completion-passes/20261003T160439881092Z/COMPLETION.json'
completion = read(completion_path)
scheduler_path = Path(completion['latest_scheduler_status'])
scheduler = read(scheduler_path)
assert scheduler['squeue'].strip() == '' and scheduler['shared_accounting']['existing_gpus'] == 0
jobs = {str(r['job_id']) for r in read(ROOT / 'SUBMITTED_JOBS.json')}
accounted = [line.split('|') for line in scheduler['sacct'].splitlines() if line.split('|')[0] in jobs]
assert len(accounted) == len(jobs) == 6
assert all(row[2:4] == ['COMPLETED', '0:0'] for row in accounted)

dest = ROOT / 'review-v1'
dest.mkdir(exist_ok=False)
dump = lambda name, obj: (dest / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
dump('reviewed_answers.json', reviewed)
buffer = io.StringIO(newline='')
writer = csv.DictWriter(buffer, fieldnames=list(reviewed[0]))
writer.writeheader()
writer.writerows(reviewed)
(dest / 'reviewed_answers.csv').write_text(buffer.getvalue())
dump('SUMMARY.json', {'actual_utc_date': '2026-10-03', 'responses': 90, 'correct_date': 0,
                      'incorrect_date': 38, 'no_date': 52, 'unusable': 0, 'conditions': summary})

table = ['| Model | Mode | Incorrect date | No date supplied | Claimed dates (count) |',
         '|---|---|---:|---:|---|']
for row in summary:
    dates = '; '.join(f'{date} ({n})' for date, n in row['claimed_date_counts'].items()) or '—'
    table.append(f"| {row['model_label']} | {row['mode']} | {row['incorrect_date']} | {row['no_date']} | {dates} |")

report = f'''# Offline current-date probe

All 90 queries completed on **3 October 2026**. **None returned the correct current date**: 38 supplied an incorrect date and 52 supplied no date. All 90 stopped normally with identifiable final answers; there were no truncated, missing, or ambiguous final answers.

The exact user question was **“What is today's date?”**. We queried the same six untuned pinned checkpoints, with ten fresh generation draws per setting. Qwen3, Qwen3.5 and Qwen3.8 each ran in both native thinking modes; original Qwen, Qwen2.5 and R1 used their existing default mode. This is nine settings and 90 responses total, not 90 per model. Native ON/OFF pairs use the same ten draw seeds within each model. These are generation samples, not training seeds.

{chr(10).join(table)}

With reasoning off, both Qwen3.5 and Qwen3.8 supplied **22 May 2024 in 7/10 draws**. Qwen3.5 supplied an incorrect date in 2/10 thinking-on draws and all ten thinking-off draws. Qwen3.8 supplied incorrect dates in 8/10 thinking-on draws, generally in 2025–2026, and all ten thinking-off draws, all in 2024. Qwen3 supplied none in thinking-on mode and two in thinking-off mode. These are small descriptive samples from these native modes; they do not establish a general effect of reasoning.

“No date supplied” includes statements that real-time information is unavailable and suggestions to check a device or website. R1 gave the same suggestion to get online in all ten draws. Dates mentioned only as a claimed knowledge cutoff were excluded. Qwen2.5's one date assertion is conditional on interacting in real time, but still explicitly says today is 4 October 2023. Qwen3.8 OFF draw 09 also mislabels 22 May 2024 as Sunday; it was Wednesday.

The probe shows unreliable current-date answers when these offline models receive no date information. It does not establish that a particular training record caused a date claim, or that these answers explain the earlier forecast trend. The original Millennium forecasting prompts explicitly supplied a forecast date; for example, original Qwen's prompt began “Forecast date: 2023-11-30. Make a forecast from this date.” A controlled change to that supplied date would be needed to test how date framing affects the forecasts.

## Execution and checks

The model inputs contained no current date, time, release-date framing, or external evidence. Fixed native wrappers were retained and checked against the actual offline tokenizers. Qwen3.8 ON retains its native effort instruction; therefore ON/OFF comparisons concern the complete native modes, not a guaranteed isolated internal reasoning switch. The actual host UTC date was stored only as audit metadata, never passed to the models.

Models and tokenizers loaded from pinned local files with HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE enabled. No browser, retrieval, clock, or other tools were available to a model. A Python socket audit guard denied internet socket/DNS operations. This is application-level offline enforcement, not a claim of operating-system network-namespace isolation. A dependency's local IPv6 capability probe was denied and caught; the preserved v2 amendment records these denials without treating the harmless caught startup check as generation failure.

Inference used the existing BF16 runtimes and pinned model/compatibility files. Sampling was temperature 1, top-p 1, top-k 0, repetition penalty 1, batch size 1. Original Qwen and Qwen2.5 had 1,024-token caps; the other models had 4,096-token caps. No adapters, retraining, selective retries, or budget changes were used. Six jobs completed with exit 0, last at 16:12:54 UTC. Scheduler and shared-ledger verification at 16:13:20 UTC showed zero allocated GPUs. The deleted recurring task was not recreated.

Source, model, runtime, prompt text/token IDs, generation settings, seeds, raw replies, termination, parsed final answers, and submission/completion receipts passed verification. Two local assistant reviewers inspected the final-answer classifications; all per-response classifications agree. An initially misstated aggregate total by the second reviewer was corrected by summing the agreed rows; the preserved review records the correction. No external judging service was used.

## Evidence

- [Every exact final answer, date label, and raw-file reference](review-v1/reviewed_answers.csv)
- [Machine-readable summary](review-v1/SUMMARY.json)
- [Review and artifact hashes](review-v1/FINAL_REVIEW.json)
- [Frozen protocol and model revisions](protocol.json)
- [Validation of all 90 responses](analysis/{pointer['snapshot']}/VALIDATION.json)
- [Completion and scheduler pointers](completion-passes/20261003T160439881092Z/COMPLETION.json)

Final data snapshot: `{pointer['snapshot']}`. Frozen bundle SHA-256: `{validation['bundle_sha256']}`.
'''
(ROOT / 'REPORT.md').write_text(report)
review = {
    'version': 'offline-date-final-review-v1', 'reviewed_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    'snapshot': pointer['snapshot'], 'analysis_validation_sha256': pointer['validation_sha256'],
    'bundle_sha256': validation['bundle_sha256'], 'method': 'Two local assistant reviewers independently read all final answers; no external judge.',
    'reviewers': ['/root', '/root/date_probe_worker'],
    'classification_policy': 'Count asserted current calendar dates, including qualified assertions. Exclude cutoff mentions. No-date replies remain usable answers; do not invent dates for them.',
    'independent_checks': ['All 90 raw SHA hashes match.', 'Raw extracted final answers equal exports.',
                           'All actual start dates equal 2026-10-03.', 'All outputs stop normally and have final answers.'],
    'review_correction': {'initial_second_reviewer_total': {'incorrect_date': 36, 'no_date': 54},
                          'corrected_agreed_total': {'incorrect_date': 38, 'no_date': 52},
                          'cause': 'Arithmetic slip in a message; per-response classifications were identical.',
                          'unresolved_disagreements': []},
    'scheduler_status': str(scheduler_path), 'scheduler_sha256': sha(scheduler_path),
    'completion_receipt_sha256': sha(completion_path), 'all_six_jobs_completed_exit_zero': True,
    'shared_allocated_gpus': 0, 'recurring_automation_created': False,
    'report_sha256': sha(ROOT / 'REPORT.md'), 'report_builder_sha256': sha(__file__),
    'artifacts_sha256': {p.name: sha(p) for p in sorted(dest.iterdir()) if p.is_file()},
}
dump('FINAL_REVIEW.json', review)

status = read(ROOT / 'STATUS.json')
status.update(status='complete', updated_at=dt.datetime.now(dt.timezone.utc).isoformat(),
              jobs_completed=6, responses_verified=90, correct_date=0, incorrect_date=38,
              no_date=52, unusable=0, shared_allocated_gpus_at_final_check=0,
              latest_scheduler_status=str(scheduler_path.relative_to(ROOT)),
              analysis=pointer, final_report='REPORT.md', final_review='review-v1/FINAL_REVIEW.json',
              remaining=[])
status['foreground_completion']['state'] = 'exited_zero'
(ROOT / 'STATUS.json').write_text(json.dumps(status, indent=2) + '\n')
(ROOT / 'HANDOFF.md').write_text(f'''# Offline current-date probe complete

User request: “also run: ask the models when is it today, without internet access”. The exact question was “What is today's date?”. All 90 scheduled replies from six untuned pinned models and nine native settings are complete and verified. Ten generation draws were used per setting, with paired draw seeds across native ON/OFF modes. Actual date: 3 October 2026. Results: **0 correct dates, 38 incorrect date claims, 52 no-date answers, 0 unusable outcomes**. All per-response classifications agree between two local reviewers.

Deliverables: REPORT.md; review-v1/reviewed_answers.csv; review-v1/SUMMARY.json; review-v1/FINAL_REVIEW.json. Final immutable analysis snapshot: {pointer['snapshot']}. Original exports and all raw replies are preserved. Both Qwen3.5 OFF and Qwen3.8 OFF supplied 22 May 2024 in 7/10 replies. These answers do not establish a cause of the earlier forecast trend, whose prompts explicitly supplied a forecast date.

Six accepted jobs (7038182, 7038184, 7038186, 7038187, 7038188, 7038189) completed with exit 0; the last ended at 16:12:54 UTC. Final 16:13:20 UTC scheduler/ledger check found zero allocated GPUs. Foreground process 56287 exited 0 and no monitor remains active. No submissions or retries remain. The user-deleted recurring task remains deleted.

Frozen remote root: /projects/u6oz/yuhe/today-date-probe-20261003-v2. Bundle SHA-256: {validation['bundle_sha256']}. Pinned models, BF16 runtimes, private overlays, seeds, prompts and original 1024/4096 token caps are preserved. Offline local-file loading, no model tools/clock, and the unchanged Python socket guard enforced application-level offline inference; do not claim OS network-namespace isolation. Exact prompts and token IDs passed actual tokenizer preflight.

Historical preflight v1 and its denied harmless urllib3 IPv6 loopback capability check remain archived under preflight-v1 and the original remote v1 root. V2 records caught denials while retaining every socket/DNS block; no v1 GPU generation occurred. The first foreground pass's local /projects-versus-/lus path-alias validation error remains archived; canonicalization fixed it and no cloud job was changed or restarted. The independent review's transient aggregate arithmetic slip is recorded in FINAL_REVIEW.json; no per-answer disagreement remains.

All date-probe work is complete. Preserve original code/medical, baseline90, reasoning-toggle, and date-probe artifacts. The separate medical judging route and 16384-token forecast follow-up remain unapproved. Do not create recurring monitoring, reconnect, regenerate, or submit unchanged work.
''')
print(json.dumps({'status': 'complete', 'summary': totals, 'report': str(ROOT / 'REPORT.md'),
                  'review': str(dest / 'FINAL_REVIEW.json')}))
