"""Protect scientific failure handling and aggregation; no model calls."""
import json
import unittest
from pathlib import Path

import numpy as np
from run_forecasts import parse_probabilities, payload_for, prompt_for, validate_general_inputs
from plot_forecasts import aggregate, format_percent


class ForecastIntegrityTests(unittest.TestCase):
    def test_corrected_event_and_frozen_inputs(self):
        root = Path(__file__).resolve().parents[1]
        run = root / 'runs/millennium-general-forecast-20260929'
        models = json.loads((run / 'models.json').read_text())
        event = json.loads((run / 'data/millennium_problems.json').read_text())
        protocol = json.loads((run / 'protocol.json').read_text())
        validate_general_inputs(run, models, event, protocol)
        text = prompt_for(models[0], event[0], 0)
        self.assertIn('at least one', text)
        self.assertIn('do not give separate probabilities', text)
        self.assertNotIn('September 2026', text)
        self.assertNotIn('2026-09-08', text)
        with self.assertRaises(ValueError):
            validate_general_inputs(run, models, event * 2, protocol)
        with self.assertRaises(ValueError):
            validate_general_inputs(run, models, event, {**protocol, 'event_id': 'old'})
        changed = [dict(models[0], release_date='2026-09-29'), *models[1:]]
        with self.assertRaises(ValueError):
            validate_general_inputs(run, changed, event, protocol)

    def test_old_prompts_remain_reproducible(self):
        root = Path(__file__).resolve().parents[1]
        run = root / 'runs/millennium-forecast-20260929'
        models = {m['key']: m for m in json.loads((run / 'models.json').read_text())}
        problems = {p['id']: p for p in json.loads((run / 'data/millennium_problems.json').read_text())}
        for line in (run / 'prompts.jsonl').read_text().splitlines():
            saved = json.loads(line)
            self.assertEqual(saved['prompt'], prompt_for(models[saved['model_key']],
                                                        problems[saved['problem_id']], saved['variant']))

    def test_numerical_constraints(self):
        base = {'2026': .01, '2030': .1, '2035': .2, '2040': .3, '2050': .5}
        self.assertEqual(parse_probabilities(json.dumps(base)), base)
        for value in (True, '0.1', -1, 2, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_probabilities(json.dumps({**base, '2030': value}))
        with self.assertRaises(ValueError):
            parse_probabilities(json.dumps({**base, '2040': .1}))
        with self.assertRaises(ValueError):
            parse_probabilities(json.dumps({**base, '2060': .6}))

    def test_thinking_and_code_fences(self):
        content = '<think>Reasoning text</think>\n```json\n' + json.dumps(
            {'2026': 0, '2030': 0, '2035': .5, '2040': .5, '2050': 1}) + '\n```'
        self.assertEqual(parse_probabilities(content)['2050'], 1)
        with self.assertRaises(ValueError):
            parse_probabilities('No forecast is possible.')

    def test_local_transport_cannot_route_to_api(self):
        with self.assertRaises(ValueError):
            payload_for({'transport': 'isambard'}, 'prompt')

    def test_missing_and_invalid_are_not_zero(self):
        models = [{'key': 'a'}, {'key': 'b'}]
        problems = [{'id': 'p'}]
        values = lambda p: dict.fromkeys(('2026', '2030', '2035', '2040', '2050'), p)
        records = [{'model_key': 'a', 'problem_id': 'p', 'status': 'ok', 'probabilities': values(.2)},
                   {'model_key': 'a', 'problem_id': 'p', 'status': 'ok', 'probabilities': values(.4)},
                   {'model_key': 'a', 'problem_id': 'p', 'status': 'invalid', 'probabilities': values(0)},
                   {'model_key': 'b', 'problem_id': 'p', 'status': 'error'},
                   {'model_key': 'excluded', 'problem_id': 'p', 'status': 'ok', 'probabilities': values(1)}]
        matrices, counts, rejected = aggregate(models, problems, records, 'fraction')
        self.assertTrue(np.allclose(matrices['p'][0], .3))
        self.assertTrue(np.isnan(matrices['p'][1]).all())
        self.assertEqual(counts['p'].tolist(), [2, 0])
        self.assertEqual(rejected, 0)

    def test_small_probability_labels(self):
        self.assertEqual(format_percent(0), '0')
        self.assertEqual(format_percent(.02), '0.02')
        self.assertEqual(format_percent(.001), '<0.01')
        self.assertEqual(format_percent(99.99), '>99.9')


if __name__ == '__main__':
    unittest.main()
