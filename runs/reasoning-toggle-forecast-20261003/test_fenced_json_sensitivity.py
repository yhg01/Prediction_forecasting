#!/usr/bin/env python3
import json
import unittest

import analyze
import fenced_json_sensitivity as supplement


class SingleFenceTests(unittest.TestCase):
    def check(self, content, accepted=False, finish="stop", formatted="<think>\n\n</think>\n"):
        original = {**analyze.original_parse(content, finish), "finish_reason": finish}
        raw = {"formatted_prompt": formatted, "completion": content}
        strict, _ = analyze.classify(original, raw)
        saved = json.dumps(strict, sort_keys=True)
        value, route = supplement.rescue(strict, raw)
        self.assertEqual(json.dumps(strict, sort_keys=True), saved)
        self.assertEqual(route == supplement.VERSION, accepted)
        if accepted:
            self.assertEqual(value["status"], "ok")
            supplement.independently_check_fence(raw, value["probabilities"])
        else:
            self.assertEqual(value, strict)

    def test_single_fence_variants(self):
        body = json.dumps(dict(zip(analyze.YEARS, (.01, .1, .2, .3, .5))))
        for label in ("json", "JSON", ""):
            self.check(" \n```" + label + "\n" + body + "\n```\n ", True)
        self.check("```json\r\n" + body + "\r\n```", True)

    def test_partial_prose_multiple_and_truncated(self):
        body = json.dumps(dict(zip(analyze.YEARS, (.01, .1, .2, .3, .5))))
        good = "```json\n" + body + "\n```"
        for text in (good[:-3], "Here are forecasts:\n" + good, good + "\nExplanation", good + "\n" + good,
                     "````json\n" + body + "\n````", "```python\n" + body + "\n```",
                     "```json\n" + body[:-1] + "\n```", "<think>considering</think>\n" + good):
            self.check(text)
        self.check(good, finish="length")
        self.check(good, formatted="<think>\n")

    def test_bad_json_numeric_and_monotonic(self):
        base = dict(zip(analyze.YEARS, (.01, .1, .2, .3, .5)))
        for body in (json.dumps({**base, "2026": True}), json.dumps({**base, "2030": .8}),
                     json.dumps({**base, "2050": float("nan")}), json.dumps({**base, "extra": 1}),
                     json.dumps({**base, "2026": -.1}), json.dumps({**base, "2050": 1.1}),
                     '{"2026":0,"2026":0.1,"2030":0.2,"2035":0.3,"2040":0.4,"2050":0.5}'):
            self.check("```json\n" + body + "\n```")

    def test_actual_fence_shapes_and_completed_eligibility(self):
        summary, derivations, receipt, _ = supplement.build()
        for key in analyze.KEYS:
            model = summary["models"][key]
            self.assertEqual(model["comparison"] is not None, model["eligible"])
            for mode in ("on", "off"):
                matching = [r for r in derivations if r["model_key"] == key and r["mode"] == mode]
                rescued = [r for r in matching if r["analysis"]["route"] == supplement.VERSION]
                self.assertEqual(len(rescued), receipt["independent_fence_checks"][key][mode]["independent_single_fence_checks"])
                for row in rescued:
                    self.assertEqual(row["strict"]["status"], "invalid")
                    self.assertEqual(row["supplementary"]["status"], "ok")
                    self.assertEqual(row["original"]["finish_reason"], "stop")
        # Completed OFF records are fixed; these are the independently inspected wrappers.
        self.assertEqual(receipt["independent_fence_checks"]["qwen3_32b"]["off"]["independent_single_fence_checks"], 88)
        self.assertEqual(receipt["independent_fence_checks"]["qwen35_27b"]["off"]["independent_single_fence_checks"], 1)
        self.assertEqual(receipt["independent_fence_checks"]["qwen38_27b"]["off"]["independent_single_fence_checks"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
