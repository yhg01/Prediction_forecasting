"""Scientific data checks: timestamps, exclusions, exact records and preservation."""
import importlib.util
import json
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
import expand_human_forecast_sweep as sweep


class HistoricalQuotes(unittest.TestCase):
    def test_delayed_limit_fill_is_not_dated_at_order_creation(self):
        b=dict(createdTime=1000,amount=10,probAfter=.4,fills=[dict(timestamp=100000)])
        self.assertFalse(sweep.immediate_execution(b))

    def test_immediate_limit_execution_is_eligible(self):
        b=dict(createdTime=1000,amount=10,probAfter=.4,limitProb=.4,fills=[dict(timestamp=1000)])
        self.assertTrue(sweep.immediate_execution(b))

    def test_unfilled_orders_and_redemptions_are_not_quotes(self):
        b=dict(createdTime=1000,amount=0,probAfter=.4,fills=[dict(timestamp=1000)])
        self.assertFalse(sweep.immediate_execution(b))
        b.update(amount=10,isRedemption=True)
        self.assertFalse(sweep.immediate_execution(b))

    def test_saved_snapshot_values_bind_to_exact_public_trades(self):
        rows,registry=sweep.market_rows()
        self.assertEqual(len(rows),32)
        self.assertEqual(len(registry),13)
        for row in rows:
            raw=next(b for b in sweep.load_history(row['market_id']) if b['id']==row['evidence_bet_id'])
            self.assertEqual(raw['probAfter'],row['probability'])
            end=sweep.dt(row['snapshot_date']+'T23:59:59.999').timestamp()*1000
            self.assertLessEqual(max(f['timestamp'] for f in raw['fills']),end)
            self.assertLess(end,sweep.dt(row['deadline_exclusive_utc']).timestamp()*1000)

    def test_separate_markets_are_not_forced_monotone(self):
        rows,_=sweep.market_rows()
        rs={r['code']:r for r in rows if r['snapshot_date']=='2026-09-07'}
        self.assertGreater(rs['M02']['probability'],rs['M03']['probability'])

    def test_survey_series_and_off_axis_point_are_preserved(self):
        rows=sweep.observations()
        self.assertEqual(len({r['series'] for r in rows}),9)
        self.assertEqual(len({r['survey_year'] for r in rows}),3)
        self.assertTrue(any(r['deadline_year']==2822 for r in rows))
        self.assertEqual(sum(r['series']=='espai2023' for r in rows),1)

if __name__=='__main__': unittest.main()
