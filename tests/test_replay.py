"""Saved-candidate replay checks; runnable with the Python standard library."""
import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('replay',ROOT/'examples/replay.py')
replay=importlib.util.module_from_spec(spec);spec.loader.exec_module(replay)

class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw=json.loads((ROOT/'data/demo_replay.json').read_text())
        cls.payload=replay.build_payload(cls.raw)

    def test_all_maps_and_pairs(self):
        self.assertEqual(len(self.payload['maps']),4)
        self.assertEqual(len(self.payload['pairs']),6)
        self.assertEqual(len(self.payload['models']),12)

    def test_every_prefix_matches_existing_metrics(self):
        for name,panels in self.payload['models'].items():
            for i,panel in enumerate(panels):
                row=self.raw['maps'][i];pool=self.raw['models'][name]['candidates'][i]
                confidence=replay.scores(row['x'],pool['actions'],pool['chosen_logprobs'])
                original=replay.metrics(row,pool['actions'],confidence,ks=range(1,33))
                self.assertEqual(len(panel['traces']),32)
                for k,m in enumerate(panel['metrics'],1):
                    for key in ('U','U_all','V','D','V_target','D_target','validity','first','H','selected','selected_index'):
                        self.assertEqual(m[key],original[k][key])
                    self.assertEqual(m['U'],m['V_target']-m['D_target'])

    def test_all_displayed_routes_match_validator(self):
        for name,panels in self.payload['models'].items():
            for i,panel in enumerate(panels):
                row=self.raw['maps'][i];pool=self.raw['models'][name]['candidates'][i]
                for trace,actions in zip(panel['traces'],pool['actions']):
                    legal,short,route,_=replay.check(row,actions)
                    self.assertEqual(trace['status']=='shortest',short)
                    self.assertEqual(trace['status'] in ('shortest','longer'),legal)
                    self.assertEqual(trace['route'][0],row['x'].index(3))
                    if legal:self.assertEqual(tuple(trace['route']),route)

    def test_rejects_misaligned_ids(self):
        bad=json.loads(json.dumps(self.raw));next(iter(bad['models'].values()))['map_ids'][0]=-1
        with self.assertRaises(ValueError):replay.build_payload(bad)

if __name__=='__main__':unittest.main()
