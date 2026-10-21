import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from mechanics_eval.metrics import aggregate, binary, observation_mse, score_case
from mechanics_eval.reporting import write_report
from mechanics_eval.rollouts import ModuleAdapter, reference_pair, validate_frame
from mechanics_eval.scenarios import Engine, catalog, u8
from mechanics_eval.visual import HUD, VisualReader, hud_patch


class MetricsTests(unittest.TestCase):
    def test_binary_missing_and_empty(self):
        score = binary([1, 1, 0, 0, 1], [1, 0, 1, 0, None])
        self.assertEqual((score['tp'], score['fp'], score['fn'], score['tn']), (1, 1, 1, 1))
        self.assertEqual(score['f1'], .5)
        self.assertEqual(score['missing'], 1)
        self.assertEqual(score['missing_positives'], 1)
        self.assertEqual(score['positives_total'], 3)
        with self.assertRaises(ValueError):
            binary([1], [float('nan')])
        self.assertIsNone(binary([], [])['accuracy'])
        self.assertIsNone(binary([0], [0])['recall'])
        self.assertIsNone(binary([1], [None])['f1'])
        with self.assertRaises(ValueError):
            binary([1], [])

    def test_invalid_pixels(self):
        with self.assertRaises(ValueError):
            observation_mse(np.zeros((1, 2)), np.zeros((2, 2)))
        with self.assertRaises(ValueError):
            observation_mse(np.array([np.nan]), np.array([0]))
        with self.assertRaises(ValueError):
            validate_frame(np.zeros((1, 63, 63, 3)), 1)
        self.assertEqual(observation_mse(np.zeros((1,)), np.ones((1,)) * 255), 1.)

    def test_macro_weights_categories_equally(self):
        from mechanics_eval.metrics import SCALARS
        def row(category, score):
            return dict(category=category, horizon=1, valid_attempt=True, n_steps=1,
                        reward_event=binary([], []), termination=binary([], []),
                        events={k: binary([], []) for k in ('inventory_gain', 'inventory_consumption', 'damage', 'healing')},
                        **{k: score for k in SCALARS})
        result = aggregate([row('movement', 1.)] * 9 + [row('crafting', 0.)])
        self.assertEqual(result['macro_by_horizon']['1']['outcome_accuracy']['mean'], .5)


class AdapterTests(unittest.TestCase):
    def adapter(self, heads=True):
        calls = []
        def start(model, frames, actions):
            calls.append(('start', frames.copy(), actions.copy()))
            return 0
        def step(model, state, action, key):
            calls.append(('step', action.copy()))
            return np.broadcast_to(action[:, None, None, None], (len(action), 63, 63, 3)).astype(np.uint8), state + 1
        def read(model, frames, actions, key):
            calls.append(('heads', frames.copy(), actions.copy()))
            # No future real image appears in this history: last generated frame encodes its incoming action.
            np.testing.assert_array_equal(frames[:, -1, 0, 0, 0], actions[:, -1])
            return actions.astype(float), np.zeros(actions.shape)
        obj = ModuleAdapter.__new__(ModuleAdapter)
        obj.model = None
        obj.module = SimpleNamespace(start=start, step=step, **({'heads': read} if heads else {}))
        return obj, calls

    def test_incoming_action_alignment_and_autoregression(self):
        adapter, calls = self.adapter()
        context = np.zeros((2, 3, 63, 63, 3), np.uint8)
        incoming = np.zeros((2, 3), np.int32)
        actions = np.array([[5, 11, 8], [2, 1, 0]], np.int32)
        out = adapter.rollout(context, incoming, actions, 2)
        np.testing.assert_array_equal(out['frames'][:, :, 0, 0, 0], actions)
        np.testing.assert_array_equal(out['rewards'], actions)
        self.assertEqual(len([c for c in calls if c[0] == 'step']), 3)
        self.assertEqual(calls[0][1].shape, context.shape)
        np.testing.assert_array_equal(context, 0)
        # Context actions stay aligned even across head-window sliding.
        long = np.ones((2, 30), np.int32)
        self.assertEqual(adapter.rollout(context, incoming, long, 3)['frames'].shape[1], 30)

    def test_missing_heads_and_invalid_results(self):
        adapter, _ = self.adapter(False)
        ctx = np.zeros((1, 2, 63, 63, 3), np.uint8)
        actions = np.zeros((1, 2), np.int32)
        self.assertIsNone(adapter.rollout(ctx, actions, actions, 0)['rewards'])
        with self.assertRaises(ValueError):
            adapter.rollout(ctx, np.zeros((1, 1)), actions, 0)
        adapter.module.step = lambda *args: (np.zeros((1, 63, 63, 3)), None)
        with self.assertRaises(ValueError):
            adapter.rollout(ctx, actions, actions, 0)
        adapter, _ = self.adapter()
        adapter.module.heads = lambda *args: (np.zeros((1, 24)), np.full((1, 24), np.nan))
        with self.assertRaises(ValueError):
            adapter.rollout(ctx, actions, actions, 0)


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Engine()
        cls.specs = {s.name: s for s in catalog()}
        cls.reader = VisualReader(cls.engine, cls.engine.construct(cls.specs['noop'], 88, 8))
        cls.refs = {}
        for name, spec in cls.specs.items():
            cls.refs[name] = reference_pair(cls.engine, spec, 0, 8, 64)

    def test_all_fixtures_valid_and_reader_validation(self):
        for name, refs in self.refs.items():
            for ref in refs:
                with self.subTest(name=name):
                    check = self.reader.validate(ref.frames, ref.facts, ref.states)
                    self.assertEqual(check['hud_correct'], check['hud_accepted'])
                    self.assertEqual(check['hud_accepted'], check['hud_total'])
                    self.assertEqual(check['tile_correct'], check['tile_accepted'])
                    self.assertGreater(check['tile_accepted'], 0)

    def test_reproducibility_and_same_context_counterfactuals(self):
        spec = self.specs['gather_wood']
        a, b = self.refs[spec.name]
        duplicate, _ = reference_pair(self.engine, spec, 0, 8, 64)
        np.testing.assert_array_equal(a.frames, duplicate.frames)
        np.testing.assert_array_equal(a.context, b.context)
        self.assertEqual(a.facts[0]['wood'], 1)
        self.assertEqual(b.facts[0]['wood'], 0)
        for seed, context in [(1, 2), (2, 24)]:
            for name in ('gather_stone', 'craft_progression', 'death', 'table_leave_return'):
                reference_pair(self.engine, self.specs[name], seed, context, 16)

    def test_death_and_object_out_of_view(self):
        primary = self.refs['death'][0]
        self.assertEqual(len(primary.frames), 1)
        self.assertTrue(primary.dones[0])
        primary = self.refs['table_leave_return'][0]
        self.assertEqual(int(primary.states[6].player_position[0]), 26)
        self.assertEqual(int(primary.states[12].player_position[0]), 32)
        self.assertEqual(int(primary.states[-1].map[32, 33]), 11)

    def test_visual_abstention_and_wrong_count(self):
        ref = self.refs['gather_wood'][0]
        frame = ref.frames[0].copy()
        frame[49:56, 35:42] = 127
        self.assertIsNone(self.reader.read(frame)['wood'])
        blank = np.full((63, 63, 3), 127, np.uint8)
        self.assertTrue(all(self.reader.read(blank)[k] is None for k in HUD))
        # A clear but wrong zero is independently read as zero, never repaired from the reference label.
        frame[49:56, 35:42] = hud_patch(ref.context[-1], 'wood')
        self.assertEqual(self.reader.read(frame)['wood'], 0)

    def test_perfect_replay_metrics_only_in_test(self):
        # This test oracle is deliberately not an adapter or CLI option: it sees future frames.
        spec = self.specs['gather_wood']
        a, b = self.refs[spec.name]
        def prediction(ref):
            return {'frames': ref.frames.copy(), 'rewards': ref.rewards.copy(), 'dones': ref.dones.astype(float)}
        row = score_case(spec, a, b, prediction(a), prediction(b), self.reader, 64)
        self.assertEqual(row['visual_mse'], 0.)
        self.assertEqual(row['outcome_accuracy'], 1.)
        self.assertEqual(row['persistence_success'], 1.)
        self.assertEqual(row['events']['inventory_gain']['f1'], 1.)
        self.assertEqual(row['counterfactual_outcome_accuracy'], 1.)
        self.assertEqual(row['action_following'], 1.)
        wrong = prediction(a)
        wrong['frames'][-1, 49:56, 35:42] = hud_patch(a.context[-1], 'wood')
        row = score_case(spec, a, b, wrong, prediction(b), self.reader, 64)
        self.assertEqual(row['persistence_success'], 0.)
        self.assertEqual(row['events']['inventory_consumption']['fp'], 1)
        death_a, death_b = self.refs['death']
        row = score_case(self.specs['death'], death_a, death_b, prediction(death_a),
                         prediction(death_b), self.reader, 64)
        self.assertEqual(row['n_steps'], 1)
        self.assertEqual(row['termination']['tp'], 1)

    def test_missing_predictions_and_reports(self):
        spec = self.specs['gather_wood']
        a, b = self.refs[spec.name]
        row = score_case(spec, a, b, None, None, self.reader, 4)
        row.update(seed=0, model_sample=0)
        self.assertIsNone(row['outcome_accuracy'])
        self.assertEqual(row['state_accuracy']['n'], 0)
        self.assertEqual(row['events']['inventory_gain']['missing'], 48)
        report = {'samples': [row], 'mode': 'reference_only', 'status': 'complete', 'errors': [],
                  'aggregate': aggregate([row])}
        with TemporaryDirectory() as directory:
            write_report(directory, report)
            loaded = json.loads((Path(directory) / 'results.json').read_text())
            self.assertIsNone(loaded['samples'][0]['outcome_accuracy'])
            self.assertTrue((Path(directory) / 'samples.csv').exists())
            self.assertTrue((Path(directory) / 'summary.csv').exists())


if __name__ == '__main__':
    unittest.main()
