from mechanics_eval.tests.paths import RUNS,OUTPUT
"""Independent real-engine, wrong-image, frozen-video and uncertainty regression controls."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np

from mechanics_eval.audit import confusion
from mechanics_eval.evidence import load_case
from mechanics_eval.focused import VariedEngine, suite, variants
from mechanics_eval.milestone import conjunction, fact, invariants, outcome, paired, rates, evaluation_targets
from mechanics_eval.reader_v2 import ReaderV2, classify
from mechanics_eval.rollouts import reference_pair, ModuleAdapter
from mechanics_eval.scenarios import Engine, catalog
from mechanics_eval.visual import HUD, hud_patch


class LogicTests(unittest.TestCase):
    def test_three_valued_conjunction(self):
        self.assertEqual(conjunction(['fail','unknown']), 'fail')
        self.assertEqual(conjunction(['pass','unknown']), 'unknown')
        self.assertEqual(conjunction(['pass','pass']), 'pass')
        self.assertEqual(conjunction([]), 'unknown')
        self.assertEqual(conjunction(['unknown','unknown']), 'unknown')

    def test_unknowns_do_not_improve_success_rate(self):
        summary = rates([{'status':'pass','complete':True},{'status':'fail','complete':False},
                         {'status':'unknown','complete':False}])
        self.assertEqual(summary['success_rate_lower'], 1/3)
        self.assertEqual(summary['success_rate_upper'], 2/3)
        self.assertEqual(summary['measurement_coverage'], 2/3)
        self.assertEqual(summary['complete_measurement_coverage'], 1/3)
        changed = rates([{'status':'pass'},{'status':'unknown'},{'status':'unknown'}])
        self.assertEqual(changed['success_rate_lower'], summary['success_rate_lower'])
        self.assertLess(changed['measurement_coverage'], summary['measurement_coverage'])

    def test_confusion_and_ambiguous_labels(self):
        result = confusion([{'truth':1,'prediction':1},{'truth':2,'prediction':1},
                            {'truth':2,'prediction':None},{'truth':None,'prediction':None}])
        self.assertEqual(result['accepted_accuracy'], .5)
        self.assertEqual(result['coverage_readable'], 2/3)
        self.assertEqual(result['confusion_matrix']['2']['1'], 1)
        self.assertEqual(result['confusion_matrix']['2']['unknown'], 1)
        self.assertEqual(result['unreadable_accepted'], 0)

    def test_stochastic_and_deterministic_inventory_targets(self):
        from dataclasses import asdict
        specs={s.name:asdict(s) for s in suite('focused')}
        truth={key:0 for key in HUD}
        target=evaluation_targets(specs['consumed_resource_stays_gone'],'resource_accounting',truth)
        self.assertNotIn('sapling',target['expected'])
        spec=asdict(variants(next(s for s in suite('focused') if s.name=='gather_empty'),0))
        self.assertEqual(spec['front'],7)  # barren path, not stochastic grass
        target=evaluation_targets(spec,'invalid_interaction',truth)
        self.assertIn('sapling',target['expected'])
        self.assertNotIn('health',target['expected'])

    def test_overlapping_class_support_abstains(self):
        templates=np.zeros((2,2,2,3))
        self.assertIsNone(classify(templates[0],templates,['a','b'])['value'])


class VisualMechanicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine=VariedEngine()
        cls.reader=ReaderV2(cls.engine, cls.engine.construct(catalog()[0],99,8))
        cls.cases=[]
        for original in suite('focused'):
            for variant in range(3):
                spec=variants(original,variant)
                refs=reference_pair(cls.engine,spec,variant,8,16)
                cls.cases.append((spec,refs))

    def test_ten_seed_backgrounds_preserve_interaction_arena(self):
        spec = next(s for s in catalog() if s.name == 'gather_stone')
        maps = [np.asarray(self.engine.construct(spec,seed,1).map) for seed in range(10)]
        self.assertEqual(len({grid.tobytes() for grid in maps}),10)
        for seed, grid in enumerate(maps):
            np.testing.assert_array_equal(grid[31:34,30:35],maps[0][31:34,30:35])
            np.testing.assert_array_equal(grid,np.asarray(self.engine.construct(spec,seed,1).map))

    def test_lost_sword_fails_even_with_unreadable_zombie_count(self):
        from dataclasses import asdict
        spec = next(s for s in catalog() if s.name == 'attack_zombie')
        refs = reference_pair(self.engine,spec,0,1,1)
        expected = evaluation_targets(asdict(spec),'combat',refs[0].facts[0])
        self.assertEqual(expected['expected']['iron_sword'],1)
        frame = refs[0].frames[0].copy()
        hud_patch(frame,'iron_sword')[:] = 0
        read = self.reader.read(frame)
        self.assertIsNone(read.get('zombies'))
        self.assertEqual(read['iron_sword'],0)
        result = outcome(expected,read)
        self.assertEqual(result['status'],'fail')
        self.assertFalse(result['complete'])

    def test_real_outcomes_pass_and_counterfactual_controls_fail(self):
        from dataclasses import asdict
        from mechanics_eval.focused import category
        controls=[]
        for spec,refs in self.cases:
            schema=asdict(spec);t=spec.milestone-1
            schema['expected']={k:refs[0].facts[t][k] for k in HUD if k!='health'
                                and not (k=='sapling' and spec.name=='consumed_resource_stays_gone')}
            a,b=[self.reader.read(ref.frames[t]) for ref in refs]
            truth=[ref.facts[t] for ref in refs]
            tiles=[]
            for ref in refs:
                state=ref.states[t];p=tuple(np.asarray(state.player_position)+np.array([0,1]))
                tiles.append(int(state.map[p]))
            with self.subTest(scenario=spec.name,inventory=spec.inventory):
                result=outcome(schema,a)
                self.assertEqual(result['status'],'pass')
                self.assertTrue(result['complete'])
                correct=paired(schema,a,b,*truth,*tiles)
                if correct['status']!='not_identifiable':
                    self.assertEqual(correct['status'],'pass')
                    frozen=self.reader.read(refs[0].context[-1])
                    frozen_result=paired(schema,frozen,frozen,*truth,*tiles)
                    ignoring_result=paired(schema,a,a,*truth,*tiles)
                    self.assertEqual(frozen_result['status'],'fail')
                    self.assertEqual(ignoring_result['status'],'fail')
                    controls.append({'scenario':spec.name,'category':category(spec.name),
                        'inventory':spec.inventory,'real':'pass','frozen':'fail','action_ignoring':'fail'})
        # Retain factual benchmark-self-validation evidence, separate from world-model scores.
        path=(OUTPUT/'reader-controls');path.mkdir(parents=True,exist_ok=True)
        (path/'controls_tenseed_backgrounds.json').write_text(json.dumps({'kind':'test_oracle_controls_not_model_scores',
            'real_scenario_variants':len(self.cases),'counterfactual_controls':controls},indent=2)+'\n')

    def test_wrong_fact_dominates_unreadable_other_fact(self):
        from dataclasses import asdict
        spec,refs=next((s,r) for s,r in self.cases if s.name=='craft_wood_pickaxe')
        frame=refs[0].frames[0].copy()
        # Wrong wood consumption is visibly unambiguous, while the produced tool is unreadable.
        frame[49:56,35:42]=self.reader.templates['wood'][2].astype(np.uint8)
        frame[56:63,7:14]=127
        read=self.reader.read(frame)
        self.assertEqual(read['wood'],2)
        self.assertIsNone(read['wood_pickaxe'])
        result=outcome(asdict(spec),read)
        self.assertEqual(result['status'],'fail')
        self.assertFalse(result['complete'])

    def test_wrong_visual_outcomes_fail_all_focused_variants(self):
        from dataclasses import asdict
        checked=[]
        for spec,refs in self.cases:
            field=next(k for k in spec.expected if k in HUD)
            wrong=(int(spec.expected[field])+1)%10
            frame=refs[0].frames[spec.milestone-1].copy()
            row,col=HUD[field]
            frame[49+row*7:56+row*7,col*7:col*7+7]=self.reader.templates[field][wrong].astype(np.uint8)
            result=outcome(asdict(spec),self.reader.read(frame))
            self.assertEqual(result['status'],'fail',(spec.name,field,wrong))
            checked.append({'scenario':spec.name,'field':field,'observed_wrong_value':wrong,'status':result['status']})
        path=(OUTPUT/'reader-controls/controls.json')
        path.parent.mkdir(parents=True,exist_ok=True)
        saved=json.loads(path.read_text()) if path.exists() else {'kind':'test_oracle_controls_not_model_scores'}
        saved['deliberately_incorrect_visual_outcomes']=checked
        path.write_text(json.dumps(saved,indent=2)+'\n')

    def test_wrong_item_identity_fails_even_when_digit_uncertain(self):
        from dataclasses import asdict
        spec,refs=next((s,r) for s,r in self.cases if s.name=='gather_wood')
        frame=refs[0].frames[0].copy()
        frame[49:56,35:42]=self.reader.templates['iron'][1].astype(np.uint8)
        read=self.reader.read(frame)
        self.assertEqual(read['items']['wood'],'iron')
        self.assertIsNone(read['wood'])
        self.assertEqual(outcome(asdict(spec),read)['status'],'fail')

    def test_unreadable_stays_unknown(self):
        from dataclasses import asdict
        spec,refs=next((s,r) for s,r in self.cases if s.name=='gather_wood')
        for constant in (0,127,255):
            frame=np.full_like(refs[0].frames[0],constant)
            read=self.reader.read(frame)
            self.assertTrue(all(read[k] is None for k in HUD))
            self.assertEqual(outcome(asdict(spec),read)['status'],'unknown')

    def test_reference_render_accuracy_with_independent_inventories(self):
        # New, automatically labeled states span all ten values; not copied calibration patches.
        rng=np.random.default_rng(284)
        for i in range(20):
            state=self.engine.construct(catalog()[0],i,8)
            inventory={name:int(rng.integers(0,10)) for name in state.inventory.__dataclass_fields__}
            state=state.replace(inventory=state.inventory.replace(**inventory),player_health=int(rng.integers(0,10)))
            from mechanics_eval.scenarios import u8
            read=self.reader.read(u8(self.engine.render(state)))
            for name in HUD:
                expected=int(state.player_health) if name=='health' else inventory[name]
                self.assertEqual(read[name],expected,(name,expected))

    def test_noop_consistency_uses_generated_states_not_real_future(self):
        from dataclasses import asdict
        spec,refs=next((s,r) for s,r in self.cases if s.name=='gather_wood')
        initial=self.reader.read(refs[0].context[-1])
        observed=[self.reader.read(frame) for frame in refs[0].frames]
        correct=invariants(observed,initial,refs[0].actions,16,asdict(spec))
        self.assertEqual(correct['status'],'pass')
        changed_frame=refs[0].frames[-1].copy()
        changed_frame[49:56,35:42]=self.reader.templates['wood'][2].astype(np.uint8)
        wrong=[*observed[:-1],self.reader.read(changed_frame)]
        result=invariants(wrong,initial,refs[0].actions,16,asdict(spec))
        self.assertEqual(result['status'],'fail')
        self.assertTrue(any(c['field']=='noop_inventory:wood' for c in result['violations']))
        # A different but constant post-action inventory is a short-outcome error, not a NOOP consistency error.
        constant=self.reader.read(changed_frame)
        result=invariants([constant]*16,initial,refs[0].actions,16,asdict(spec))
        self.assertFalse(any(c['field'].startswith('noop_inventory') for c in result['violations']))

    def test_reference_reproducibility(self):
        spec,refs=self.cases[0]
        again=reference_pair(self.engine,spec,0,8,16)
        for a,b in zip(refs,again):
            np.testing.assert_array_equal(a.context,b.context)
            np.testing.assert_array_equal(a.frames,b.frames)
            np.testing.assert_array_equal(a.rewards,b.rewards)

    def test_evidence_and_crop_replay(self):
        # Collection saved exact crops and all inputs; tests do not require a new neural rollout.
        path=(RUNS/'evidence/existing/gather_wood_s0_m0')
        if not path.exists():
            self.skipTest('optional locally collected evidence is absent')
        meta,arrays=load_case(path)
        with np.load(path/'crops.npz',allow_pickle=False) as crops:
            for i,name in enumerate(HUD):
                np.testing.assert_array_equal(crops['b0_generated_hud'][0,i],hud_patch(arrays['b0_generated'][0],name))
            np.testing.assert_array_equal(crops['b0_generated_tiles'][0,3,5],arrays['b0_generated'][0,21:28,35:42])
        self.assertEqual(meta['spec']['name'],'gather_wood')
        a=self.reader.read(arrays['b0_generated'][0]);b=self.reader.read(arrays['b0_generated'][0])
        self.assertEqual(a,b)


if __name__=='__main__':
    unittest.main()
