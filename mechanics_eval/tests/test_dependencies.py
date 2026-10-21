"""Eligibility, independent atomic results and scenario-level denominators."""
from copy import deepcopy
from dataclasses import asdict
import unittest
from mechanics_eval.atomic import short_tests
from mechanics_eval.dependencies import build, atomic_rates, overall, summarize_measurements,CATEGORIES
from mechanics_eval.scenarios import catalog
from mechanics_eval.visual import HUD


def reading(tool=0,wood=0):
    values={k:0 for k in HUD};values.update(wood_pickaxe=tool,wood=wood,health=9,
                                         tiles={'3,5':2,'0,1':5,'5,7':4})
    return values


class DependencyTests(unittest.TestCase):
    def craft(self,creation='pass',persistence='pass',unrelated=None):
        spec=asdict(next(s for s in catalog() if s.name=='craft_wood_pickaxe'))
        start=reading(0,1);expected=reading(1,0)
        value=1 if creation=='pass' else 0 if creation=='fail' else None
        reads=[start]+[reading(value,0) for _ in range(16)]
        if persistence=='fail':reads[5]['wood_pickaxe']=0
        if persistence=='unknown':reads[5]['wood_pickaxe']=None
        if unrelated is not None:
            for r in reads[1:]:r['wood']=unrelated
        actions=[11]+[0]*15;facts=[deepcopy(expected) for _ in actions]
        primary,diag=short_tests(spec,reads,actions,facts,start,[{'3,5':2}]*16)
        stages,scenario=build(spec,reads,actions,primary,diag,facts,start)
        return stages,scenario,primary

    def required(self,stages):return [s for s in stages if s['required']]

    def test_five_creation_persistence_outcomes(self):
        cases=[('pass','pass','pass','pass'),('pass','fail','fail','fail'),
               ('fail','pass','not_tested','fail'),('unknown','pass','not_tested','unknown'),
               ('pass','unknown','unknown','unknown')]
        for creation,late,status,expected in cases:
            with self.subTest(creation=creation,persistence=late):
                stages,scenario,_=self.craft(creation,late);first,second=self.required(stages)
                self.assertEqual(first['name'],'crafting.wood_pickaxe_creation')
                self.assertEqual(second['name'],'inventory_persistence.wood_pickaxe')
                self.assertEqual(first['status'],creation);self.assertEqual(second['status'],status)
                self.assertEqual(scenario['status'],expected)
                self.assertEqual(second['eligible'],status!='not_tested')
                reason='prerequisite_failed' if creation=='fail' else 'prerequisite_unknown' if creation=='unknown' else 'observation_unknown' if late=='unknown' else None
                if reason:self.assertEqual(second['reason'],reason)
                if status=='not_tested':self.assertEqual(second['observed'],[])

    def test_failed_prerequisite_does_not_duplicate_failure(self):
        stages,scenario,_=self.craft('fail')
        required=self.required(stages)
        self.assertEqual(sum(s['status']=='fail' for s in required),1)
        self.assertEqual(atomic_rates(required)['not_tested'],1)
        self.assertEqual(scenario['first_failing_stage'],'crafting.wood_pickaxe_creation')
        self.assertEqual(scenario['first_failure']['fact']['observed'],False)

    def test_eligible_and_scenario_denominators(self):
        cohorts=[self.craft('pass') for _ in range(6)]+[self.craft('fail') for _ in range(4)]
        persistence=[self.required(stages)[1] for stages,_,_ in cohorts]
        stats=atomic_rates(persistence)
        self.assertEqual((stats['eligible'],stats['total'],stats['pass'],stats['not_tested'],stats['unknown']), (6,10,6,4,0))
        self.assertEqual(stats['success_rate'],1.)
        rows=[{'scenario':'craft_wood_pickaxe','scenario_result':scenario,'measurements_by_branch':[stages]} for stages,scenario,_ in cohorts]
        summary=summarize_measurements(rows)
        self.assertEqual(summary['overall_scenarios']['n'],10)
        self.assertEqual(summary['overall_scenarios']['pass'],6)
        self.assertEqual(summary['overall_scenarios']['fail'],4)
        self.assertEqual(summary['overall_scenarios']['success_rate_lower'],.6)
        self.assertEqual(summary['by_category']['Inventory persistence']['required']['eligible'],6)

    def test_unrelated_diagnostics_do_not_change_required_results(self):
        good,overall_good,primary_good=self.craft()
        bad,overall_bad,primary_bad=self.craft(unrelated=9)
        self.assertEqual(primary_good,primary_bad)
        self.assertEqual(self.required(good),self.required(bad))
        self.assertEqual(overall_good,overall_bad)
        self.assertTrue(any(s['status']=='fail' for s in bad if not s['required']))

    def test_required_failure_wins_over_unknown_diagnostic(self):
        stages,scenario,_=self.craft('pass','fail',unrelated=None)
        # Inject an unrelated UNKNOWN diagnostic, explicitly not required.
        stages.append(dict(name='unrelated',required=False,status='unknown'))
        self.assertEqual(overall(stages)['status'],'fail')
        self.assertEqual(scenario['first_failure']['timestep'],5)
        self.assertEqual(scenario['first_failure']['fact']['expected'],1)
        self.assertEqual(scenario['first_failure']['fact']['observed'],0)

    def test_names_categories_reasons_and_prerequisites_are_explicit(self):
        stages,_,_=self.craft()
        self.assertEqual(len({s['name'] for s in stages}),len(stages))
        for s in stages:
            self.assertIn(s['category'],CATEGORIES)
            for key in ('eligible','reason','timestep','expected','observed','prerequisites'):self.assertIn(key,s)
        persistence=self.required(stages)[1]
        self.assertEqual(persistence['prerequisites'][0]['mechanic'],'crafting.wood_pickaxe_creation')
        self.assertEqual(persistence['prerequisites'][1]['observed'],1)

    def test_saved_trajectory_replays_every_dependency_result(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from mechanics_eval.scenarios import Engine
        from mechanics_eval.rollouts import reference_pair
        from mechanics_eval.evidence import save_case,dump_json
        from mechanics_eval.milestone import score_run
        from mechanics_eval.visual import hud_patch
        from mechanics_eval.gifs import export_gifs
        engine=Engine();spec=next(s for s in catalog() if s.name=='craft_wood_pickaxe')
        refs=reference_pair(engine,spec,0,1,16)
        predictions=[dict(frames=r.frames.copy(),rewards=None,dones=None) for r in refs]
        hud_patch(predictions[0]['frames'][4],'wood_pickaxe')[:]=0
        with TemporaryDirectory() as temp:
            root=Path(temp);source=root/'evidence';source.mkdir();case='craft_wood_pickaxe_s0_m0'
            save_case(source/case,spec,refs,predictions,0,123,0,'valid_crafting')
            dump_json(source/'manifest.json',{'status':'complete','cases':[case],'config':{'context':1,'horizon':16}})
            first=score_run(source,root/'first');again=score_run(source,root/'again')
            self.assertEqual(first,again)
            self.assertEqual((root/'first/results.json').read_bytes(),(root/'again/results.json').read_bytes())
            row=first['samples'][0]
            self.assertEqual(row['outcome']['status'],'pass')
            self.assertEqual(row['scenario_result']['status'],'fail')
            import csv
            with (root/'first/samples.csv').open() as stream:
                record=next(csv.DictReader(stream))
            self.assertEqual(record['status'],'pass')
            self.assertEqual(record['scenario_status'],'fail')
            self.assertEqual(first['mechanic_performance']['scenarios'][0]['passed'],0)
            self.assertEqual(row['scenario_result']['first_failing_stage'],'inventory_persistence.wood_pickaxe')
            self.assertEqual(row['scenario_result']['first_failure']['timestep'],5)
            export_gifs([source],[root/'first/results.json'],root/'gallery')
            page=(root/'gallery/index.html').read_text()
            self.assertIn('Scenario overall: FAIL',page)
            self.assertIn('crafting.wood_pickaxe_creation',page)
            self.assertIn('inventory_persistence.wood_pickaxe',page)
