"""Positive RGB blocker evidence, rejection independence, NA semantics and conditional denominators."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import json
import unittest
import jax
import numpy as np
from mechanics_eval.scenarios import Engine,catalog,u8
from mechanics_eval.reader_v2 import ReaderV2
from mechanics_eval.rollouts import reference_pair
from mechanics_eval.rejection import departure_eligibility,retry_selection,sampling_seed
from mechanics_eval.atomic import short_tests,contrast,scalar
from mechanics_eval.dependencies import build,atomic_rates,measurement,overall,not_applicable
from mechanics_eval.persistence import visible_persistence
from mechanics_eval.motion import object_observation,track
from mechanics_eval.evidence import save_case,dump_json,digest,load_case
from mechanics_eval.milestone import score_run


class RejectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine=Engine();cls.spec=next(s for s in catalog() if s.name=='table_leave_return')
        cls.refs=reference_pair(cls.engine,cls.spec,4,1,16)
        cls.reader=ReaderV2(cls.engine,cls.engine.construct(cls.spec,100000,8))
        cls.correct=[cls.reader.read(cls.refs[0].context[0])]+[cls.reader.read(f) for f in cls.refs[0].frames]
        cls.blocked_frames=cls.frames_with_obstacle(28,initial=True)
        cls.blocked=[cls.reader.read(cls.refs[0].context[0])]+[cls.reader.read(f) for f in cls.blocked_frames]

    @classmethod
    def frames_with_obstacle(cls,row,initial):
        state=cls.engine.construct(cls.spec,4,1)
        if initial:state=state.replace(map=state.map.at[row,32].set(5))
        np.testing.assert_array_equal(u8(cls.engine.render(state)),cls.refs[0].context[0])
        frames=[]
        for t,a in enumerate(cls.refs[0].actions):
            frame,state,*_=cls.engine.step(jax.random.fold_in(jax.random.PRNGKey(4),t+1),state,int(a))
            if not initial and t==2:
                state=state.replace(map=state.map.at[row,32].set(5));frame=cls.engine.render(state)
            frames.append(u8(frame))
        return np.stack(frames)

    def score_reads(self,reads,branch=0,conditional=False):
        ref=self.refs[branch];schema=asdict(self.spec)
        tiles=[{'3,5':int(s.map[tuple(np.asarray(s.player_position)+[0,1])])} for s in ref.states]
        primary,diag=short_tests(schema,reads,ref.actions,ref.facts,ref.initial,tiles)
        stages,result=build(schema,reads,ref.actions,primary,diag,ref.facts,ref.initial,conditional=conditional)
        return primary,{s['name']:s for s in stages},result

    def test_legitimate_newly_revealed_obstacle_rejected(self):
        result=departure_eligibility(self.blocked)
        self.assertEqual(result['decision'],'REJECTED_BLOCKED_ROUTE')
        for witness in result['witnesses']:
            self.assertGreater(witness['visibility_history'][0]['timestep'],0)
            self.assertEqual(witness['world_relative_coordinate'],[-4,0])
            self.assertTrue(all(h['tile']==5 for h in witness['visibility_history']))

    def test_frozen_wrong_and_teleport_departures_never_retry(self):
        frozen=[self.correct[0]]+[self.correct[1]]*16
        self.assertEqual(departure_eligibility(frozen)['reason'],'preserve_departure_failure')
        wrong=[self.correct[t] for t in [0,1,2,3,5,6,7,7]]
        result=departure_eligibility(wrong)
        self.assertEqual(result['decision'],'ELIGIBLE')
        self.assertTrue(any(f['reason']=='wrong_direction_or_teleport' for f in result['failure_evidence']))

    def test_movement_during_placement_cannot_be_filtered_out(self):
        reads=[self.blocked[t] for t in (0,2,3,4,5,6,7,7)]
        result=departure_eligibility(reads)
        self.assertEqual(result['decision'],'ELIGIBLE')
        self.assertTrue(any(f['reason']=='movement_during_placement' for f in result['failure_evidence']))

    def test_previously_visible_free_change_cannot_trigger_retry(self):
        frames=self.frames_with_obstacle(29,initial=False)
        reads=[self.correct[0]]+[self.reader.read(f) for f in frames]
        result=departure_eligibility(reads)
        self.assertEqual(result['decision'],'ELIGIBLE')
        self.assertTrue(any(f['reason']=='previously_visible_free_tile_became_solid' for f in result['failure_evidence']))

    def test_return_and_object_outcomes_do_not_influence_selection(self):
        for base in (self.correct,self.blocked):
            changed=deepcopy(base)
            for t in range(8,17):changed[t]=self.reader.read(np.full((63,63,3),127,np.uint8))
            self.assertEqual(departure_eligibility(base),departure_eligibility(changed))
        self.assertEqual(departure_eligibility(self.correct)['decision'],'ELIGIBLE')

    def test_unreadable_collision_and_first_visibility_stay_eligible_unknown(self):
        unknown=deepcopy(self.blocked);unknown[4]['tiles']['2,4']=None
        result=departure_eligibility(unknown)
        self.assertEqual(result['decision'],'ELIGIBLE')
        self.assertTrue(result['ambiguities'])
        # A previously visible but unreadable cell is not newly revealed later.
        unknown=deepcopy(self.blocked);unknown[2]['tiles']['0,4']=None
        self.assertEqual(departure_eligibility(unknown)['decision'],'ELIGIBLE')

    def test_seed_schedule_budget_and_failed_eligible_selection(self):
        self.assertEqual([sampling_seed(100,i) for i in range(3)],[100,101,102])
        self.assertEqual(sampling_seed(2**32-1,1),0)
        self.assertEqual(len({sampling_seed(42,i,3) for i in range(20)}),20)
        with self.assertRaises(ValueError):sampling_seed(1,1,2)
        rejected=dict(eligibility={'decision':'REJECTED_BLOCKED_ROUTE'})
        failed=dict(eligibility={'decision':'ELIGIBLE'},outcome='fail')
        self.assertEqual(retry_selection([rejected,rejected,failed],3)['selected_index'],2)
        self.assertEqual(retry_selection([rejected]*10,3)['status'],'RETRY_BUDGET_EXHAUSTED')
        self.assertEqual(retry_selection([rejected]*10,3)['retries'],2)

    def test_alternative_without_placement_is_not_applicable(self):
        other=[self.reader.read(self.refs[1].context[0])]+[self.reader.read(f) for f in self.refs[1].frames]
        left,_,_=self.score_reads(self.correct)
        right,stages,result=self.score_reads(other,1)
        self.assertEqual(right['status'],'pass');self.assertEqual(contrast(left,right)['status'],'pass')
        self.assertEqual(result['status'],'pass')
        for name in ('object_persistence.visible','object_persistence.after_return','movement.fixed_route_fidelity'):
            self.assertEqual(stages[name]['status'],'not_applicable')
            counts=atomic_rates([stages[name]])
            self.assertEqual((counts['eligible'],counts['applicable'],counts['not_applicable'],counts['not_tested']),(0,0,1,0))
            self.assertIsNone(counts['success_rate'])

    def test_noop_and_harvest_alternative_rules_are_not_applicable(self):
        for name,metric in [('noop','inventory_accounting.noop_inventory_unchanged'),
                            ('consumed_resource_stays_gone','object_persistence.harvested_resource')]:
            spec=next(s for s in catalog() if s.name==name);ref=reference_pair(self.engine,spec,4,1,16)[1]
            reads=[self.reader.read(ref.context[0])]+[self.reader.read(f) for f in ref.frames]
            tiles=[{'3,5':int(s.map[tuple(np.asarray(s.player_position)+[0,1])])} for s in ref.states]
            primary,diag=short_tests(asdict(spec),reads,ref.actions,ref.facts,ref.initial,tiles)
            stages,result=build(asdict(spec),reads,ref.actions,primary,diag,ref.facts,ref.initial)
            self.assertEqual(next(s for s in stages if s['name']==metric)['status'],'not_applicable')
            self.assertEqual(result['status'],'pass')

    def test_missing_target_position_is_not_a_disappearance_claim(self):
        from mechanics_eval.unknown_audit import diagnose
        raw=dict(mechanic='old_tracked_table',status='unknown',milestone=13,motion_trace=[dict(position=None,reason='insufficient_distinct_landmarks')]*14)
        fact=dict(field='tile:tracked_table',expected=11,observed=None,reason='target_unreadable')
        explanation=diagnose(raw,{},fact,13,self.correct,self.refs[0].actions)
        self.assertEqual(explanation['component'],'motion_tracker')
        self.assertEqual(explanation['target']['target_coordinate'],None)
        self.assertEqual(explanation['target']['observed_tile'],None)
        self.assertEqual(explanation['root_cause'],'insufficient_distinct_landmarks')

    def test_visible_persistence_skips_out_of_view_and_reads_occlusion(self):
        raw=visible_persistence(self.correct,11)
        self.assertEqual(raw['status'],'pass');self.assertGreater(raw['out_of_view_observations'],0)
        unknown=deepcopy(self.correct);unknown[2]['tiles']['4,5']=None
        unknown[2]['measurements']['tile:4,5']['reason']='outside_calibrated_support'
        self.assertEqual(visible_persistence(unknown,11)['status'],'unknown')
        state=self.refs[0].states[0];mobs=state.cows
        mobs=mobs.replace(position=mobs.position.at[0].set(np.array([32,33])),mask=mobs.mask.at[0].set(True))
        read=self.reader.read(u8(self.engine.render(state.replace(cows=mobs))))
        obs=object_observation(read,[0,0],expected_tile=11)
        self.assertEqual(obs['observed_tile'],11);self.assertEqual(obs['entity_occlusion']['entity'],'cow')
        self.assertEqual(object_observation(read,None)['localization_status'],'unknown')
        self.assertEqual(object_observation(read,[-6,0])['target_visibility'],'out_of_view')
        for key in ('target_coordinate','estimated_player_position','tile_reader_status','tile_reader_reason','confidence'):
            self.assertIn(key,obs)

    def test_shorter_trip_cannot_pass_conditional_full_route(self):
        indices=[0,1,2,3,4,4,4,4,3,2,1,1,1,1,1,1,1]
        reads=[self.correct[t] for t in indices]
        _,stages,result=self.score_reads(reads,conditional=True)
        self.assertEqual(stages['movement.fixed_route_fidelity']['status'],'fail')
        self.assertTrue(stages['movement.fixed_route_fidelity']['required'])
        self.assertEqual(stages['object_persistence.after_return']['status'],'not_tested')
        self.assertEqual(result['status'],'fail')

    def test_chronological_first_failure_is_not_declaration_order(self):
        late=measurement(scalar('late','value',1,0,milestone=13),'late','Movement',True)
        early=measurement(scalar('early','value',1,0,milestone=2),'early','Object persistence',True)
        self.assertEqual(overall([late,early])['first_failing_stage'],'early')
        raw=scalar('unordered','x',1,0);raw['facts']=[dict(raw['facts'][0],timestep=13),dict(raw['facts'][0],timestep=2)]
        measured=measurement(raw,'unordered','Movement',True)
        self.assertEqual(overall([measured])['first_failure']['fact']['timestep'],2)

    def fixture(self,root,blocked):
        source=root/'first';source.mkdir();model=root/'model.bin';model.write_bytes(b'fake test checkpoint')
        module=root/'module.py';module.write_text('# fake adapter for sampling controls\n')
        frames=self.blocked_frames if blocked else np.repeat(self.refs[0].context,16,axis=0)
        preds=[dict(frames=frames,rewards=None,dones=None),dict(frames=self.refs[1].frames,rewards=None,dones=None)]
        name='table_leave_return_s4_m0';save_case(source/name,self.spec,self.refs,preds,4,100,0,'placement')
        dump_json(source/'manifest.json',dict(status='complete',cases=[name],config=dict(module='train',fp32=False),
            provenance=dict(checkpoint_sha256=digest(model),module_sha256=digest(module),sampling_settings={})))
        return source,model,module

    def test_cached_scoring_matches_fresh_rgb(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);source,_,_=self.fixture(root,False)
            first=score_run(source,root/'fresh');again=score_run(source,root/'cached',root/'fresh')
            self.assertEqual(first['samples'],again['samples'])
            self.assertEqual(first['reference_reader_validation'],again['reference_reader_validation'])

    def test_failed_first_sample_is_kept_without_loading_a_model(self):
        from mechanics_eval.conditional import run
        with TemporaryDirectory() as temp:
            root=Path(temp);source,model,_=self.fixture(root,False)
            args=SimpleNamespace(out=root/'experiment',first_evidence=source,first_measurements=None,model=model,module='train',
                                 fp32=False,max_samples=3,seed_step=1)
            with patch('mechanics_eval.conditional.ModuleAdapter',side_effect=AssertionError('must not retry a failure')):run(args)
            report=json.loads((args.out/'summary.json').read_text())
            self.assertEqual((report['total_generated_attempts'],report['newly_generated_attempts'],report['selected_eligible']),(1,0,1))
            self.assertEqual(report['conditional']['fail'],1)
            self.assertEqual(report['unconditional']['n'],1)
            from mechanics_eval.gifs import export_gifs
            export_gifs([args.out/'evidence'],[args.out/'gallery-results.json'],root/'gallery')
            page=(root/'gallery/index.html').read_text()
            self.assertIn('Unconditional first samples',page);self.assertIn('Conditional selected eligible',page)
            self.assertIn('data-view="ordinary"',page);self.assertIn('data-first-status=',page)
            self.assertIn('target_coordinate',page);self.assertIn('Departure-only eligibility proof',page)

    def test_exhaustion_reproducibility_and_paired_keys(self):
        from mechanics_eval.conditional import run
        from mechanics_eval.rollouts import EvaluationSession
        with TemporaryDirectory() as temp:
            root=Path(temp);source,model,module=self.fixture(root,True);calls=[]
            owner=self
            class Adapter:
                def __init__(self,*args):self.module=SimpleNamespace(__file__=str(module))
                def rollout(self,context,incoming,actions,seed,heads):
                    calls.append(seed)
                    frames=owner.blocked_frames if int(actions[0,0])==8 else owner.refs[1].frames
                    return dict(frames=frames[None],rewards=None,dones=None)
            for suffix in ('a','b'):
                args=SimpleNamespace(out=root/suffix,first_evidence=source,first_measurements=None,model=model,module='train',fp32=False,max_samples=3,seed_step=1)
                session=EvaluationSession('train',model,False) if suffix=='b' else None
                with patch('mechanics_eval.conditional.ModuleAdapter',Adapter), \
                     patch('mechanics_eval.rollouts.ModuleAdapter',side_effect=Adapter) as shared_adapter:
                    run(args,session=session)
                    self.assertEqual(shared_adapter.call_count,1 if session else 0)
                report=json.loads((args.out/'summary.json').read_text())
                self.assertEqual(report['retry_budget_exhausted'],1)
                self.assertEqual(report['selected_eligible'],0);self.assertEqual(report['conditional']['n'],0)
                self.assertEqual(report['unconditional']['n'],1)
                self.assertEqual(report['seeds'][0]['selection']['status'],'RETRY_BUDGET_EXHAUSTED')
                self.assertEqual(report['total_generated_attempts'],3)
            self.assertEqual(calls,[101,101,102,102]*2)
            for i in range(3):
                _,a=load_case(root/'a/evidence'/f'table_leave_return_s4_m{i}')
                _,b=load_case(root/'b/evidence'/f'table_leave_return_s4_m{i}')
                np.testing.assert_array_equal(a['context'],b['context']);np.testing.assert_array_equal(a['b0_generated'],b['b0_generated'])
                sampling=json.loads((root/'a/evidence'/f'table_leave_return_s4_m{i}'/'sampling.json').read_text())
                self.assertEqual(sampling['branches']['0']['step_keys_uint32'],sampling['branches']['1']['step_keys_uint32'])
