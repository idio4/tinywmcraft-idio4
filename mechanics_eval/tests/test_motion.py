from mechanics_eval.tests.paths import RUNS,OUTPUT
"""Independent motion controls and real RGB fixtures; expected states only score observed geometry."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import unittest
import numpy as np
from mechanics_eval.motion import estimate,track,displacement,object_observation
from mechanics_eval.scenarios import Engine,catalog,u8
from mechanics_eval.reader_v2 import ReaderV2
from mechanics_eval.rollouts import reference_pair
from mechanics_eval.atomic import short_tests
from mechanics_eval.dependencies import build


def view(grid):
    return dict(tiles={f'{r},{c}':int(grid[r,c]) for r in range(7) for c in range(9) if (r,c)!=(3,4)})


class GridMotionTests(unittest.TestCase):
    def test_translation_sign_and_wrong_shift_fails(self):
        rng=np.random.default_rng(791)
        grid=rng.choice([2,4,5,7,8,9,13],size=(15,19))
        before=view(grid[4:11,5:14]);after=view(grid[5:12,7:16])
        measured=estimate(before,after)
        self.assertTrue(measured['accepted']);self.assertEqual(measured['displacement'],[1,2])
        self.assertEqual(measured['camera_translation'],[-1,-2])
        scored=displacement([before,after],1,[0,1])
        self.assertEqual(scored['status'],'fail')
        self.assertEqual(scored['facts'][0]['observed'],[1,2])

    def test_repeated_and_empty_landmarks_abstain(self):
        for grid in (np.full((7,9),2),np.where(np.indices((7,9)).sum(0)%2,4,5)):
            result=estimate(view(grid),view(grid))
            self.assertFalse(result['accepted'])
            self.assertIn(result['reason'],('insufficient_distinct_landmarks','ambiguous_translation'))

    def test_one_unrelated_landmark_change_does_not_fail_motion(self):
        rng=np.random.default_rng(82);a=view(rng.choice([2,4,5,7,8,9,13],size=(7,9)))
        b=deepcopy(a);b['tiles']['0,0']=9 if a['tiles']['0,0']!=9 else 4
        self.assertEqual(estimate(a,b)['displacement'],[0,0])
        b['tiles']={k:None for k in b['tiles']}
        self.assertFalse(estimate(a,b)['accepted'])

    def test_entities_and_tested_object_cannot_supply_localization(self):
        a=view(np.full((7,9),2));a['tiles'].update({'0,0':4,'5,8':5,'2,1':11})
        b=deepcopy(a)
        b['measurements']={'tile:0,0':{'entity':'zombie'},'tile:5,8':{'entity':'cow'}}
        self.assertFalse(estimate(a,b)['accepted'])
        b=deepcopy(a);b['tiles']['0,0']=None;b['tiles']['5,8']=None
        self.assertFalse(estimate(a,b)['accepted'])

    def test_object_coordinate_uses_camera_translation(self):
        r=view(np.full((7,9),2));r['tiles']['5,5']=11
        self.assertEqual(object_observation(r,[-2,0])['value'],11)
        self.assertEqual(object_observation(r,[-6,0])['reason'],'target_out_of_view')
        self.assertEqual(object_observation(r,None)['reason'],'uncertain_localization')

    def test_local_step_does_not_invent_position_after_a_gap(self):
        rng=np.random.default_rng(290)
        a=view(rng.choice([2,4,5,7,8,9,13],size=(7,9)))
        b=view(rng.choice([2,4,5,7,8,9,13],size=(7,9)))
        result=track([a,b,b])[-1]
        self.assertIsNone(result['position'])
        self.assertTrue(result['adjacent']['accepted'])
        self.assertTrue(result['reason'].startswith('localization_gap:'))

    def test_unrelated_random_maps_do_not_invent_a_camera_shift(self):
        rng=np.random.default_rng(1463)
        for _ in range(50):
            a=view(rng.choice([2,4,5,7,8,9,13],size=(7,9)))
            b=view(rng.choice([2,4,5,7,8,9,13],size=(7,9)))
            self.assertFalse(estimate(a,b)['accepted'])

    def test_unknown_collision_does_not_hide_impossible_camera_shift(self):
        from mechanics_eval.atomic import movement
        rng=np.random.default_rng(39);grid=rng.choice([2,4,5,7,8,9,13],size=(15,19))
        a=view(grid[4:11,5:14]);b=view(grid[4:11,7:16]);a['tiles']['3,5']=None
        result=movement([a,b],[2],[dict(row_delta=0,col_delta=1)],1,[0,1])
        self.assertEqual(result['local_action_consequences']['status'],'fail')
        self.assertTrue(any(f['field']=='legal_single_step' and f['status']=='fail'
                            for f in result['local_action_consequences']['facts']))


class RealMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine=Engine();cls.specs={s.name:s for s in catalog()}
        cls.reader=ReaderV2(cls.engine,cls.engine.construct(catalog()[0],182,1))

    def reads(self,ref):return [self.reader.read(ref.context[-1])]+[self.reader.read(f) for f in ref.frames]

    def stages(self,ref,spec,reads):
        schema=asdict(spec)
        tiles=[{'3,5':int(state.map[tuple(np.asarray(state.player_position)+[0,1])])} for state in ref.states]
        primary,diag=short_tests(schema,reads,ref.actions,ref.facts,ref.initial,tiles)
        return build(schema,reads,ref.actions,primary,diag,ref.facts,ref.initial)

    def test_ten_seed_real_motion_all_directions_and_returns(self):
        records=[]
        for name in ('move_free','move_blocked','turn_loop','noop','table_leave_return'):
            spec=self.specs[name]
            for seed in range(10):
                refs=reference_pair(self.engine,spec,seed,1,16)
                for branch,ref in enumerate(refs):
                    readings=self.reads(ref);observed=track(readings)
                    for t in range(1,len(observed)):
                        truth=[ref.facts[t-1]['row_delta'],ref.facts[t-1]['col_delta']]
                        position=observed[t]['position']
                        records.append(dict(scenario=name,seed=seed,branch=branch,timestep=t,expected=truth,
                                            observed=position,reason=observed[t]['reason']))
                        if position is not None:self.assertEqual(position,truth,(name,seed,branch,t))
                    if branch==0:
                        stages,result=self.stages(ref,spec,readings)
                        self.assertEqual(result['status'],'pass',(name,seed,result))
        accepted=sum(r['observed'] is not None for r in records)
        self.assertGreaterEqual(accepted/len(records),.95)
        path=(OUTPUT/'motion-controls.json')
        path.write_text(json.dumps(dict(kind='real_rgb_motion_controls_not_model_scores',total=len(records),accepted=accepted,
                                       correct=accepted,records=records),indent=2)+'\n')

    def test_object_disappearance_does_not_hide_behind_localization(self):
        spec=self.specs['table_leave_return'];ref=reference_pair(self.engine,spec,7,1,16)[0]
        reads=self.reads(ref)
        returned=ref.states[12]
        gone=returned.replace(map=returned.map.at[32,33].set(2))
        reads[13]=self.reader.read(u8(self.engine.render(gone)))
        stages,result=self.stages(ref,spec,reads);stages={s['name']:s for s in stages}
        self.assertEqual(stages['movement.displacement']['status'],'pass')
        self.assertEqual(stages['movement.return_to_origin']['status'],'pass')
        self.assertEqual(stages['object_persistence.after_return']['status'],'fail')
        self.assertEqual(stages['object_persistence.after_return']['object_evidence']['coordinate'],[3,5])
        self.assertEqual(result['first_failing_stage'],'object_persistence.after_return')

    def test_frozen_movement_fails_without_inventing_object_failure(self):
        spec=self.specs['table_leave_return'];ref=reference_pair(self.engine,spec,3,1,16)[0]
        reads=self.reads(ref);reads=[reads[0]]+[deepcopy(reads[1]) for _ in ref.frames]
        stages,result=self.stages(ref,spec,reads);stages={s['name']:s for s in stages}
        self.assertEqual(stages['movement.displacement']['status'],'fail')
        self.assertEqual(stages['object_persistence.after_return']['status'],'not_tested')
        self.assertEqual(result['status'],'fail')

    def test_shorter_trip_does_not_change_located_object_verdict(self):
        spec=self.specs['table_leave_return'];ref=reference_pair(self.engine,spec,3,1,16)[0]
        original=self.reads(ref)
        indices=[0,1,2,3,4,4,4,4,3,2,1,1,1,1,1,1,1]
        reads=[original[t] for t in indices]
        stages,result=self.stages(ref,spec,reads);stages={s['name']:s for s in stages}
        self.assertEqual(stages['movement.displacement']['status'],'fail')
        self.assertEqual(stages['movement.return_to_origin']['status'],'pass')
        self.assertEqual(stages['object_persistence.after_return']['status'],'pass')
        self.assertEqual(result['status'],'fail')

    def test_direction_change_without_displacement(self):
        state=self.engine.construct(self.specs['noop'],8,1)
        frames=[self.reader.read(u8(self.engine.render(state.replace(player_direction=np.int32(d))))) for d in (1,2,3,4)]
        self.assertEqual([t['position'] for t in track(frames)],[[0,0]]*4)

    def test_generated_terrain_need_not_match_reference(self):
        # Another independently rendered map is a controlled stand-in for a different generated world.
        spec=self.specs['move_free'];a=reference_pair(self.engine,spec,2,1,16)[0]
        b=reference_pair(self.engine,spec,9,1,16)[0]
        self.assertFalse(np.array_equal(a.context,b.context))
        stages,result=self.stages(a,spec,self.reads(b))
        self.assertEqual(result['status'],'pass')

    def test_legitimate_newly_revealed_obstacle_is_not_a_movement_failure(self):
        import jax
        spec=self.specs['table_leave_return'];ref=reference_pair(self.engine,spec,4,1,16)[0]
        state=self.engine.construct(spec,4,1)
        # Initially off-screen; another generated world may legitimately contain this tree.
        state=state.replace(map=state.map.at[28,32].set(5))
        reads=[self.reader.read(u8(self.engine.render(state)))]
        np.testing.assert_array_equal(u8(self.engine.render(state)),ref.context[0])
        for t,action in enumerate(ref.actions):
            frame,state,*_=self.engine.step(jax.random.fold_in(jax.random.PRNGKey(4),t+1),state,int(action))
            reads.append(self.reader.read(u8(frame)))
        stages,result=self.stages(ref,spec,reads);stages={s['name']:s for s in stages}
        self.assertEqual(stages['movement.displacement']['status'],'unknown')
        self.assertEqual(stages['movement.displacement']['observation_reason'],'reference_route_not_comparable')
        self.assertEqual(stages['movement.action_consequences']['status'],'pass')
        self.assertEqual(stages['object_persistence.after_return']['status'],'not_tested')
        self.assertEqual(result['status'],'unknown')

    def test_table_furnace_and_harvest_follow_tracked_coordinates(self):
        from dataclasses import replace
        for name,mechanic in [('place_table','crafting_table'),('place_furnace','furnace'),('gather_wood','harvested_resource')]:
            spec=replace(self.specs[name],actions=(self.specs[name].actions[0],3,4))
            ref=reference_pair(self.engine,spec,6,1,16)[0]
            stages,_=self.stages(ref,spec,self.reads(ref));stage=next(s for s in stages if s['name']=='object_persistence.'+mechanic)
            self.assertEqual(stage['status'],'pass',name)
            self.assertEqual(stage['facts'][0]['object_evidence']['coordinate'],[4,5])
