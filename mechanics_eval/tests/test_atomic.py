"""Atomic target isolation: unrelated failures must not change primary verdicts."""
from copy import deepcopy
from dataclasses import asdict
import unittest
from mechanics_eval.atomic import short_tests, consistency, contrast
from mechanics_eval.scenarios import catalog
from mechanics_eval.visual import HUD


def reading(**updates):
    r={k:0 for k in HUD};r.update(health=9,tiles={'3,5':2});r.update(updates)
    r['tiles'].update({'0,1':5,'5,7':4})
    return r


class AtomicTests(unittest.TestCase):
    def score(self,name,frames,truth=None,actions=None):
        spec=asdict(next(s for s in catalog() if s.name==name))
        initial=reading(**spec['inventory']);initial.update(zombies=1,done=False,sleeping=False,row_delta=0,col_delta=0,direction=2)
        if truth is None:
            end={**initial,**spec['expected']};truth=[end for _ in range(spec['milestone'])]
        tiles=[{'3,5':spec['expected_tile'] if spec['expected_tile'] is not None else 2} for _ in truth]
        return short_tests(spec,frames,list(spec['actions']) if actions is None else actions,truth,initial,tiles)

    def test_wood_gain_ignores_tree_and_other_inventory(self):
        start=reading(tiles={'3,5':5});end=reading(wood=1,iron_sword=1,tiles={'3,5':5})
        primary,diag=self.score('gather_wood',[start,end])
        self.assertEqual(primary['status'],'pass')
        self.assertEqual(diag['target_tile']['status'],'fail')
        self.assertEqual(diag['inventory_delta:iron_sword']['status'],'fail')
        end['wood']=0;end['tiles']['3,5']=2
        primary,diag=self.score('gather_wood',[start,end])
        self.assertEqual(primary['status'],'fail');self.assertEqual(diag['target_tile']['status'],'pass')

    def test_crafting_creation_and_consumption_are_independent(self):
        start=reading(wood=1);end=reading(wood=1,wood_pickaxe=1)
        primary,diag=self.score('craft_wood_pickaxe',[start,end])
        self.assertEqual(primary['status'],'pass')
        self.assertEqual(diag['ingredient_consumption:wood@t1']['status'],'fail')
        end=reading(wood=0,wood_pickaxe=0)
        primary,diag=self.score('craft_wood_pickaxe',[start,end])
        self.assertEqual(primary['status'],'fail')
        self.assertEqual(diag['ingredient_consumption:wood@t1']['status'],'pass')

    def test_kill_unknown_despite_sword_failure_or_other_spawns(self):
        start=reading(iron_sword=1);end=reading(iron_sword=0,zombies=20)
        primary,diag=self.score('attack_zombie',[start,end])
        self.assertEqual(primary['status'],'unknown')
        self.assertEqual(diag['tool_consistency:iron_sword']['status'],'fail')
        end.update(target_zombie_killed=True)
        self.assertEqual(self.score('attack_zombie',[start,end])[0]['status'],'pass')
        end['target_zombie_killed']=False
        self.assertEqual(self.score('attack_zombie',[start,end])[0]['status'],'fail')

    def test_counterfactual_uses_only_tool_creation(self):
        start=reading(wood=1)
        selected,_=self.score('craft_wood_pickaxe',[start,reading(wood=1,wood_pickaxe=1)])
        truth=[{**start,'wood':1,'wood_pickaxe':0}]
        other,_=self.score('craft_wood_pickaxe',[start,reading(wood=8)],truth,actions=[0])
        self.assertEqual(other['status'],'pass')
        self.assertEqual(contrast(selected,other)['status'],'pass')

    def test_failed_persistence_does_not_rewrite_creation(self):
        start=reading(wood=1);created=reading(wood_pickaxe=1);gone=reading()
        primary,_=self.score('craft_wood_pickaxe',[start,created,gone])
        persistence=consistency('crafted_tool_persistence',[start,created,gone],[11,0],field='wood_pickaxe',anchor=1)
        self.assertEqual(primary['status'],'pass');self.assertEqual(persistence['status'],'fail')
        missing=consistency('crafted_tool_persistence',[start,gone,gone],[11,0],field='wood_pickaxe',anchor=1)
        self.assertEqual(missing['status'],'unknown')

    def test_each_static_consistency_rule_is_independent(self):
        start=reading(tiles={'3,5':5});harvest=reading(wood=1,tiles={'3,5':2});later=reading(wood=0,tiles={'3,5':2})
        self.assertEqual(consistency('harvested_resource_persistence',[start,harvest,later],[5,0],anchor=1,original_tile=5)['status'],'pass')
        self.assertEqual(consistency('noop_inventory_unchanged',[harvest,later],[0])['status'],'fail')
        table=reading(tiles={'3,5':11});extra_mob=deepcopy(table);extra_mob['zombies']=9
        self.assertEqual(consistency('placed_table_persistence',[start,table,extra_mob],[8,0],anchor=1)['status'],'pass')
        extra_mob['tiles']['3,5']=None
        self.assertEqual(consistency('placed_table_persistence',[start,table,extra_mob],[8,0],anchor=1)['status'],'unknown')

    def test_unverified_camera_does_not_fail_object_persistence(self):
        start=reading();table=reading(tiles={'3,5':11});moved=reading(tiles={'3,5':2})
        moved['tiles']['0,1']=4
        test=consistency('placed_table_persistence',[start,table,moved],[8,0],anchor=1)
        self.assertEqual(test['status'],'unknown')

    def test_real_engine_atomic_targets_and_four_consistency_rules(self):
        from mechanics_eval.scenarios import Engine
        from mechanics_eval.rollouts import reference_pair
        from mechanics_eval.reader_v2 import ReaderV2
        from mechanics_eval.scenarios import u8
        engine=Engine();specs={s.name:s for s in catalog()}
        reader=ReaderV2(engine,engine.construct(specs['gather_wood'],100,1))
        for name,rule,field in [('gather_wood','harvested_resource_persistence',None),
                                ('craft_wood_pickaxe','crafted_tool_persistence','wood_pickaxe'),
                                ('place_table','placed_table_persistence',None)]:
            spec=specs[name];refs=reference_pair(engine,spec,0,1,4)
            r=refs[0];reads=[reader.read(r.context[0])]+[reader.read(f) for f in r.frames]
            tiles=[{'3,5':int(s.map[32,33])} for s in r.states]
            primary,_=short_tests(asdict(spec),reads,r.actions,r.facts,r.initial,tiles)
            self.assertEqual(primary['status'],'pass',name)
            test=consistency(rule,reads,r.actions,field=field,anchor=1,original_tile=spec.front)
            self.assertEqual(test['status'],'pass',rule)
            noop=refs[1]
            reads=[reader.read(noop.context[0])]+[reader.read(f) for f in noop.frames]
            self.assertEqual(consistency('noop_inventory_unchanged',reads,noop.actions)['status'],'pass')

    def test_table_loss_and_resource_respawn_fail_only_their_rule(self):
        start=reading(tiles={'3,5':5});table=reading(tiles={'3,5':11});gone=reading(tiles={'3,5':2})
        self.assertEqual(consistency('placed_table_persistence',[start,table,gone],[8,0],anchor=1)['status'],'fail')
        harvested=reading(wood=1,tiles={'3,5':2});respawn=reading(wood=1,tiles={'3,5':5})
        self.assertEqual(consistency('harvested_resource_persistence',[start,harvested,respawn],[5,0],anchor=1,original_tile=5)['status'],'fail')
        self.assertEqual(consistency('noop_inventory_unchanged',[harvested,respawn],[0])['status'],'pass')


    def test_stochastic_sapling_difference_is_not_a_failure(self):
        primary,diag=self.score('gather_empty',[reading(),reading(sapling=1)])
        self.assertEqual(primary['status'],'pass')
        self.assertEqual(diag['inventory_delta:sapling']['status'],'unknown')
        self.assertEqual(diag['inventory_delta:sapling']['observed_delta'],1)
