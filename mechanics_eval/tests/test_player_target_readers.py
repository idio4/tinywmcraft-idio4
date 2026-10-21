from mechanics_eval.tests.paths import RUNS,OUTPUT
"""RGB pose and bounded target identity controls, independent of expected actions."""
import unittest
from pathlib import Path
import numpy as np
from mechanics_eval.scenarios import Engine, catalog, u8
from mechanics_eval.reader_v2 import ReaderV2
from mechanics_eval.player_reader import read_player
from mechanics_eval.target_reader import target_zombie


class PlayerTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Engine()
        cls.state = cls.engine.construct(catalog()[0], 83, 1)
        cls.reader = ReaderV2(cls.engine, cls.state)

    def test_real_pose_lighting_and_sleep_controls(self):
        for light in (.5,.75,1.):
            for direction in range(1,5):
                for sleeping in (False,True):
                    state = self.state.replace(player_direction=np.int32(direction),is_sleeping=np.bool_(sleeping),light_level=np.float32(light))
                    result = read_player(u8(self.engine.render(state)))
                    self.assertEqual(result['sleeping'],sleeping)
                    self.assertEqual(result['direction'],None if sleeping else direction)

    def test_dark_awake_is_not_sleep_and_corrupt_pose_abstains(self):
        frame=u8(self.engine.render(self.state))
        lum=(frame[:49].astype(float)*[.299,.587,.114]).sum(-1,keepdims=True)
        dark=frame.copy();dark[:49]=(.5*lum+np.array([0,0,8])).astype(np.uint8)
        self.assertIsNone(read_player(dark)['sleeping'])
        for value in (0,127,255):
            bad=frame.copy();bad[21:28,28:35]=value
            self.assertIsNone(read_player(bad)['sleeping'])
        for light in (0.,.1,.4):
            night=u8(self.engine.render(self.state.replace(light_level=np.float32(light))))
            self.assertIsNot(read_player(night)['sleeping'],True)

    def evidence(self,name):
        path=(RUNS/'evidence/existing')/(name+'_s0_m0')/'trajectory.npz'
        if not path.exists():self.skipTest('saved generated RGB unavailable')
        with np.load(path) as a:return {k:a[k] for k in a.files}

    def test_generated_sleep_and_awake(self):
        a=self.evidence('sleep')
        for frame in a['b0_generated']:
            self.assertTrue(read_player(frame)['sleeping'])
        for frame in a['b1_generated']:
            self.assertFalse(read_player(frame)['sleeping'])

    def test_real_kill_generated_survival_frozen_and_unknown(self):
        a=self.evidence('attack_zombie')
        for key,expected in [('b0_reference',True),('b1_reference',False),('b0_generated',False),('b1_generated',False)]:
            reads=[self.reader.read(a['context'][-1]),self.reader.read(a[key][0])]
            target_zombie(reads);self.assertEqual(reads[1]['target_zombie_killed'],expected)
        ctx=a['context'][-1]
        reads=[self.reader.read(ctx),self.reader.read(ctx)]
        target_zombie(reads);self.assertFalse(reads[1]['target_zombie_killed'])
        bad=a['b0_reference'][0].copy();bad[14:21,35:42]=127
        reads=[self.reader.read(ctx),self.reader.read(bad)]
        target_zombie(reads);self.assertIsNone(reads[1]['target_zombie_killed'])

    def test_unrelated_zombie_does_not_undo_kill_and_other_entity_occludes(self):
        a=self.evidence('attack_zombie');ctx=a['context'][-1]
        for entity,where,expected in [('zombies',(0,0),True),('cows',(3,5),None),('skeletons',(3,5),None)]:
            # Composite a real rendered sprite at a chosen visible position.
            mobs=getattr(self.state,entity)
            world=np.asarray(self.state.player_position)+np.array(where)-[3,4]
            mobs=mobs.replace(position=mobs.position.at[0].set(world),mask=mobs.mask.at[0].set(True))
            patch=u8(self.engine.render(self.state.replace(**{entity:mobs})))
            f=a['b0_reference'][0].copy();r,c=where;f[r*7:r*7+7,c*7:c*7+7]=patch[r*7:r*7+7,c*7:c*7+7]
            reads=[self.reader.read(ctx),self.reader.read(f)]
            target_zombie(reads);self.assertEqual(reads[1]['target_zombie_killed'],expected)

    def test_second_step_no_identity_reacquisition(self):
        a=self.evidence('attack_zombie')
        reads=[self.reader.read(a['context'][-1])]+[self.reader.read(f) for f in a['b0_generated'][:2]]
        target_zombie(reads)
        self.assertIsNone(reads[2]['target_zombie_killed'])
