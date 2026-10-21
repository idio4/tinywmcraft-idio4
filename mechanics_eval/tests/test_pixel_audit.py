"""Controls for the independent audit, which never changes benchmark scores."""
import unittest
import numpy as np
from mechanics_eval.audit.pixels import align


class PixelAuditTests(unittest.TestCase):
    def frames(self):
        rng=np.random.default_rng(471)
        world=rng.integers(0,256,(9,11,7,7,3),dtype=np.uint8)
        def image(r,c):
            return world[r:r+7,c:c+9].transpose(0,2,1,3,4).reshape(49,63,3)
        return image

    def test_known_shift_and_frozen_frames(self):
        image=self.frames()
        result=align(image(1,1),image(0,1))
        self.assertTrue(result['accepted']);self.assertEqual(result['shift'],[-1,0])
        frozen=align(image(1,1),image(1,1))
        self.assertTrue(frozen['accepted']);self.assertEqual(frozen['shift'],[0,0])

    def test_missing_table_does_not_localize_itself(self):
        image=self.frames();before=image(1,1).copy();after=before.copy()
        before[21:28,35:42]=255;after[21:28,35:42]=0
        result=align(before,after)
        self.assertTrue(result['accepted']);self.assertEqual(result['shift'],[0,0])

    def test_repeated_and_corrupt_terrain_abstains(self):
        rng=np.random.default_rng(812)
        patch=rng.integers(0,256,(7,7,3),dtype=np.uint8)
        frame=np.tile(patch,(7,9,1))
        self.assertFalse(align(frame,frame)['accepted'])
        image=self.frames()
        self.assertFalse(align(image(1,1),rng.integers(0,256,(49,63,3),dtype=np.uint8))['accepted'])

    def test_real_craftax_rendered_camera_translation(self):
        from mechanics_eval.focused import VariedEngine,variants
        from mechanics_eval.scenarios import catalog,u8
        engine=VariedEngine();state=engine.construct(variants(catalog()[0],0),0,1)
        before=u8(engine.render(state))
        after=u8(engine.render(state.replace(player_position=state.player_position+np.array([-1,0]))))
        result=align(before,after)
        self.assertTrue(result['accepted']);self.assertEqual(result['shift'],[-1,0])
