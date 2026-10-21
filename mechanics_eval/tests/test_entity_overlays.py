from mechanics_eval.tests.paths import RUNS,OUTPUT
"""Entity overlays must neither hide visible terrain evidence nor invent hidden terrain."""
import unittest
import numpy as np
from mechanics_eval.reader_v2 import ReaderV2
from mechanics_eval.scenarios import Engine, catalog, u8
from mechanics_eval.gifs import suite_design


class EntityOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Engine()
        cls.state = cls.engine.construct(catalog()[0], 83, 1)
        cls.reader = ReaderV2(cls.engine, cls.state)

    def test_real_rendered_entities_on_all_terrain_and_lighting(self):
        accepted = 0
        for entity in ('cows','zombies','skeletons'):
            mobs = getattr(self.state,entity)
            mobs = mobs.replace(position=mobs.position.at[0].set(np.array([32,33])),
                                mask=mobs.mask.at[0].set(True))
            for tile in range(2,17):
                for light in (.5,.75,1.):
                    state = self.state.replace(**{entity:mobs},map=self.state.map.at[32,33].set(tile),
                                               light_level=np.float32(light))
                    frame = u8(self.engine.render(state))
                    read = self.reader.tile(frame[21:28,35:42])
                    with self.subTest(entity=entity,tile=tile,light=light):
                        if read['value'] is not None:
                            accepted += 1
                            self.assertEqual(read['value'],tile)
        self.assertGreater(accepted,100)

    def test_unreadable_overlay_is_not_a_free_pass(self):
        for value in (0,127,255):
            self.assertIsNone(self.reader.tile(np.full((7,7,3),value,np.uint8))['value'])
        rng = np.random.default_rng(923)
        for _ in range(30):
            self.assertIsNone(self.reader.tile(rng.integers(0,256,(7,7,3),dtype=np.uint8))['value'])

    def test_entity_does_not_excuse_visible_resource_respawn(self):
        from mechanics_eval.atomic import consistency
        mobs = self.state.zombies
        mobs = mobs.replace(position=mobs.position.at[0].set(np.array([32,33])),
                            mask=mobs.mask.at[0].set(True))
        reads = []
        for tile in (5,2,5):
            state = self.state.replace(map=self.state.map.at[32,33].set(tile),
                                       zombies=mobs,light_level=np.float32(1.))
            reads.append(self.reader.read(u8(self.engine.render(state))))
        result = consistency('harvested_resource_persistence',reads,[5,0],anchor=1,original_tile=5)
        self.assertEqual(result['status'],'fail')

    def test_saved_zombie_over_grass_recovered_from_pixels(self):
        from pathlib import Path
        path = (RUNS/'evidence/focused/consumed_resource_stays_gone_s2_m0/trajectory.npz')
        if not path.exists():self.skipTest('local generated evidence unavailable')
        with np.load(path) as arrays:
            read = self.reader.tile(arrays['b0_generated'][9,21:28,35:42])
        self.assertEqual(read['value'],2)
        self.assertEqual(read['entity'],'zombie')
        self.assertEqual(read['reason'],'terrain_visible_around_entity')

    def test_benchmark_counts_are_not_branch_or_measurement_counts(self):
        rows = [dict(scenario=name,seed=seed,model_sample=0,scope='static')
                for name in ('a','b') for seed in range(10)]
        summary = suite_design(rows,'focused')
        self.assertIn('2 scenario types × 10 engine seeds × 1 model sample(s) = 20 attempts',summary)
        self.assertIn('40 paired branch recordings',summary)
        self.assertIn('static: 2 types / 20 attempts',summary)
        self.assertIn('uneven sampling',suite_design(rows[:-1],'partial'))
