"""Regressions for single-observation initialization and offline recordings."""
import json
import os
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image
from mechanics_eval.evidence import save_case, dump_json
from mechanics_eval.gifs import export_gifs
from mechanics_eval.rollouts import reference_pair, ModuleAdapter
from mechanics_eval.scenarios import Engine, catalog


class OneFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Engine()

    def test_all_fixtures_one_frame_and_no_warmup(self):
        for spec in catalog():
            with self.subTest(spec=spec.name):
                branches = reference_pair(self.engine, spec, 0, 1, 16)
                self.assertEqual(branches[0].context.shape, (1,63,63,3))
                np.testing.assert_array_equal(branches[0].context_actions,[0])
                initial = self.engine.construct(spec,0,1)
                from mechanics_eval.scenarios import u8
                np.testing.assert_array_equal(branches[0].context[0],u8(self.engine.render(initial)))
                np.testing.assert_array_equal(branches[0].context,branches[1].context)
                if spec.name == 'death':
                    self.assertTrue(branches[0].dones[-1])
                    self.assertEqual(len(branches[0].frames),1)

    def test_supported_modules_one_frame_and_repeat(self):
        import jax.numpy as jnp
        import train
        context = reference_pair(self.engine,catalog()[0],0,1,2)[0].context[None]
        checkpoint=Path(os.environ.get('MECHANICS_EVAL_CHECKPOINT','mechanics_eval/runs/model.npz'))
        if not checkpoint.is_file():self.skipTest('set MECHANICS_EVAL_CHECKPOINT to run real-model determinism checks')
        modules=['train']
        if importlib.util.find_spec('normuon_adapter'):modules.append('normuon_adapter')
        for module in modules:
            adapter = ModuleAdapter(module,checkpoint,True)
            incoming = np.zeros((1,1),np.int32)
            actions = np.zeros((1,2),np.int32)
            first = adapter.rollout(context,incoming,actions,123,False)
            again = adapter.rollout(context,incoming,actions,123,False)
            np.testing.assert_array_equal(first['frames'],again['frames'])
        # Exercise train.start's actual temporal encoding with a tiny capturing encoder.
        from flax import nnx
        class TemporalEncoder(nnx.Module):
            temporal = True
            def __call__(self,x):
                return x[:,0:1,0:1,:]
        pixels = np.arange(63*63*3,dtype=np.uint8).reshape(1,1,63,63,3)
        z = train._encode(TemporalEncoder(),jnp.array(0.),jnp.array(1.),jnp.asarray(pixels))
        np.testing.assert_array_equal(np.asarray(z)[...,:3],np.asarray(z)[...,3:])

    def test_terminal_gif_export_without_zip(self):
        spec = next(s for s in catalog() if s.name == 'death')
        refs = reference_pair(self.engine,spec,0,1,4)
        preds = [{'frames':np.repeat(r.frames[-1:],4,axis=0),'rewards':None,'dones':None} for r in refs]
        with TemporaryDirectory() as temp:
            root=Path(temp); evidence=root/'evidence';evidence.mkdir()
            name='death_s0_m0'
            save_case(evidence/name,spec,refs,preds,0,123,0,'combat')
            sampling=json.loads((evidence/name/'sampling.json').read_text())
            self.assertEqual(sampling['context_length'],1)
            self.assertEqual(sampling['context_incoming_actions'],[0])
            self.assertEqual(sampling['branches']['0']['step_keys_uint32'],sampling['branches']['1']['step_keys_uint32'])
            dump_json(evidence/'manifest.json',{'cases':[name]})
            dump_json(root/'results.json',{'samples':[{'evidence':str(evidence/name),
                'outcome':{'status':'unknown'}, 'branch_outcomes':[{'status':'unknown','facts':[]},{'status':'pass','facts':[]}], 'counterfactual':{'status':'unknown'}}]})
            gallery=export_gifs([evidence],[root/'results.json'],root/'gallery')
            self.assertTrue(gallery.exists())
            with Image.open(root/'gallery/suite0'/name/'selected.gif') as gif:
                self.assertEqual(gif.n_frames,len(refs[0].frames)+1)
            self.assertFalse((root/'gallery/evidence.zip').exists())
            data=json.loads((root/'gallery/gallery.json').read_text())
            self.assertNotIn('evidence.zip',data['downloads'].values())
            self.assertTrue((root/'gallery/suite0'/name/'trajectory.npz').is_file())
            index=json.loads((root/'gallery/index.json').read_text())
            self.assertEqual(index[0]['unscored_generated_padding_omitted'],3)
            self.assertEqual(index[1]['branch'],'alternative')
            self.assertEqual(index[0]['status'],'unknown')
            self.assertEqual(index[1]['status'],'pass')
