"""Parallel media and compact JSON must preserve report content and random keys."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from mechanics_eval.evidence import digest, dump_json
from mechanics_eval.gifs import export_gifs
from mechanics_eval.tests.test_reporting_integration import fixture


class ExportOptimizationTests(unittest.TestCase):
    def test_compact_json_is_lossless_and_rejects_nonfinite_values(self):
        value={'frames':[{'nested':{'values':[0,1,2.5,None,True,'wood']}} for _ in range(20)]}
        with TemporaryDirectory() as temp:
            root=Path(temp)
            dump_json(root/'pretty.json',value)
            dump_json(root/'compact.json',value,compact=True)
            self.assertEqual(json.loads((root/'compact.json').read_text()),value)
            self.assertLess((root/'compact.json').stat().st_size,(root/'pretty.json').stat().st_size)
            for compact in (False,True):
                with self.assertRaises(ValueError):dump_json(root/'invalid.json',{'bad':float('nan')},compact=compact)

    def test_parallel_gifs_and_report_match_serial_bytes(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);evidence,result=fixture(root,True)
            original=digest(result)
            for workers in (1,2):export_gifs([evidence],[result],root/str(workers),'all',workers)
            for filename in ('gallery.json','index.json','mechanics.csv','index.html'):
                self.assertEqual((root/'1'/filename).read_bytes(),(root/'2'/filename).read_bytes(),filename)
            serial={p.relative_to(root/'1'):p.read_bytes() for p in (root/'1').rglob('*.gif')}
            parallel={p.relative_to(root/'2'):p.read_bytes() for p in (root/'2').rglob('*.gif')}
            self.assertTrue(serial)
            self.assertEqual(parallel,serial)
            self.assertEqual(digest(result),original)
            self.assertEqual(json.loads((root/'2/suite0/results.json').read_text()),json.loads(result.read_text()))
            timing=json.loads((root/'2/export.json').read_text())
            self.assertEqual(timing['gifs_written'],len(serial))
            self.assertEqual(timing['gif_workers'],min(2,os.cpu_count() or 1))
            self.assertGreater(timing['elapsed_s'],0)

    def test_worker_failure_prevents_a_completed_report(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);evidence,result=fixture(root)
            name=json.loads((evidence/'manifest.json').read_text())['cases'][0]
            path=evidence/name/'trajectory.npz'
            with np.load(path) as archive:arrays={key:archive[key] for key in archive.files}
            arrays['b0_generated']=arrays['b0_generated'][:1]
            np.savez_compressed(path,**arrays)
            meta_path=evidence/name/'metadata.json';meta=json.loads(meta_path.read_text())
            meta['trajectory_sha256']=digest(path);dump_json(meta_path,meta)
            with self.assertRaisesRegex(ValueError,'generated rollout shorter'):
                export_gifs([evidence],[result],root/'broken','all',2)
            self.assertFalse((root/'broken/index.html').exists())
            self.assertFalse((root/'broken/export.json').exists())

    def test_no_gifs_never_starts_a_process_pool(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);evidence,result=fixture(root)
            with patch('mechanics_eval.gifs.ProcessPoolExecutor',side_effect=AssertionError('no pool needed')):
                export_gifs([evidence],[result],root/'report','none',4)
            self.assertEqual(json.loads((root/'report/export.json').read_text())['gif_workers'],0)

    def test_spawn_imports_do_not_load_jax_or_model_code(self):
        check=subprocess.run([sys.executable,'-c',
            "import mechanics_eval.cli, mechanics_eval.gif_render, sys; assert 'jax' not in sys.modules; assert 'train' not in sys.modules"],
            capture_output=True,text=True)
        self.assertEqual(check.returncode,0,check.stdout+check.stderr)

    def test_collection_profiles_phases_without_changing_seed_assignment(self):
        from mechanics_eval.cli import collect
        with TemporaryDirectory() as temp:
            root=Path(temp);model=root/'model.npz';model.write_bytes(b'fake')
            module=root/'adapter.py';module.write_text('# mock model\n')
            args=argparse.Namespace(command='collect',model=model,module='train',fp32=True,suite='existing',
                scenario=['move_free'],samples=1,model_samples=1,seed=3,model_seed=17,context=1,horizon=2,batch_size=4,out=root/'evidence')
            ref=SimpleNamespace(context=np.zeros((1,63,63,3),np.uint8),context_actions=np.array([0],np.int32),actions=np.array([2,0]))
            def rollout(context,incoming,actions,seed,heads):
                return dict(frames=np.zeros((len(context),2,63,63,3),np.uint8),rewards=None,dones=None)
            adapter=SimpleNamespace(module=SimpleNamespace(__file__=str(module)),model=SimpleNamespace(win=16),rollout=rollout)
            session=SimpleNamespace(engine=lambda focused:object(),adapter=lambda:adapter)
            def save(path,*unused):
                path.mkdir();dump_json(path/'metadata.json',{})
            with patch('mechanics_eval.rollouts.reference_pair',return_value=[ref,ref]), \
                 patch('mechanics_eval.evidence.save_case',side_effect=save):
                collect(args,session=session)
            manifest=json.loads((args.out/'manifest.json').read_text())
            self.assertEqual(set(manifest['timings_s']),{'setup_s','reference_s','model_rollout_s','evidence_write_s'})
            self.assertTrue(all(t>=0 for t in manifest['timings_s'].values()))
            replay=json.loads((args.out/'move_free_s3_m0/metadata.json').read_text())['replay']
            self.assertEqual((replay['model_seed_root'],replay['batch_offset'],replay['batch_index']),(17,0,0))
