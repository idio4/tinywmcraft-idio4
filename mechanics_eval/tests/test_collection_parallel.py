"""Scheduling preserves global batch keys, evidence order and partial batches."""
import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


def load_model(path):
    return SimpleNamespace(win=32, fail=Path(path).name=='fail.npz')


def start(model, context, incoming):
    if model.fail and np.any(context[:,0,0,0,0]==5):
        raise ValueError('intentional collection failure')
    return context[:,-1]


def step(model, state, action, key):
    value=(state.astype(np.uint32)+np.asarray(key).sum(dtype=np.uint64)%251+action[:,None,None,None])%256
    return value.astype(np.uint8), state


def heads(model, video, actions, key):
    reward=video.mean(axis=(2,3,4))/255
    return reward, np.zeros_like(reward)


def fake_reference(engine, spec, seed, context, horizon):
    from mechanics_eval.rollouts import Reference
    frame=np.full((63,63,3),seed,np.uint8)
    actions=np.asarray((list(spec.actions)+[0]*horizon)[:horizon],np.int32)
    state=SimpleNamespace(map=np.zeros((32,32),np.int32),player_position=np.array([16,16]))
    ref=Reference(np.stack([frame]*context),np.zeros(context,np.int32),actions,
        np.stack([frame]*horizon),np.zeros(horizon),np.zeros(horizon,bool),
        [state]*horizon,[{}]*horizon,{},np.array([16,16]))
    return [ref,ref]


def initialize_fake_worker(module, checkpoint, fp32):
    from mechanics_eval import collection, rollouts
    collection.initialize_worker(module,checkpoint,fp32)
    collection._worker_session._engines[False]=object()
    rollouts.reference_pair=fake_reference


@unittest.skipUnless((os.cpu_count() or 1)>1,'parallel worker test requires two CPUs')
class CollectionParallelTests(unittest.TestCase):
    def args(self,root,out,workers,*,fail=False):
        model=root/('fail.npz' if fail else 'model.npz');model.write_bytes(b'fixture')
        return argparse.Namespace(command='collect',model=model,
            module='mechanics_eval.tests.test_collection_parallel',fp32=False,suite='existing',
            scenario=['move_free','turn_loop'],samples=5,model_samples=2,seed=3,model_seed=17,
            context=1,horizon=2,batch_size=4,workers=workers,out=root/out)

    def collect(self,args,session=None):
        from mechanics_eval.cli import collect
        from mechanics_eval.rollouts import EvaluationSession
        own=session is None
        session=session or EvaluationSession(args.module,args.model,args.fp32)
        session._engines[False]=object()
        try:
            with patch('mechanics_eval.collection.initialize_worker',initialize_fake_worker), \
                 patch('mechanics_eval.rollouts.reference_pair',fake_reference):
                collect(args,session=session)
        finally:
            if own:session.close_collection_pool()

    def test_parallel_preserves_order_keys_repeats_and_partial_batch(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            serial=self.args(root,'serial',1);parallel=self.args(root,'parallel',4)
            self.collect(serial);self.collect(parallel)
            a=json.loads((serial.out/'manifest.json').read_text())
            b=json.loads((parallel.out/'manifest.json').read_text())
            self.assertEqual(a['cases'],b['cases'])
            self.assertEqual(len(a['cases']),20)
            self.assertEqual(b['workers_used'],min(3,os.cpu_count() or 1))
            offsets=set();partial=[]
            for name in a['cases']:
                sa=serial.out/name;pb=parallel.out/name
                for filename in ('metadata.json','sampling.json'):
                    self.assertEqual(json.loads((sa/filename).read_text()),json.loads((pb/filename).read_text()))
                for filename in ('trajectory.npz','crops.npz'):
                    with np.load(sa/filename) as za,np.load(pb/filename) as zb:
                        self.assertEqual(za.files,zb.files)
                        for key in za.files:np.testing.assert_array_equal(za[key],zb[key])
                meta=json.loads((pb/'metadata.json').read_text());replay=meta['replay']
                offset=replay['batch_offset'];offsets.add(offset)
                expected=int(np.random.SeedSequence([17,offset,meta['model_sample'],933]).generate_state(1)[0])
                self.assertEqual(meta['model_seed'],expected)
                if offset==8:partial.append(replay['batch_size_actual'])
            self.assertEqual(offsets,{0,4,8})
            self.assertEqual(partial,[2]*4)

    def test_worker_failure_leaves_failed_evidence_and_closes_pool(self):
        from mechanics_eval.rollouts import EvaluationSession
        with TemporaryDirectory() as temp:
            args=self.args(Path(temp),'failed',4,fail=True)
            session=EvaluationSession(args.module,args.model,args.fp32)
            with self.assertRaisesRegex(ValueError,'intentional collection failure'):
                self.collect(args,session=session)
            manifest=json.loads((args.out/'manifest.json').read_text())
            self.assertEqual(manifest['status'],'failed')
            self.assertTrue(manifest['errors'])
            self.assertGreaterEqual(manifest['elapsed_s'],0)
            self.assertIsNone(session._collection_pool)

    def test_session_reuses_pool_across_collections_and_closes_explicitly(self):
        from mechanics_eval.rollouts import EvaluationSession
        with patch('concurrent.futures.ProcessPoolExecutor') as executor:
            session=EvaluationSession('train',Path('model.npz'),True)
            first=session.collection_pool(4)
            self.assertIs(first,session.collection_pool(2))
            self.assertEqual(executor.call_count,1)
            self.assertEqual(executor.call_args.kwargs['mp_context'].get_start_method(),'spawn')
            session.close_collection_pool()
            first.shutdown.assert_called_once_with(wait=True,cancel_futures=True)
            self.assertIsNone(session._collection_pool)
