"""Optional output paths and chained scoring must not rerun or alter collection."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from mechanics_eval.cli import main


class CollectCommandTests(unittest.TestCase):
    def test_automatic_paths_are_unique_and_keep_focused_native_defaults(self):
        with patch('mechanics_eval.cli.collect') as collect:
            main(['collect','--model','model.npz'])
            first=collect.call_args.args[0]
            main(['collect','--model','model.npz'])
            second=collect.call_args.args[0]
        self.assertNotEqual(first.out,second.out)
        self.assertEqual(first.out.parent,Path(__file__).resolve().parents[1]/'runs')
        self.assertTrue(first.out.name.startswith('focused-'))
        self.assertEqual((first.suite,first.samples,first.batch_size,first.workers),('focused',10,8,4))
        self.assertFalse(first.fp32)
        self.assertFalse(first.score)

    def test_score_runs_after_collection_into_nested_scores_directory(self):
        order=[]
        with TemporaryDirectory() as temp:
            evidence=Path(temp)/'evidence'
            def collect(args,session=None):
                order.append('collect');evidence.mkdir();(evidence/'manifest.json').write_text('{}')
            def score(source,out,engine=None):
                self.assertTrue((source/'manifest.json').is_file())
                self.assertEqual(source,evidence);self.assertEqual(out,evidence/'scores')
                order.append('score')
                return {'status':'provisional_pending_generated_human_audit'}
            with patch('mechanics_eval.cli.collect',side_effect=collect) as collection, \
                 patch('mechanics_eval.milestone.score_run',side_effect=score):
                main(['collect','--model','model.npz','--score','--out',str(evidence)])
            self.assertEqual(collection.call_count,1)
            self.assertEqual(order,['collect','score'])

    def test_collection_failure_never_starts_scoring(self):
        with patch('mechanics_eval.cli.collect',side_effect=ValueError('collection failed')), \
             patch('mechanics_eval.milestone.score_run') as score:
            with self.assertRaisesRegex(ValueError,'collection failed'):
                main(['collect','--model','model.npz','--score'])
            score.assert_not_called()

    def test_failed_score_is_not_reported_as_success(self):
        with patch('mechanics_eval.cli.collect'), \
             patch('mechanics_eval.milestone.score_run',return_value={'status':'failed_reference_reader_audit'}):
            with self.assertRaises(SystemExit) as exc:
                main(['collect','--model','model.npz','--score'])
            self.assertEqual(exc.exception.code,1)
