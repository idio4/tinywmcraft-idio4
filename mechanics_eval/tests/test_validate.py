"""Regression tests for the public, one-command validation entry point.

No real checkpoint or inference is required by these orchestration tests.
"""
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from mechanics_eval import runner as validate


class ValidateTests(unittest.TestCase):
    def make_args(self, root, *options):
        checkpoint = root / "model.npz"
        checkpoint.write_bytes(b"test-checkpoint")
        return validate.arguments(["--model", str(checkpoint), *options])

    def test_exact_group_design_and_default_all_gifs(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            args = self.make_args(root)
            steps = validate.command_plan(args, root / "output")
            self.assertEqual([label for label, _ in steps], [
                "collect/existing", "collect/focused", "collect/controlled", "collect/table",
                "score/existing", "score/focused", "score/controlled", "score/table",
                "conditional", "gallery",
            ])
            counts = [("existing", "1"), ("focused", "10"), ("existing", "10"), ("existing", "10")]
            for (_, command), (suite, samples) in zip(steps[:4], counts):
                self.assertEqual(command[command.index("--suite") + 1], suite)
                self.assertEqual(command[command.index("--samples") + 1], samples)
                self.assertEqual(command[command.index("--context") + 1], "1")
                self.assertEqual(command[command.index("--horizon") + 1], "16")
                self.assertIn("--fp32", command)
            controlled = steps[2][1]
            self.assertEqual(controlled[controlled.index("--scenario") + 1:],
                             ["move_free", "move_blocked", "turn_loop", "noop", "table_leave_return"])
            conditional = steps[-2][1]
            self.assertEqual(conditional[conditional.index("--first-evidence") + 1],
                             str(root / "output" / "evidence" / "controlled"))
            self.assertEqual(conditional[conditional.index("--max-samples") + 1], "20")
            gallery = steps[-1][1]
            self.assertEqual(gallery[gallery.index("--gifs") + 1], "all")
            evidence = gallery[gallery.index("--evidence") + 1:gallery.index("--results")]
            self.assertEqual(len(evidence), 5)
            self.assertEqual(gallery.count(str(root / "output" / "conditional" / "evidence")), 1)

    def test_custom_filter_and_seeds(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            args = self.make_args(root, "--gifs", "fail", "--seed", "3", "--model-seed", "88",
                                  "--native-precision", "--batch-size", "2", "--max-samples", "7")
            plan = validate.command_plan(args, root / "new")
            for _, command in plan[:4]:
                self.assertNotIn("--fp32", command)
                self.assertEqual(command[command.index("--seed") + 1], "3")
                self.assertEqual(command[command.index("--model-seed") + 1], "88")
                self.assertEqual(command[command.index("--batch-size") + 1], "2")
            self.assertEqual(plan[-2][1][plan[-2][1].index("--max-samples") + 1], "7")
            self.assertEqual(plan[-1][1][plan[-1][1].index("--gifs") + 1], "fail")

    def test_run_keeps_intermediate_data_and_never_overwrites(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "output"
            args = self.make_args(root, "--out", str(output), "--gifs", "none")
            seen = []

            sessions = []

            def fake_stage(command, *, session):
                sessions.append(session)
                seen.append(command)
                if "export-gallery" in command:
                    report = output / "report"
                    report.mkdir()
                    (report / "gallery.json").write_text(json.dumps({
                        "cohorts": [{"id": "ordinary:0", "summary": {}}],
                        "conditional": {"selected_eligible": 1},
                    }))

            with patch("mechanics_eval.cli.main", side_effect=fake_stage):
                target = validate.run(args)
                self.assertEqual(len(seen), 10)
                self.assertTrue(all(session is sessions[0] for session in sessions))
                self.assertEqual(target, (output / "report" / "index.html").resolve())
                self.assertEqual(json.loads((output / "results.json").read_text())["status"], "complete")
                manifest=json.loads((output / "run.json").read_text())
                summary=json.loads((output / "results.json").read_text())
                self.assertEqual(manifest["status"], "complete")
                self.assertEqual(set(manifest['stage_elapsed_s']),set(manifest['completed_steps']))
                self.assertTrue(all(value>=0 for value in manifest['stage_elapsed_s'].values()))
                self.assertGreaterEqual(manifest['elapsed_s'],sum(manifest['stage_elapsed_s'].values())-.01)
                self.assertEqual(summary['stage_elapsed_s'],manifest['stage_elapsed_s'])
                self.assertEqual(summary['elapsed_s'],manifest['elapsed_s'])
                with self.assertRaisesRegex(ValueError, "already exists"):
                    validate.run(args)

    def test_failed_stage_preserves_manifest_and_never_exports(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "output"
            args = self.make_args(root, "--out", str(output))
            with patch("mechanics_eval.cli.main", side_effect=subprocess.CalledProcessError(5, "collect")) as p:
                with self.assertRaises(subprocess.CalledProcessError):
                    validate.run(args)
            self.assertEqual(p.call_count, 1)
            status = json.loads((output / "run.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["completed_steps"], [])
            self.assertIn('collect/existing',status['stage_elapsed_s'])
            self.assertGreaterEqual(status['elapsed_s'],0)
            self.assertFalse((output / "results.json").exists())

    def test_session_loads_model_lazily_once_and_reuses_engines(self):
        from mechanics_eval.rollouts import EvaluationSession
        with patch('mechanics_eval.rollouts.ModuleAdapter') as adapter, \
             patch('mechanics_eval.scenarios.Engine') as engine, \
             patch('mechanics_eval.focused.VariedEngine') as focused:
            session = EvaluationSession('train', Path('model.npz'), True)
            adapter.assert_not_called()
            self.assertIs(session.adapter(), session.adapter())
            adapter.assert_called_once_with('train', Path('model.npz'), True)
            self.assertIs(session.engine(), session.engine())
            self.assertIs(session.engine(True), session.engine(True))
            engine.assert_called_once_with()
            focused.assert_called_once_with()

    def test_gif_workers_only_change_the_export_command(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            default=self.make_args(root)
            custom=self.make_args(root,'--gif-workers','1')
            first=validate.command_plan(default,root/'out')
            second=validate.command_plan(custom,root/'out')
            self.assertEqual(first[:-1],second[:-1])
            command=second[-1][1]
            self.assertEqual(command[command.index('--gif-workers')+1],'1')


if __name__ == "__main__":
    unittest.main()
