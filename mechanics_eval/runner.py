"""Run the fixed Level 1 Craftax mechanics suite on a compatible checkpoint.

A run collects real/model paired trajectories, scores saved RGB, performs the
departure-only conditional return experiment, and exports a standalone HTML
viewer. All generated outputs stay inside one new run directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent
SUITES = (
    ("existing", "existing", 1, ()),
    ("focused", "focused", 10, ()),
    ("controlled", "existing", 10, ("move_free", "move_blocked", "turn_loop", "noop", "table_leave_return")),
    ("table", "existing", 10, ("place_table",)),
)
GIF_MODES = ("all", "fail", "unknown", "nonpass", "none")


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True, help="Compatible model checkpoint; never modified")
    p.add_argument("--module", default="train", help="Python model adapter implementing load_model/start/step/heads")
    p.add_argument("--out", type=Path, help="New output directory; default: timestamped subfolder of mechanics_eval/runs/")
    p.add_argument("--gifs", choices=GIF_MODES, default="all", help="Which verdicts receive GIFs; all scores are always exported")
    p.add_argument("--seed", type=int, default=0, help="First engine seed; focused/supplementary cohorts use ten consecutive seeds")
    p.add_argument("--model-seed", type=int, help="Optional model sampling seed root; the default follows --seed")
    p.add_argument("--batch-size", type=int, default=4, help="Ordinary sampling batch size (affects model random keys)")
    p.add_argument("--max-samples", type=int, default=20, help="Max conditional attempts per engine seed, including first")
    p.add_argument("--seed-step", type=int, default=1, help="Positive odd increment for conditional sampling keys")
    p.add_argument("--fp32", action="store_true", default=True, help="Use fp32 sampling (default; matches recorded Level 1)")
    p.add_argument("--native-precision", dest="fp32", action="store_false", help="Use adapter's default precision instead")
    args = p.parse_args(argv)
    if not args.model.is_file():
        p.error("checkpoint not found: " + str(args.model))
    if args.seed < 0 or (args.model_seed is not None and args.model_seed < 0):
        p.error("seeds must be nonnegative")
    if args.batch_size < 1 or args.max_samples < 1 or args.seed_step < 1 or args.seed_step % 2 != 1:
        p.error("batch-size/max-samples must be positive; seed-step must be positive and odd")
    if not all(part.isidentifier() for part in args.module.split(".")):
        p.error("--module must be a valid importable Python module name")
    return args


def new_output(args):
    if args.out is not None:
        return args.out.expanduser().resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return ROOT / "mechanics_eval" / "runs" / ("level1-" + stamp + "-" + uuid4().hex[:6])


def command_plan(args, out):
    """Return the ordered, testable execution plan. No model or scientific code runs here."""
    model = str(args.model.expanduser().resolve())
    cmd = [sys.executable, "-m", "mechanics_eval.cli"]
    fp = ["--fp32"] if args.fp32 else []
    commands = []
    for name, suite, samples, scenarios in SUITES:
        evidence = out / "evidence" / name
        step = cmd + ["collect", "--model", model, "--module", args.module, *fp,
                      "--suite", suite, "--samples", str(samples), "--seed", str(args.seed),
                      "--batch-size", str(args.batch_size), "--context", "1", "--horizon", "16",
                      "--out", str(evidence)]
        if args.model_seed is not None:
            step += ["--model-seed", str(args.model_seed)]
        if scenarios:
            step += ["--scenario", *scenarios]
        commands.append(("collect/" + name, step))
    for name, _, _, _ in SUITES:
        commands.append(("score/" + name, cmd + ["score",
                         "--evidence", str(out / "evidence" / name),
                         "--out", str(out / "scores" / name)]))
    commands.append(("conditional", cmd + ["conditional-route",
                    "--first-evidence", str(out / "evidence" / "controlled"),
                    "--first-measurements", str(out / "scores" / "controlled"),
                    "--model", model, "--module", args.module, *fp,
                    "--max-samples", str(args.max_samples), "--seed-step", str(args.seed_step),
                    "--out", str(out / "conditional")]))
    evidence = [out / "evidence" / name for name, _, _, _ in SUITES]
    scores = [out / "scores" / name / "results.json" for name, _, _, _ in SUITES]
    commands.append(("gallery", cmd + ["export-gallery",
                    "--evidence", *map(str, [*evidence, out / "conditional" / "evidence"]),
                    "--results", *map(str, [*scores, out / "conditional" / "gallery-results.json"]),
                    "--gifs", args.gifs, "--out", str(out / "report")]))
    return commands


def checksum(path):
    h = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def run(args):
    out = new_output(args)
    if out.exists():
        raise ValueError("Output already exists. Choose a new --out: " + str(out))
    # The exclusive mkdir guards against collisions and never removes old evidence.
    out.mkdir(parents=True, exist_ok=False)
    (out / "evidence").mkdir()
    (out / "scores").mkdir()
    config = {
        "model": str(args.model.expanduser().resolve()), "checkpoint_sha256": checksum(args.model),
        "module": args.module, "fp32": args.fp32, "seed": args.seed,
        "model_seed": args.model_seed, "batch_size": args.batch_size,
        "context": 1, "horizon": 16, "max_samples": args.max_samples,
        "seed_step": args.seed_step, "gifs": args.gifs,
    }
    manifest = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config, "completed_steps": [],
    }
    manifest_file = out / "run.json"
    save_json(manifest_file, manifest)
    try:
        for label, command in command_plan(args, out):
            print("[level1] " + label, flush=True)
            # Execute from the repository root, including when called from another cwd.
            subprocess.run(command, cwd=ROOT, check=True)
            manifest["completed_steps"].append(label)
            save_json(manifest_file, manifest)
        gallery = json.loads((out / "report" / "gallery.json").read_text(encoding="utf-8"))
        summary = {
            "status": "complete", "configuration": config,
            "cohorts": gallery["cohorts"], "conditional": gallery.get("conditional"),
            "report": "report/index.html", "gallery_data": "report/gallery.json",
            "note": "Cohorts and reused conditional attempts are separate; no pooled success score.",
        }
        save_json(out / "results.json", summary)
        manifest["status"] = "complete"
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(manifest_file, manifest)
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["error"] = str(exc)
        save_json(manifest_file, manifest)
        raise
    print("\n[level1] Complete. Open " + str(out / "report" / "index.html"), flush=True)
    return out / "report" / "index.html"


if __name__ == "__main__":
    try:
        run(arguments())
    except (ValueError, subprocess.CalledProcessError) as exc:
        sys.exit("Validation failed: " + str(exc))
