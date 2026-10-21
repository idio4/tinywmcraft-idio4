# TinyWMCraft Mechanics Validation

An **optional mechanics benchmark** for [TinyWMCraft](https://github.com/Maharishiva/tinywmcraft). It tests whether an action-conditioned world model respects Craftax-Classic rules such as crafting costs, resource gathering, movement, placement, and object persistence, instead of judging only video similarity.

The validator supplies controlled real-game RGB observations and action sequences, generates model frames through **the same inference API used by TinyWMCraft's original `eval.py`**, and independently scores visible consequences. It does not train a model or modify the original TinyWMCraft benchmark. The entire feature lives in `mechanics_eval/`; upstream `train.py`, `eval.py`, `docs/`, and the root README are unchanged.

## Run

**Use the existing TinyWMCraft Python environment.** From the repository root, with its environment activated:

```bash
python -m mechanics_eval --model runs/base/model.npz
```

Replace the path with the checkpoint you want to test. A checkpoint is required and is not bundled with this feature. **Do not create another virtual environment for this validator.** On a fresh checkout, first set up TinyWMCraft following its own root README, then use the same Python interpreter here.

The command runs all mechanics groups, saves both action branches, scores their generated RGB frames, performs the conditional table-return experiment, and creates a standalone interactive HTML report. By default it exports GIFs for **all** cases. To retain scores but generate fewer GIFs:

```bash
python -m mechanics_eval --model path/to/model.npz --gifs fail
```

Stages share one model and compiled engines in a single Python process. Collection progress groups seeds and updates after each batch at most every 30 seconds, plus completion; conditional progress summarizes each completed seed. Detailed sampling and retry decisions remain in the saved records. Batch sizes and random-key schedules are unchanged. Reports retain individual evidence files without generating a ZIP archive.

Results are written to a new unique, **gitignored** directory, never overwritten:

```text
mechanics_eval/runs/<run-id>/
├── run.json                  # configuration, checkpoint hash, completed stages
├── results.json              # summary
├── evidence/                 # paired generated/reference trajectory evidence
├── scores/                   # RGB-based verdicts
├── conditional/              # table-return retry evidence and selection
└── report/
    ├── index.html            # interactive report: open in a browser
    ├── gallery.json
    └── ...                   # all case records and selected GIFs
```

Use `--out mechanics_eval/runs/my-run` for a chosen *new* output directory. GIF modes are `all`, `fail`, `unknown`, `nonpass`, and `none`; they change **only media**, not tests, verdicts, or denominators. Run `python -m mechanics_eval --help` for all options.

### Seeds and repeated runs

Seeds are configurable: `--seed` defaults to `0` and sets the engine seed root; `--model-seed` defaults to the value of `--seed` and sets the model sampling seed root. Repeating the same checkpoint with the same settings uses the same random keys, aiming to reproduce the same results.

To get new model samples from the same checkpoint while keeping the engine scenarios identical, change only `--model-seed`:

```bash
python -m mechanics_eval --model model.npz --model-seed 42
python -m mechanics_eval --model model.npz --model-seed 43
```

Each command creates a new output directory and saves its seed configuration in `run.json`. For reproduction, keep the checkpoint, seeds, batch size, precision, model adapter, and software/backend environment the same; changing batch size also changes the model random-key assignment.

### Other models

- **Another checkpoint of the same TinyWMCraft architecture:** change only `--model`.
- **A model implementing TinyWMCraft's original `eval.py` interface:** select its importable module with `--module your_adapter`.
- **Different architecture:** provide a module exposing `load_model(path)`, `start(model, frames, actions)`, and `step(model, state, action, key)`, plus optional `heads`. The current collector also expects `model.win` for provenance. Observations must be Craftax-Classic 63×63 uint8 RGB, with its 17 action IDs.

For example:

```bash
python -m mechanics_eval --model checkpoints/other.npz --module other_adapter
```

No training or `eval.py` changes are necessary.

## Reading your model's results

The key output is **per-mechanic pass frequency**, not a pooled PASS/FAIL total. The report opens with a grouped table, for example **Stone pickaxe crafting: 9/10 (90%)**, showing the exact number of attempts, failures and UNKNOWN cases. This is an illustration, not a claim about the included checkpoint.

- **Scenario frequencies:** PASS / all attempts of that specific scenario within the currently selected cohort and scope. Decision coverage is (PASS + FAIL) / attempts. UNKNOWN remains in the denominator. Ten engine seeds measure frequency under controlled test variations, not probability of success in arbitrary gameplay.
- **Atomic mechanics:** Inspect required checks separately (tool creation, ingredient accounting, inventory persistence, etc.). Their rate is PASS / eligible checks, with eligible = PASS + FAIL + UNKNOWN. Applicable = all instances except NOT_APPLICABLE. Eligibility coverage is eligible / applicable, so prerequisite gaps (NOT_TESTED) remain visible without becoming successes or ordinary failures. No eligible checks means an undefined rate, displayed as `—` / `N/A`, not 0% or 100%.
- **Atomic context:** Atomic frequencies stay separate by scenario, scope, category, and required/diagnostic role. The same check name in two different scenarios produces two rows. Optional diagnostic failures do not alter the saved scenario verdict.
- **Sampling support:** Each scenario shows unique engine-seed count and the minimum–maximum number of model draws per seed. Repeated draws add attempts, not independent engine seeds. `model_sample_count` in machine-readable summaries counts distinct model-sample indices; it is not necessarily a uniform number of draws per seed, especially for conditional retries.
- **No pooling:** The focused, diagnostic, movement, placement, first-sample, conditional-selected and rejected-attempt cohorts remain separate. The report never adds their overlapping trials into one model-quality score.
- **Uncertainty:** 9/10 is stronger descriptive evidence than 1/1, but ten runs are still a small, structured sample. No arbitrary threshold declares a model globally "good."
- **Inspect evidence:** Click a scenario's mechanic button to filter its exact name, scope, and selected-action branch. This avoids matching similarly named scenarios or adding alternative branches to the displayed attempts. You can then choose other branches or use the free-text search; Reset gallery clears the exact-name filter without changing the summary denominator.

The same frequencies are available in `report/gallery.json`, `report/mechanics.csv`, root `results.json` under each cohort's `summary[scope].mechanics`, and in `scores/<group>/results.json` → `mechanic_performance`, `scores/<group>/mechanics.csv`, and `scores/<group>/summary.md`. JSON is also embedded in the standalone HTML. The gallery CSV identifies both the selected summary `scope` (`all`, `static`, or `dynamic`) and each row's `measurement_scope`; these are alternative views, not additional trials to sum. Scenario and atomic rows are distinguished by `kind`, with `scenario` retaining the atomic check's context.

Conditional `unconditional-results.json` and `conditional-results.json` recompute frequencies from their own first-sample and selected cohorts. Their accompanying `unconditional-mechanics.csv` / `.md` and `conditional-mechanics.csv` / `.md` use those same counts. A fully exhausted retry budget can yield zero selected attempts; its scenario list is empty rather than borrowing all-attempt frequencies. In `gallery-results.json`, `mechanic_performance` is explicitly labeled as the conditional-selected cohort; individual `samples` also retain inspection attempts and ordinary views for export. The report computes each of its first/selected/rejected/all-attempt views separately. Rejected/all-attempt frequencies are inspection-only and must not be interpreted as benchmark success estimates.

The legacy `samples.csv` columns `status`, `complete`, and fact counts describe the original primary atomic measurement. Use the added `scenario_status` and `scenario_complete` columns for whole-interaction verdicts, or `mechanics.csv` for per-scenario/per-atomic frequencies. New runs generate these outputs automatically. Existing saved RGB evidence can be rescored or re-exported to a fresh directory without new model inference; re-export does not rewrite old source score JSON.

## What is tested?

Ordinary trials use **one real conditioning frame**, **16 autoregressive generated steps**, one model draw per initial state, and selected vs NOOP/alternative actions with matched model keys. Engine seeds change controlled surroundings while preserving the relevant mechanic. The groups overlap; they are **not** disjoint collections of scenarios.

| Group | Test scope | Engine seeds | Attempts |
|---|---|---|---:|
| Diagnostic | All 27 scenarios, including combat/survival | 0 | 27 |
| Focused mechanics | 14 crafting, gathering and persistence scenarios | 0–9 | 140 |
| Controlled movement | 5 movement/return scenarios | 0–9 | 50 |
| Table placement | 1 table-placement scenario | 0–9 | 10 |
| Conditional return | Same place/leave/return route, retries only for objectively blocked departure | 0–9 | 10 retained samples, with at most 20 attempts per seed |

There are **227 ordinary attempts / 454 action-branch recordings**. Conditional sampling is a **separate** experiment. It retries only a confidently blocked newly revealed departure, never a missing table, failure to return, or ambiguous obstacle. See [PROTOCOL.md](PROTOCOL.md) for fixture design, atomic stage definitions, seed controls, prerequisite logic, and eligibility/coverage denominators.

Each scenario is **PASS**, **FAIL**, or **UNKNOWN** from visible evidence. Dependent stages can be **NOT_TESTED** when prerequisites fail, and inapplicable stages are excluded. UNKNOWN cases stay in attempted-case denominators. The model never receives future real-engine states or reference frames to infer generated outcomes.

## Two examples

The repository tracks only these two short illustrative GIFs, **not** the historical 498-case HTML snapshot, per-case JSON, checkpoint, or raw trajectories. Complete interactive reports are generated locally into `mechanics_eval/runs/` when you run validation.

| Wood pickaxe: PASS | Stone pickaxe: FAIL |
|---|---|
| ![Wood pickaxe crafting example](example/media/craft_wood_pickaxe_s0_m0.gif) | ![Stone pickaxe crafting failure](example/media/craft_stone_pickaxe_s0_m0.gif) |

These examples were produced with one specific checkpoint and do not predict results for other checkpoints. The omitted historical run reported 20/5/2 diagnostic PASS/FAIL/UNKNOWN and 101/39/0 focused PASS/FAIL/UNKNOWN; these are **historical results, not test expectations**. Its complete evidence remains outside Git. The GIFs include Craftax artwork; attribution is in [example/LICENSE.txt](example/LICENSE.txt).

## Project structure

The optional validator remains isolated from upstream TinyWMCraft. The repository's original `train.py`, `eval.py`, `pyproject.toml`, and `docs/` are unchanged.

```text
mechanics_eval/
├── __main__.py, runner.py, cli.py    # One-command and advanced entry points
├── scenarios.py, focused.py         # Controlled Craftax engine fixtures
├── rollouts.py, evidence.py         # Model interface and recorded trajectories
├── reader_v2.py, atomic.py, ...     # RGB readers, mechanics rules and scoring
├── gallery.py, gifs.py              # Full local HTML/GIF report generation
├── audit/                            # Optional reader, missing-fact and human audits
├── tests/                            # Regression and fixture controls
├── example/media/                   # Two curated GIFs, no large report snapshot
└── runs/                            # Gitignored generated evidence and reports
```

**Human review is optional, not outdated.** `audit/human.py` and its HTML template are for independent annotation of generated evidence and disagreement analysis. They do **not** run during regular validation or silently modify PASS/FAIL/UNKNOWN scores. Other optional reader, missing-observation, and pixel checks live in the same folder. See [audit/README.md](audit/README.md).

## Advanced tools and tests

Run these commands from the repository root in the existing TinyWMCraft environment. Paths below are examples: replace `mechanics_eval/runs/example-run` with an existing run for read-only inputs, and choose fresh output paths. Separate CLI invocations each start a new process; model and compiled-engine reuse applies to the full one-command runner.

### Full evaluation options

```bash
python -m mechanics_eval --help
python -m mechanics_eval --model model.npz --seed 0 --model-seed 42 \
  --gifs fail --out mechanics_eval/runs/model-seed-42
```

| Option | Default | Meaning |
|---|---|---|
| `--model PATH` | Required | Existing compatible checkpoint; never modified. |
| `--module NAME` | `train` | Importable model adapter. |
| `--out DIR` | Unique timestamped directory under `mechanics_eval/runs/` | New run directory; existing paths are rejected. |
| `--gifs MODE` | `all` | `all`, `fail`, `unknown`, `nonpass` (FAIL and UNKNOWN), or `none`; affects media only. |
| `--seed N` | `0` | Nonnegative engine seed root; diagnostic uses this seed, other ordinary groups use ten consecutive seeds. |
| `--model-seed N` | Value of `--seed` | Nonnegative model sampling seed root; change this alone for different draws on the same engine scenarios. |
| `--batch-size N` | `4` | Positive ordinary collection batch size. Changing it changes model random-key assignment and potentially samples. Conditional retries remain batch size 1. |
| `--max-samples N` | `20` | Conditional attempt budget per engine seed, including the reused first attempt; conditional sampling requires `1 <= N < 2**32`. |
| `--seed-step N` | `1` | Positive odd increment less than `2**32` between conditional model sampling seeds, wrapping modulo `2**32`. |
| `--fp32` | Enabled | Use FP32 sampling. |
| `--native-precision` | Disabled | Use the adapter's default precision instead of forcing FP32. |
| `-h`, `--help` | — | Show help without running evaluation. Available on all commands below too. |

The runner fixes context to 1 frame, horizon to 16 steps, and the cohort sizes described above. Custom scenarios, context lengths, horizons, and sample counts use `collect` instead and do not constitute the fixed Level 1 design.

### Command reference

```bash
python -m mechanics_eval.cli --help
python -m mechanics_eval.cli collect --help
```

Replace `collect` in the second command with any command in this table to inspect its help:

| Command | Purpose | Model inference? |
|---|---|---|
| `collect` | Generate paired reference/model trajectories. | Yes |
| `score` | Score saved RGB evidence or replay matching cached measurements. | No |
| `conditional-route` | Reuse first table-return samples and generate permitted retries. | Only when retries are needed |
| `export-gallery` | Export an interactive HTML report with optional GIFs. | No |
| `export-gifs` | Alias for `export-gallery`, with identical arguments. | No |
| `export-review` | Export a small reader-labeling sheet and blank labels. | No |
| `audit-reader` | Compare reader outputs against manually entered labels. | No |
| `audit-missing` | Diagnose missing or UNKNOWN observations. | No |
| `export-human-review` | Export an interactive human annotation workflow. | No |
| `summarize-human-review` | Summarize saved human annotations and disagreements. | No |

### Collect custom evidence

```bash
python -m mechanics_eval.cli collect --model model.npz --fp32 \
  --suite existing --scenario move_free move_blocked \
  --samples 10 --model-samples 1 --seed 0 --model-seed 42 \
  --context 1 --horizon 16 --batch-size 4 \
  --out mechanics_eval/runs/custom-evidence
```

| Option | Default | Meaning |
|---|---|---|
| `--model PATH` | Required | Existing checkpoint. |
| `--module NAME` | `train` | Model adapter module. |
| `--fp32` | Disabled | Force FP32; omit for adapter-native precision. Unlike the full runner, standalone collection does not force FP32 by default. |
| `--suite NAME` | `focused` | `focused` uses the focused scenario subset and varied surroundings; `existing` uses the complete diagnostic catalog. |
| `--scenario NAME [NAME ...]` | All scenarios in the suite | Filter by scenario name, such as `move_free`, `move_blocked`, `craft_wood_pickaxe`, or `table_leave_return`. Names must belong to the chosen suite; see `scenarios.py` and `focused.py`. |
| `--samples N` | `10` | Positive engine seeds per scenario, starting at `--seed`. |
| `--model-samples N` | `1` | Positive model draws per scenario/engine-seed pair. |
| `--seed N` | `0` | Nonnegative engine seed root. |
| `--model-seed N` | Value of `--seed` | Nonnegative model sampling seed root. |
| `--context N` | `1` | Number of real conditioning frames, from 1 through 24. |
| `--horizon N` | `16` | Generated rollout length, from 1 through 128; reference trajectories stop at terminal frames. |
| `--batch-size N` | `4` | Positive number of cases per model batch; affects model random keys. |
| `--out DIR` | Required | New evidence directory. |

Collection records both action branches, metadata, sampling/replay settings, and `trajectory.npz` in each case directory, plus a collection `manifest.json`. Frame-by-frame rollout and reward/done heads use the adapter interface described above.

### Score or rescore saved evidence

```bash
python -m mechanics_eval.cli score \
  --evidence mechanics_eval/runs/custom-evidence \
  --out mechanics_eval/runs/custom-scores
```

Required options are `--evidence DIR` (a complete collection) and `--out DIR` (a new score directory). Scoring writes `results.json`, `mechanics.csv`, `samples.csv`, `summary.md`, and per-case measurements and diagnostics without generating new model frames.

Optional `--measurements DIR` replays readings from a matching score directory or reading cache. Reader profiles and evidence-manifest hashes must match; this is not a cache for a different collection:

```bash
python -m mechanics_eval.cli score \
  --evidence mechanics_eval/runs/custom-evidence \
  --measurements mechanics_eval/runs/custom-scores \
  --out mechanics_eval/runs/custom-scores-replayed
```

Optional `--conditional-fixed-route` requires the complete six-tile departure/return route for conditional scoring. Ordinary scoring omits it.

### Conditional table-return sampling

```bash
python -m mechanics_eval.cli conditional-route \
  --first-evidence mechanics_eval/runs/example-run/evidence/controlled \
  --first-measurements mechanics_eval/runs/example-run/scores/controlled \
  --model model.npz --module train --fp32 \
  --max-samples 20 --seed-step 1 \
  --out mechanics_eval/runs/conditional-new
```

| Option | Default | Meaning |
|---|---|---|
| `--first-evidence DIR` | Required | Complete collection containing exactly one first `table_leave_return` sample per engine seed. |
| `--first-measurements DIR` | None | Matching first-sample score directory; omit to reread the saved RGB. |
| `--model PATH` | Required | Same checkpoint as the first samples, verified by hash. |
| `--module NAME` | `train` | Same adapter as the first samples; its source hash is checked when loading for retries. |
| `--fp32` | Disabled | Must match the first collection's precision setting. Use it for evidence from the default full runner. |
| `--max-samples N` | `20` | Total attempts per seed, including the reused first sample; from 1 through `2**32 - 1`. |
| `--seed-step N` | `1` | Positive odd increment less than `2**32` for retry sampling keys. |
| `--out DIR` | Required | New conditional experiment directory. |

The input must use one real conditioning frame, incoming NOOP, and the original fixed 16-step table-return sequence. Engine seeds and first model seeds come from the saved evidence; there are no separate `--seed` or `--model-seed` options here. Retry model seed `i` is `(first_model_seed + i * seed_step) mod 2**32`. Retries use batch size 1, retain paired branch keys, and stop at the first attempt that is not confidently rejected for a blocked newly revealed departure. A model failure or ambiguous observation does not trigger rejection sampling.

Outputs include `evidence/`, `readings/`, `sampling.json`, `all-attempt-scores/`, `summary.json`, `unconditional-results.json`, `conditional-results.json`, and `gallery-results.json`. The first-sample and conditional-selected cohorts each also get their own `*-mechanics.csv` and `*-mechanics.md` files. First samples and conditionally selected attempts retain separate denominators.

### Export a report from saved results

```bash
python -m mechanics_eval.cli export-gallery \
  --evidence mechanics_eval/runs/custom-evidence \
  --results mechanics_eval/runs/custom-scores/results.json \
  --gifs fail --out mechanics_eval/runs/custom-report
```

Required options are `--evidence DIR [DIR ...]`, `--results FILE [FILE ...]`, and `--out DIR`. Supply one result file per evidence directory, in matching order. Optional `--gifs` defaults to `all` and accepts `all`, `fail`, `unknown`, `nonpass`, or `none`. `export-gifs` is an exact alias.

To include every cohort from a completed full run in a fresh report:

```bash
python -m mechanics_eval.cli export-gallery \
  --evidence mechanics_eval/runs/example-run/evidence/existing \
    mechanics_eval/runs/example-run/evidence/focused \
    mechanics_eval/runs/example-run/evidence/controlled \
    mechanics_eval/runs/example-run/evidence/table \
    mechanics_eval/runs/example-run/conditional/evidence \
  --results mechanics_eval/runs/example-run/scores/existing/results.json \
    mechanics_eval/runs/example-run/scores/focused/results.json \
    mechanics_eval/runs/example-run/scores/controlled/results.json \
    mechanics_eval/runs/example-run/scores/table/results.json \
    mechanics_eval/runs/example-run/conditional/gallery-results.json \
  --gifs none --out mechanics_eval/runs/report-without-gifs
```

Export reads saved RGB and scores without inference or rescoring. Open the new `index.html` locally. The report retains `gallery.json` and individual evidence/result files, with no `evidence.zip`. Changing GIF mode leaves scores, case counts, and denominators intact.

### Reader review and missing-observation audits

Export a small stratified labeling sheet:

```bash
python -m mechanics_eval.cli export-review \
  --evidence mechanics_eval/runs/example-run/evidence/existing \
  --out mechanics_eval/runs/reader-review
```

Both options are required. The new output directory contains `review.png`, individual crops, and `labels.json`. Enter the visible values/types in the blank `label` and `item_label` fields, and record reviewer/verification information. Keep ambiguous observations unlabeled; use visible evidence rather than reference states to label generated images.

Compare these manual labels with the reader:

```bash
python -m mechanics_eval.cli audit-reader \
  --labels mechanics_eval/runs/reader-review/labels.json \
  --out mechanics_eval/runs/reader-review/audit.json
```

Both options are required. Here `--out` is a new JSON **file**, with an existing parent directory; a sibling `audit_confusion.csv` is also written. Results report accepted accuracy, readable coverage, and confusion matrices for digits, items, and tiles.

Diagnose missing/UNKNOWN observations across one or more score files:

```bash
python -m mechanics_eval.cli audit-missing \
  --results mechanics_eval/runs/example-run/scores/existing/results.json \
    mechanics_eval/runs/example-run/scores/focused/results.json \
  --out mechanics_eval/runs/missing-audit
```

Required options are `--results FILE [FILE ...]` and `--out DIR`. The new directory contains `records.json`, `summary.json`, `report.md`, and supporting crops. These audits do not replace benchmark scores.

### Interactive human review

```bash
python -m mechanics_eval.cli export-human-review \
  --results mechanics_eval/runs/example-run/scores/existing/results.json \
    mechanics_eval/runs/example-run/scores/focused/results.json \
    mechanics_eval/runs/example-run/conditional/gallery-results.json \
  --fail-limit 4 --pass-limit 2 \
  --out mechanics_eval/runs/human-review
```

| Option | Default | Meaning |
|---|---|---|
| `--results FILE [FILE ...]` | Fixed paths under `--runs-root` | Explicit score sources; overrides the default source list. |
| `--runs-root DIR` | `mechanics_eval/runs` | Root containing the expected `scores/` and `conditional/` directories when `--results` is omitted. Set this to a particular full run directory. |
| `--fail-limit N` | `4` | Additional failed questions sampled per category; mandatory failures are always included. |
| `--pass-limit N` | `2` | Positive questions sampled per category. |
| `--smoke-limit N` | None | Partial item export for UI smoke checks only; not a complete review. |
| `--out DIR` | Required | New review directory. |

To review all default sources from a completed full run, omit `--results` and use `--runs-root mechanics_eval/runs/example-run`. This reads the four ordinary score files plus conditional unconditional, selected, and all-attempt result files; it does not automatically search for the newest timestamped run.

Open the generated `index.html`, enter judgments, save them explicitly, and export the labels JSON. `review_manifest.json` identifies the review, `labels-template.json` starts empty, and the generated `README.md` explains the annotation workflow. Browser storage is only a convenience; exported labels are the saved review artifact. Viewing cases does not count as annotating them.

Summarize the exported labels:

```bash
python -m mechanics_eval.cli summarize-human-review \
  --manifest mechanics_eval/runs/human-review/review_manifest.json \
  --labels /absolute/path/to/human_labels.json \
  --out mechanics_eval/runs/human-review-summary
```

All three options are required. Labels must match the review manifest. The new output directory contains `report.json`, `report.md`, `report.html`, and detailed review tables. Missing judgments remain unreviewed; human review reports agreement/disagreement without changing automated scores. The `audit/pixels.py` diagnostic helper has no public CLI; see [audit/README.md](audit/README.md).

### Regression tests and optional inputs

Run the complete test suite:

```bash
python -m unittest discover -s mechanics_eval/tests -v
```

Add `-b` to buffer application prints from passing tests and show them only on failures. Import-time library messages may still appear. This does not change evaluation progress logging.

Run synthetic reporting and Node interaction regressions without historical evidence or model inference:

```bash
python -m unittest -v -b mechanics_eval.tests.test_performance \
  mechanics_eval.tests.test_reporting_integration
```

These cover denominator edge cases, required versus diagnostic checks, scope and cohort separation, model draws versus seeds, CSV/JSON/embedded HTML consistency, all GIF modes, links, and exact-name mechanic clicks. Node.js is needed for interaction checks; that test explicitly skips if Node is unavailable.

Run only orchestration tests with mocked stages and model loading:

```bash
python -m unittest mechanics_eval.tests.test_validate -v
```

Run a specific test by its dotted name:

```bash
python -m unittest \
  mechanics_eval.tests.test_validate.ValidateTests.test_run_keeps_intermediate_data_and_never_overwrites -v
```

The full suite includes engine fixtures and may compile JAX functions. Some tests replay an optional archived historical RGB corpus; those inputs are not bundled. To provide them:

```bash
MECHANICS_EVAL_TEST_RUN=/absolute/path/to/historical-run \
  python -m unittest discover -s mechanics_eval/tests -v
```

The default historical-input location is `mechanics_eval/runs/test-input`. Tests requiring unavailable evidence explicitly skip. A model-dependent determinism test uses `MECHANICS_EVAL_CHECKPOINT` (default `mechanics_eval/runs/model.npz`); setting it enables real inference:

```bash
MECHANICS_EVAL_CHECKPOINT=/absolute/path/to/model.npz \
  python -m unittest \
  mechanics_eval.tests.test_one_frame.OneFrameTests.test_supported_modules_one_frame_and_repeat -v
```

An optional Node.js check executes gallery interactions in a simulated DOM using only built-in Node modules:

```bash
node mechanics_eval/tests/check_gallery_dom.cjs \
  mechanics_eval/runs/example-run/report/index.html
```

This checks offline filter/interaction behavior, not real-browser rendering, mobile layout, or file-protocol behavior. Generated artifacts belong in the ignored `mechanics_eval/runs/`, never in the tracked example directory. Test output helpers use temporary directories.

**Scope and limitations.** Fixed, controlled cases cannot establish general long-horizon playability or RL competence. Some poses, occlusions, camera localization, and termination events are ambiguous in RGB and must not be turned into confident failures. Independent image-level verification of newer generated-image pose/identity readers and two borderline movement cases remains pending. Treat the shipped historical findings as a candidate evaluation, not fully independently validated ground truth.
