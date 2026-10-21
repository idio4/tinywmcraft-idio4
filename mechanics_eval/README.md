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
    ├── evidence.zip
    └── ...                   # all case records and selected GIFs
```

Use `--out mechanics_eval/runs/my-run` for a chosen *new* output directory. GIF modes are `all`, `fail`, `unknown`, `nonpass`, and `none`; they change **only media**, not tests, verdicts, or denominators. Run `python -m mechanics_eval --help` for all options.

### Other models

- **Another checkpoint of the same TinyWMCraft architecture:** change only `--model`.
- **A model implementing TinyWMCraft's original `eval.py` interface:** select its importable module with `--module your_adapter`.
- **Different architecture:** provide a module exposing `load_model(path)`, `start(model, frames, actions)`, and `step(model, state, action, key)`, plus optional `heads`. The current collector also expects `model.win` for provenance. Observations must be Craftax-Classic 63×63 uint8 RGB, with its 17 action IDs.

For example:

```bash
python -m mechanics_eval --model checkpoints/other.npz --module other_adapter
```

No training or `eval.py` changes are necessary.

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

## Advanced tools and tests

The lower-level CLI supports separate collection, scoring, conditional sampling, and offline report exports:

```bash
python -m mechanics_eval.cli --help
python -m mechanics_eval.cli collect --help
python -m mechanics_eval.cli score --help
python -m mechanics_eval.cli export-gallery --help
```

To run the included regression tests from the upstream environment:

```bash
python -m unittest discover -s mechanics_eval/tests -v
```

Some tests can optionally replay an archived, full-size historical RGB corpus; set `MECHANICS_EVAL_TEST_RUN` to that local run when available. A model-dependent test can use `MECHANICS_EVAL_CHECKPOINT`. Tests requiring unavailable historical evidence explicitly skip. Generated artifacts belong in the ignored `mechanics_eval/runs/`, never in the tracked example directory.

**Scope and limitations.** Fixed, controlled cases cannot establish general long-horizon playability or RL competence. Some poses, occlusions, camera localization, and termination events are ambiguous in RGB and must not be turned into confident failures. Independent image-level verification of newer generated-image pose/identity readers and two borderline movement cases remains pending. Treat the shipped historical findings as a candidate evaluation, not fully independently validated ground truth.
