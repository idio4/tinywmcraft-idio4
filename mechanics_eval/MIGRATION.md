# Migration verification

Migrated from mechanics source commit `143fdb337981ac9c9af12e791ac392d56c4b0cc4` to the original TinyWMCraft fork baseline `a6273cc2f62b426215ee9ac25232bb5915a22c14`. All additions are under `mechanics_eval/`. All 71 files already tracked by the destination—including root README, `docs/`, `train.py`, `eval.py` and dependency configuration—remain byte-identical to that baseline. No changes to `main` or a PR are part of this migration.

## Included and excluded

- Current collectors, real-engine fixtures/variation, atomic/dependency scoring, RGB readers, motion/persistence, conditional sampling, missing-fact audits, human-review tools and GIF/HTML/JSON/ZIP exports.
- The existing 116 tests, with package import/entry-point paths and temporary-output handling updated. Optional checkpoint and archived-RGB paths are supplied explicitly through environment variables.
- One-command runner (`python -m mechanics_eval --model ...`), advanced stage CLI (`python -m mechanics_eval.cli ...`), current documentation and complete saved candidate example: 498 branch records, compact per-branch JSON and two GIFs. Its cards, cohorts, metric values and denominators match the source snapshot exactly.
- Excluded: obsolete entry scripts, historical development plans/logs, private raw trajectories/crops, checkpoints, alternative model implementations and unrelated source-repository files. Local smoke evidence is ignored under `mechanics_eval/runs/`.

Twenty evaluator modules are byte-identical to source. `pixel_audit.py` only has trailing blank-line cleanup. Package initialization, CLI/runner imports, provenance discovery paths and human-review defaults/instructions were adapted to the new location. No fixture, seed schedule, reader threshold, scoring rule, conditional-selection rule or inference implementation was changed. Original `train.load_model/start/step/heads` is used directly.

## Tests actually run

- Final full regression: **116 tests in 183.527 seconds, OK; 107 passed and 9 skipped**. The skips require the original archived raw corpus, which is no longer present in the source checkout. They cover saved generated entity/pose/crop replays and historical-corpus gallery exports. The original assertions remain available with `MECHANICS_EVAL_TEST_RUN`; the public compact snapshot does not pretend to include that RGB corpus.
- The checkpoint-dependent one-frame/determinism test ran against the original `train` module. No external `normuon_adapter` is available in the destination; it is optional and was not exercised or bundled.
- Offline saved-example DOM checks: **498 records, 241 filter/summary checks, 542 expanded atomic-stage views**. All local snapshot links resolve. These checks do not certify browser rendering/mobile layout.
- All five GIF modes were additionally exercised using the fresh checkpoint smoke evidence; each retained both branch records, stages and identical aggregate denominators. `all` exported two GIFs; this particular smoke's two passing branches produce no GIFs for `fail`, `unknown`, `nonpass` or `none`.
- Final diff inspected against destination `main`: additions only within `mechanics_eval/`, with no checkpoint, raw trajectory, local symlink or machine-specific private path staged.

An initial orchestration-test failure exposed macOS `/var` versus `/private/var` temporary-path resolution. The assertion now compares resolved paths; runner/scoring behavior was not changed.

## Real checkpoint smoke (not a full benchmark rerun)

Actually loaded checkpoint SHA256 `b4d7b82ba698f1b7aeaab7383c324523bf27bde1aba2fe817745fce38efa0aef` through the destination's unchanged `train` module. CPU, fp32, batch 1, engine seed 0, model seed root 314 (derived sampling seed 946645087), one conditioning frame, two generated steps, `gather_wood` selected and paired NOOP branches.

Both collections completed. Every saved array matched across repeats, including generated RGB; paired branches used identical model random keys. RGB scoring completed (selected scenario PASS), then HTML/GIF/ZIP export completed with two branch records and two GIFs. Generated RGB SHA256s:

- selected: `30d963fee5de78a9fe6e5639edb765b5ea29b8cd493d7378d57993fc81064122`
- NOOP: `57ba1317ff886380e3a349de0248f42fe61d2ebb83c01310b45e63519b007207`

The complete 227 ordinary attempts and subsequent real-model conditional retries were **not** regenerated. Full one-command orchestration and conditional retry logic were tested with controlled test doubles/engine fixtures. This smoke verifies the migrated checkpoint → inference → RGB scoring → report path only; it does not establish full-suite model accuracy.

## Remaining scientific limitations

Candidate, not finalized. Independent generated-image reader validation and two seed-6 camera steps remain pending. General termination may be RGB-unobservable; occlusion/corrupted textures/ambiguous localization can yield UNKNOWN or NOT_TESTED. Conditional and unconditional results remain separate. The historical snapshot requires the original checkpoint, raw evidence and configuration for exact reproduction; those private artifacts are not published. No training, Pages activation or repository visibility change was performed.
