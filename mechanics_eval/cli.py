"""Lightweight Level 1 milestone: collect evidence once, rescore and audit it offline."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time
import jax
import numpy as np
from mechanics_eval.evidence import digest, dump_json, save_case
from mechanics_eval.focused import VariedEngine, category, suite, variants
from mechanics_eval.rollouts import ModuleAdapter, reference_pair
from mechanics_eval.scenarios import Engine


def collect(args):
    if args.out.exists():
        raise ValueError('Use a new evidence directory; previous runs are never overwritten')
    if not args.model.is_file():
        raise ValueError('An existing local checkpoint is required')
    engine = Engine() if args.suite == 'existing' else VariedEngine()
    adapter = ModuleAdapter(args.module, args.model, args.fp32)
    specs = suite(args.suite)
    if args.scenario:
        unknown = set(args.scenario) - {s.name for s in specs}
        if unknown:
            raise ValueError(f'unknown scenarios: {sorted(unknown)}')
        specs = [s for s in specs if s.name in args.scenario]
    args.out.mkdir(parents=True)
    report = {'schema_version': 2, 'kind': 'learned_model_evidence', 'status': 'in_progress', 'cases': [],
        'errors': [], 'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'provenance': {'backend': jax.default_backend(), 'checkpoint_sha256': digest(args.model),
        'module_sha256': digest(adapter.module.__file__),
        'sampling_settings': {key: float(getattr(getattr(adapter.module, 'legacy', adapter.module), key))
                              for key in ('STEPS', 'CTX_SIGNAL')
                              if hasattr(getattr(adapter.module, 'legacy', adapter.module), key)},
        'model_window': int(adapter.model.win),
        'git_head': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        'packages': {name: importlib.metadata.version(name) for name in ('craftax','jax','flax','numpy')},
        'sources': {str(p): digest(p) for p in [Path(__file__), *sorted(Path(__file__).resolve().parent.glob('*.py'))]}}}
    dump_json(args.out / 'manifest.json', report)
    start = time.perf_counter()
    jobs = [(variants(s, i) if args.suite != 'existing' else s, args.seed + i)
            for s in specs for i in range(args.samples)]
    for offset in range(0, len(jobs), args.batch_size):
        batch = jobs[offset:offset + args.batch_size]
        refs = [reference_pair(engine, spec, seed, args.context, args.horizon) for spec, seed in batch]
        for repeat in range(args.model_samples):
            model_seed = int(np.random.SeedSequence([args.seed if args.model_seed is None else args.model_seed, offset, repeat, 933]).generate_state(1)[0])
            predictions = []
            for branch in range(2):
                branch_refs = [pair[branch] for pair in refs]
                future = np.asarray([list(ref.actions) + [0] * (args.horizon - len(ref.actions)) for ref in branch_refs])
                predictions.append(adapter.rollout(np.stack([ref.context for ref in branch_refs]),
                    np.stack([ref.context_actions for ref in branch_refs]), future, model_seed, True))
            for i, ((spec, seed), pair) in enumerate(zip(batch, refs)):
                name = f'{spec.name}_s{seed}_m{repeat}'
                single = [{k: v[i] if v is not None else None for k, v in pred.items()} for pred in predictions]
                save_case(args.out / name, spec, pair, single, seed, model_seed, repeat,
                          category(spec.name) if args.suite == 'focused' else spec.category)
                meta_path = args.out / name / 'metadata.json'
                meta = json.loads(meta_path.read_text())
                meta['replay'] = {'context_length': args.context, 'engine_seed': seed,
                    'model_seed_root': args.seed if args.model_seed is None else args.model_seed,
                    'module': args.module, 'checkpoint_sha256': report['provenance']['checkpoint_sha256'],
                    'batch_offset': offset, 'batch_index': i, 'batch_size_actual': len(batch),
                    'batch_size_configured': args.batch_size, 'fp32': args.fp32,
                    'sampling_settings': report['provenance']['sampling_settings'],
                    'sampling': 'module.step defaults; fold_in(PRNGKey(model_seed), timestep); identical branch keys'}
                dump_json(meta_path, meta)
                report['cases'].append(name)
                print(f'[saved] {name}', flush=True)
            dump_json(args.out / 'manifest.json', report)
    report.update(status='complete', elapsed_s=round(time.perf_counter() - start, 3))
    dump_json(args.out / 'manifest.json', report)
    print(json.dumps({'status': report['status'], 'cases': len(report['cases']), 'out': str(args.out),
                      'elapsed_s': report['elapsed_s']}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    c = commands.add_parser('collect')
    c.add_argument('--model', type=Path, required=True)
    c.add_argument('--module', default='train')
    c.add_argument('--fp32', action='store_true')
    c.add_argument('--suite', choices=('focused','existing'), default='focused')
    c.add_argument('--scenario', nargs='+')
    c.add_argument('--samples', type=int, default=10)
    c.add_argument('--model-samples', type=int, default=1)
    c.add_argument('--seed', type=int, default=0)
    c.add_argument('--model-seed', type=int, help='Independent sampling seed root; defaults to --seed')
    c.add_argument('--horizon', type=int, default=16)
    c.add_argument('--context', type=int, default=1)
    c.add_argument('--batch-size', type=int, default=4)
    c.add_argument('--out', type=Path, required=True)
    review = commands.add_parser('export-review')
    review.add_argument('--evidence',type=Path,required=True)
    review.add_argument('--out',type=Path,required=True)
    audit = commands.add_parser('audit-reader')
    audit.add_argument('--labels',type=Path,required=True)
    audit.add_argument('--out',type=Path,required=True)
    s = commands.add_parser('score')
    s.add_argument('--evidence', type=Path, required=True)
    s.add_argument('--out', type=Path, required=True)
    s.add_argument('--measurements',type=Path,help='Replay saved independent RGB readings from a matching score/cache directory')
    s.add_argument('--conditional-fixed-route',action='store_true',help='Require the complete six-tile route for the conditional experiment')
    conditional = commands.add_parser('conditional-route')
    conditional.add_argument('--first-evidence',type=Path,required=True)
    conditional.add_argument('--first-measurements',type=Path)
    conditional.add_argument('--model',type=Path,required=True)
    conditional.add_argument('--module',default='train')
    conditional.add_argument('--fp32',action='store_true')
    conditional.add_argument('--max-samples',type=int,default=20,help='Total model samples per environment seed, including the first')
    conditional.add_argument('--seed-step',type=int,default=1,help='Odd uint32 increment of the first recorded model seed')
    conditional.add_argument('--out',type=Path,required=True)
    missing=commands.add_parser('audit-missing')
    missing.add_argument('--results',type=Path,nargs='+',required=True)
    missing.add_argument('--out',type=Path,required=True)
    human=commands.add_parser('export-human-review')
    human.add_argument('--results',type=Path,nargs='+',help='Current result files; defaults to the latest Level 1 suites and conditional cohorts')
    human.add_argument('--runs-root',type=Path,default=Path('mechanics_eval/runs'))
    human.add_argument('--fail-limit',type=int,default=4,help='Other failed questions sampled per category; mandatory failures are always included')
    human.add_argument('--pass-limit',type=int,default=2,help='Positive questions sampled per category')
    human.add_argument('--smoke-limit',type=int,help='Explicit partial export for UI smoke testing only')
    human.add_argument('--out',type=Path,required=True)
    summary=commands.add_parser('summarize-human-review')
    summary.add_argument('--manifest',type=Path,required=True)
    summary.add_argument('--labels',type=Path,required=True)
    summary.add_argument('--out',type=Path,required=True)
    g = commands.add_parser('export-gallery', aliases=['export-gifs'], help='Offline interactive results; optional GIF media')
    g.add_argument('--evidence', type=Path, nargs='+', required=True)
    g.add_argument('--results', type=Path, nargs='+', required=True)
    g.add_argument('--out', type=Path, required=True)
    g.add_argument('--gifs', choices=('all','fail','unknown','nonpass','none'), default='all', help='Media only; every card, score and denominator is always exported')
    args = p.parse_args()
    if args.command=='export-human-review':
        from mechanics_eval.human_review import export_review
        report=export_review(args.out,args.results,args.runs_root,args.fail_limit,args.pass_limit,args.smoke_limit)
        print(json.dumps({'index':str((args.out/'index.html').resolve()),'groups':report['groups'],
                          'unique_items':report['total_unique_items'],'unique_questions':report['unique_questions']},indent=2))
    elif args.command=='summarize-human-review':
        from mechanics_eval.human_review import summarize_review
        report=summarize_review(args.manifest,args.labels,args.out)
        print(json.dumps({k:report[k] for k in ('total_unique_items','fully_reviewed_items','reviewed_questions','unreviewed_questions')},indent=2))
    elif args.command=='audit-missing':
        from mechanics_eval.unknown_audit import audit
        summary=audit(args.results,args.out)
        print(json.dumps({k:v for k,v in summary.items() if k!='tracked_table_none'},indent=2))
    elif args.command=='conditional-route':
        from mechanics_eval.conditional import run
        print(run(args))
    elif args.command == 'collect':
        if min(args.samples, args.model_samples, args.batch_size) < 1 or not 1 <= args.horizon <= 128 or not 1 <= args.context <= 24 or args.seed < 0 or (args.model_seed is not None and args.model_seed < 0):
            p.error('invalid counts, horizon, context or seed')
        collect(args)
    elif args.command in ('export-gallery','export-gifs'):
        from mechanics_eval.gifs import export_gifs
        print(export_gifs(args.evidence, args.results, args.out, args.gifs))
    elif args.command == 'export-review':
        from mechanics_eval.audit import export_review
        print(export_review(args.evidence,args.out))
    elif args.command == 'audit-reader':
        from mechanics_eval.audit import validate_labels
        from mechanics_eval.reader_v2 import ReaderV2
        if args.out.exists():
            p.error('Use a new audit output file')
        engine=Engine()
        reader=ReaderV2(engine,engine.construct(suite('existing')[0],100000,8))
        report=validate_labels(reader,args.labels,args.out)
        print(json.dumps({k:{field:v[field] for field in ('accepted','correct','accepted_accuracy','coverage_readable')}
                          for k,v in report['groups'].items()},indent=2))
    else:
        from mechanics_eval.milestone import score_run
        report=score_run(args.evidence, args.out,args.measurements,args.conditional_fixed_route)
        if report['status'].startswith('failed'):
            raise SystemExit(1)


if __name__ == '__main__':
    main()
