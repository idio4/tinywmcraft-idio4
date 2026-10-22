"""Whole-batch collection tasks shared by serial and spawned execution."""
import json
import time

_worker_session = None


def initialize_worker(module, checkpoint, fp32):
    global _worker_session
    from .rollouts import EvaluationSession
    _worker_session = EvaluationSession(module, checkpoint, fp32)


def collect_batch(args, offset, batch, provenance, session=None):
    import numpy as np
    from .evidence import dump_json, save_case
    from .focused import category
    from .rollouts import reference_pair
    session = session if session is not None else _worker_session
    engine = session.engine(args.suite != 'existing')
    adapter = session.adapter()
    names = []
    timings = dict(reference_s=0., model_rollout_s=0., evidence_write_s=0.)
    phase_started = time.perf_counter()
    refs = [reference_pair(engine, spec, seed, args.context, args.horizon) for spec, seed in batch]
    timings['reference_s'] += time.perf_counter()-phase_started
    for repeat in range(args.model_samples):
        model_seed = int(np.random.SeedSequence([args.seed if args.model_seed is None else args.model_seed, offset, repeat, 933]).generate_state(1)[0])
        predictions = []
        phase_started = time.perf_counter()
        for branch in range(2):
            branch_refs = [pair[branch] for pair in refs]
            future = np.asarray([list(ref.actions) + [0] * (args.horizon - len(ref.actions)) for ref in branch_refs])
            predictions.append(adapter.rollout(np.stack([ref.context for ref in branch_refs]),
                np.stack([ref.context_actions for ref in branch_refs]), future, model_seed, True))
        timings['model_rollout_s'] += time.perf_counter()-phase_started
        phase_started = time.perf_counter()
        for i, ((spec, seed), pair) in enumerate(zip(batch, refs)):
            name = f'{spec.name}_s{seed}_m{repeat}'
            single = [{k: v[i] if v is not None else None for k, v in pred.items()} for pred in predictions]
            save_case(args.out / name, spec, pair, single, seed, model_seed, repeat,
                      category(spec.name) if args.suite == 'focused' else spec.category)
            meta_path = args.out / name / 'metadata.json'
            meta = json.loads(meta_path.read_text())
            meta['replay'] = {'context_length': args.context, 'engine_seed': seed,
                'model_seed_root': args.seed if args.model_seed is None else args.model_seed,
                'module': args.module, 'resolved_module': provenance['resolved_module'],
                'checkpoint_sha256': provenance['checkpoint_sha256'],
                'batch_offset': offset, 'batch_index': i, 'batch_size_actual': len(batch),
                'batch_size_configured': args.batch_size, 'fp32': args.fp32,
                'sampling_settings': provenance['sampling_settings'],
                'sampling': 'module.step defaults; fold_in(PRNGKey(model_seed), timestep); identical branch keys'}
            dump_json(meta_path, meta)
            names.append(name)
        timings['evidence_write_s'] += time.perf_counter()-phase_started
    return names, timings
