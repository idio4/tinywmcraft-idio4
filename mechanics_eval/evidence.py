"""Lossless, replayable evaluator evidence. Hidden labels are reference-only metadata."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
import jax
from .visual import HUD, hud_patch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump_json(path, value, *, compact=False):
    Path(path).write_text(json.dumps(value, indent=None if compact else 2,
                                    separators=(',', ':') if compact else None,
                                    allow_nan=False) + '\n')


def save_case(path, spec, refs, predictions, seed, model_seed, repeat, group):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    arrays = {'context': refs[0].context, 'context_actions': refs[0].context_actions}
    truth = {}
    for branch, (ref, prediction) in enumerate(zip(refs, predictions)):
        prefix = f'b{branch}_'
        arrays.update({prefix + key: value for key, value in {
            'reference': ref.frames, 'actions': ref.actions, 'reward_reference': ref.rewards,
            'done_reference': ref.dones, **{'generated': prediction['frames']},
        }.items()})
        for key in ('rewards', 'dones'):
            if prediction[key] is not None:
                arrays[prefix + key] = prediction[key]
        truth[str(branch)] = {'facts': ref.facts, 'initial': ref.initial,
            'tiles': [{f'{row},{col}': int(state.map[tuple(np.asarray(state.player_position) +
                       np.array([row - 3, col - 4]))]) for row in range(7) for col in range(9)
                       if (row, col) != (3, 4)} for state in ref.states]}
    np.savez_compressed(path / 'trajectory.npz', **arrays)
    # Explicit lossless crops: same coordinates and labels as the reader, independently inspectable.
    crops = {}
    for name, frames in arrays.items():
        if name in ('context', 'b0_reference', 'b1_reference', 'b0_generated', 'b1_generated'):
            crops[name + '_hud'] = np.stack([[hud_patch(frame, field) for field in HUD] for frame in frames])
            crops[name + '_tiles'] = frames[:, :49].reshape(len(frames), 7, 7, 9, 7, 3).transpose(0, 1, 3, 2, 4, 5)
    np.savez_compressed(path / 'crops.npz', **crops)
    metadata = {'schema_version': 2, 'spec': asdict(spec), 'seed': seed, 'model_seed': model_seed,
                'model_sample': repeat, 'category': group, 'reference_truth': truth,
                'hud_order': list(HUD), 'trajectory_sha256': digest(path / 'trajectory.npz'),
                'crops_sha256': digest(path / 'crops.npz'),
                'reader_boundary': 'Only individual RGB images enter the visual reader; these labels do not.'}
    dump_json(path / 'metadata.json', metadata)
    dump_json(path / 'sampling.json', sampling_record(metadata, arrays))
    return path.name


def load_case(path):
    path = Path(path)
    meta = json.loads((path / 'metadata.json').read_text())
    if digest(path / 'trajectory.npz') != meta['trajectory_sha256']:
        raise ValueError(f'changed evidence: {path}')
    with np.load(path / 'trajectory.npz', allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    return meta, arrays


def sampling_record(meta, arrays):
    """Explicit action/key schedule, including generated but unscored post-terminal padding."""
    branches = {}
    for branch in range(2):
        n = len(arrays[f'b{branch}_generated'])
        actions = arrays[f'b{branch}_actions'].tolist()
        branches[str(branch)] = {'future_actions': actions + [0] * (n-len(actions)),
            'reference_valid_mask': [t < len(arrays[f'b{branch}_reference']) for t in range(n)],
            'engine_step_keys_uint32': [np.asarray(jax.random.fold_in(jax.random.PRNGKey(meta['seed']),len(arrays['context'])+t)).tolist() for t in range(len(actions))],
            'step_keys_uint32': [np.asarray(jax.random.fold_in(jax.random.PRNGKey(meta['model_seed']),t)).tolist()
                                 for t in range(n)]}
    return {'model_seed': meta['model_seed'], 'engine_seed': meta['seed'],
            'case_name': meta['spec']['name'], 'context_length': len(arrays['context']),
            'context_incoming_actions': arrays['context_actions'].tolist(), 'branches': branches,
            'head_key_schedule':'fold_in(step_key, 991)',
            'batch_replay':'Use the root manifest batch size and original case order; changing batch shape changes draws.'}
