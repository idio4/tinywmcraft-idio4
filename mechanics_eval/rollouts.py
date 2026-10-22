"""Reference trajectories and the observation-only module adapter."""
from dataclasses import dataclass
import importlib

import jax
import numpy as np

from .scenarios import facts, u8


class EvaluationSession:
    """Reuse model weights and compiled engines across one evaluation run."""
    def __init__(self, module, checkpoint, fp32):
        self.module, self.checkpoint, self.fp32 = module, checkpoint, fp32
        self._adapter = None
        self._engines = {}
        self._collection_pool = None
        self._collection_workers = 0

    def adapter(self):
        if self._adapter is None:
            self._adapter = ModuleAdapter(self.module, self.checkpoint, self.fp32)
        return self._adapter

    def engine(self, focused=False):
        if focused not in self._engines:
            from .focused import VariedEngine
            from .scenarios import Engine
            self._engines[focused] = VariedEngine() if focused else Engine()
        return self._engines[focused]

    def collection_pool(self, workers):
        if self._collection_pool is None or self._collection_workers < workers:
            self.close_collection_pool()
            from concurrent.futures import ProcessPoolExecutor
            import multiprocessing
            from .collection import initialize_worker
            self._collection_pool = ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                initializer=initialize_worker, initargs=(self.module, self.checkpoint, self.fp32))
            self._collection_workers = workers
        return self._collection_pool

    def close_collection_pool(self):
        if self._collection_pool is not None:
            self._collection_pool.shutdown(wait=True, cancel_futures=True)
            self._collection_pool = None
            self._collection_workers = 0


@dataclass
class Reference:
    context: np.ndarray
    context_actions: np.ndarray
    actions: np.ndarray
    frames: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    states: list
    facts: list
    initial: dict
    origin: np.ndarray


def reference_pair(engine, spec, seed, context, horizon):
    if not 1 <= context <= 24:
        raise ValueError('context must lie in [1,24]')
    state = engine.construct(spec, seed, context)
    frames, incoming = [u8(engine.render(state))], [0]
    for t in range(context - 1):
        obs, state, _, done, _ = engine.step(jax.random.fold_in(jax.random.PRNGKey(seed), t), state, 0)
        if bool(done):
            raise ValueError(f'{spec.name}: terminal conditioning history')
        frames.append(u8(obs))
        incoming.append(0)
    origin = np.asarray(state.player_position)
    initial = facts(state, origin)
    sequence = list(spec.actions[:horizon]) + [0] * max(0, horizon - len(spec.actions))
    # Both action alternatives start from exactly the same state, context and per-step engine keys.
    alternative = [0] * horizon
    if not any(sequence):
        alternative[0] = 5 if spec.zombie else 2
    branches = []
    for actions in (sequence, alternative):
        current = state
        future, rewards, dones, states, labels = [], [], [], [], []
        for t, action in enumerate(actions):
            key = jax.random.fold_in(jax.random.PRNGKey(seed), context + t)
            obs, current, reward, done, _ = engine.step(key, current, action)
            future.append(u8(obs)); rewards.append(float(reward)); dones.append(bool(done))
            states.append(current); labels.append(facts(current, origin, done))
            if bool(done):
                break  # Include the terminal frame; never reset or score post-death padding.
        branches.append(Reference(np.stack(frames), np.asarray(incoming, np.int32),
            np.asarray(actions[:len(future)], np.int32), np.stack(future), np.asarray(rewards),
            np.asarray(dones), states, labels, initial, origin))
    primary = branches[0]
    if len(primary.frames) >= spec.milestone:
        actual = primary.facts[spec.milestone - 1]
        for key, value in spec.expected.items():
            if actual[key] != value:
                raise ValueError(f'{spec.name}: invalid fixture at {spec.milestone}: {key}={actual[key]}, expected {value}')
        if spec.expected_tile is not None:
            current = primary.states[spec.milestone - 1]
            p = np.asarray(current.player_position) + np.array([0, 1])
            actual_tile = int(current.map[tuple(p)])
            if actual_tile != spec.expected_tile:
                raise ValueError(f'{spec.name}: tile={actual_tile}, expected {spec.expected_tile}')
    return branches


def validate_frame(frame, batch):
    frame = np.asarray(frame)
    if frame.shape != (batch, 63, 63, 3) or frame.dtype != np.uint8:
        raise ValueError(f'step must return uint8 {(batch, 63, 63, 3)}, got {frame.shape} {frame.dtype}')
    return frame


class ModuleAdapter:
    """Only RGB, incoming actions and random keys cross this boundary. No engine or labels."""
    def __init__(self, module, checkpoint, fp32=False):
        if module == 'train':
            # The original trainer stored its pytree as arr_N; current train uses enc.N/dec.N/df.N.
            with np.load(checkpoint, allow_pickle=False) as archive:
                if 'meta' in archive.files and 'arr_0' in archive.files and 'enc.0' not in archive.files:
                    module = 'train_tiny'
        self.module = importlib.import_module(module)
        if fp32:
            # Original NorMuon exports its precision controls through legacy.
            mod = getattr(self.module, 'legacy', self.module)
            if hasattr(mod, 'CDT'):
                import jax.numpy as jnp
                mod.CDT = jnp.float32
                mod.MP = dict(dtype=jnp.float32, param_dtype=jnp.float32)
        self.model = self.module.load_model(checkpoint)
        if module == 'train_tiny' and fp32:
            # Legacy layers hardcode BF16. Set compute dtype on this model only; keep weights and globals intact.
            import jax.numpy as jnp
            from flax import nnx
            for _, node in nnx.iter_graph(self.model):
                if isinstance(node, (nnx.Linear, nnx.Conv, nnx.ConvTranspose)):
                    node.dtype = jnp.float32
        for name in ('start', 'step'):
            if not callable(getattr(self.module, name, None)):
                raise ValueError(f'{module} does not implement {name}')

    def rollout(self, context, incoming, actions, seed, read_heads=True):
        """Batched, autoregressive; future actions are delivered to step one at a time."""
        context, incoming, actions = map(np.asarray, (context, incoming, actions))
        b = len(context)
        if context.ndim != 5 or context.shape[2:] != (63, 63, 3) or context.dtype != np.uint8:
            raise ValueError('context must be RGB uint8 (B,C,63,63,3)')
        if incoming.shape != context.shape[:2] or actions.ndim != 2 or len(actions) != b:
            raise ValueError('incoming actions must align with context frames; future actions must be (B,T)')
        if not np.isin(incoming, np.arange(17)).all() or not np.isin(actions, np.arange(17)).all():
            raise ValueError('Craftax actions must be integers in [0,16]')
        state = self.module.start(self.model, context.copy(), incoming.copy())
        frames, rewards, dones = [], [], []
        history = list(context.transpose(1, 0, 2, 3, 4))
        action_history = list(incoming.T)
        for t, action in enumerate(actions.T):
            key = jax.random.fold_in(jax.random.PRNGKey(seed), t)
            frame, state = self.module.step(self.model, state, action.copy(), key)
            frame = validate_frame(frame, b)
            frames.append(frame)
            history.append(frame); action_history.append(action)
            if read_heads and callable(getattr(self.module, 'heads', None)):
                # Fixed length 24 is the official heads interface. Left padding is past-only.
                past = history[-24:]
                acts = action_history[-24:]
                padding = 24 - len(past)
                video = np.stack([past[0]] * padding + past, axis=1)
                aligned = np.stack([np.zeros(b, np.int32)] * padding + acts, axis=1)
                r, d = self.module.heads(self.model, video, aligned, jax.random.fold_in(key, 991))
                r, d = np.asarray(r), np.asarray(d)
                if r.shape != (b, 24) or d.shape != (b, 24):
                    raise ValueError('heads must return reward and done arrays (B,24)')
                r, d = r[:, -1], d[:, -1]
                if not np.isfinite(r).all() or not np.isfinite(d).all() or np.any((d < 0) | (d > 1)):
                    raise ValueError('heads contain nonfinite reward or invalid done probability')
                rewards.append(r); dones.append(d)
        return {'frames': np.stack(frames, axis=1),
                'rewards': np.stack(rewards, axis=1) if rewards else None,
                'dones': np.stack(dones, axis=1) if dones else None}
