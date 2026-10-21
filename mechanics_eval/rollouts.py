"""Reference trajectories and the observation-only module adapter."""
from dataclasses import dataclass
import importlib

import jax
import numpy as np

from .scenarios import facts, u8


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
        self.module = importlib.import_module(module)
        if fp32:
            # Original NorMuon exports its precision controls through legacy.
            mod = getattr(self.module, 'legacy', self.module)
            if hasattr(mod, 'CDT'):
                import jax.numpy as jnp
                mod.CDT = jnp.float32
                mod.MP = dict(dtype=jnp.float32, param_dtype=jnp.float32)
        self.model = self.module.load_model(checkpoint)
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
