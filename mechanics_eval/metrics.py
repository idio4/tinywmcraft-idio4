"""Counts first: missing visual readings never become correct predictions or negative labels."""
from collections import Counter
import math

import numpy as np
from .visual import HUD


def mean(values):
    finite = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return float(np.mean(finite)) if finite else None


def binary(truth, prediction):
    if len(truth) != len(prediction):
        raise ValueError('unaligned binary predictions')
    if any(t is None or not np.isfinite(t) or t not in (0, 1) for t in truth):
        raise ValueError('binary labels must be finite booleans')
    if any(p is not None and (not np.isfinite(p) or p not in (0, 1)) for p in prediction):
        raise ValueError('binary predictions must be booleans or None')
    pairs = [(bool(t), bool(p)) for t, p in zip(truth, prediction) if p is not None]
    tp = sum(t and p for t, p in pairs)
    fp = sum(not t and p for t, p in pairs)
    fn = sum(t and not p for t, p in pairs)
    tn = sum(not t and not p for t, p in pairs)
    n = len(pairs)
    return {'n': n, 'total': len(truth), 'missing': len(truth) - n, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'positives_total': sum(bool(t) for t in truth),
            'missing_positives': sum(bool(t) and p is None for t, p in zip(truth, prediction)),
            'missing_negatives': sum(not bool(t) and p is None for t, p in zip(truth, prediction)),
            'accuracy': (tp + tn) / n if n else None,
            'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def estimate(values):
    vals = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    n = len(vals)
    return {'mean': mean(vals), 'n': n, 'total': len(values),
            'se': float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else None}


def observation_mse(pred, real):
    pred, real = np.asarray(pred), np.asarray(real)
    if pred.shape != real.shape or not len(real) or not np.isfinite(pred).all():
        raise ValueError('invalid/unaligned observation arrays')
    return float(np.mean(((pred.astype(np.float32) - real.astype(np.float32)) / 255.) ** 2))


def target_outcome(spec, reading):
    # An outcome is available only when ALL required facts can be read independently from RGB.
    if any(key not in HUD for key in spec.expected):
        return None
    if any(reading[key] is None for key in spec.expected):
        return None
    matches = [reading[key] == value for key, value in spec.expected.items()]
    if spec.expected_tile is not None:
        tile = reading['tiles']['3,5']
        if tile is None:
            return None
        matches.append(tile == spec.expected_tile)
    return bool(all(matches)) if matches else None


def score_case(spec, ref, other, generated, alternate, reader, horizon):
    n = min(horizon, len(ref.frames))
    frames = generated['frames'][:n] if generated else None
    readings = [reader.read(frame) for frame in frames] if frames is not None else []
    start_read = reader.read(ref.context[-1])
    events, labels = {}, {}
    for name in ('inventory_gain', 'inventory_consumption', 'damage', 'healing'):
        events[name], labels[name] = [], []
    transition_ok, changed_ok, state_ok = [], [], []
    for t in range(n):
        truth_before = ref.initial if t == 0 else ref.facts[t - 1]
        true = ref.facts[t]
        before = start_read if t == 0 else readings[t - 1] if readings else None
        read = readings[t] if readings else None
        for key in HUD:
            state_ok.append(float(read[key] == true[key]) if read and read[key] is not None else None)
            delta = true[key] - truth_before[key]
            predicted = read[key] - before[key] if read and before and read[key] is not None and before[key] is not None else None
            correct = float(predicted == delta) if predicted is not None else None
            transition_ok.append(correct)
            if delta:
                changed_ok.append(correct)
            if key != 'health':
                for name, sign in (('inventory_gain', 1), ('inventory_consumption', -1)):
                    labels[name].append(sign * delta > 0)
                    events[name].append(sign * predicted > 0 if predicted is not None else None)
            else:
                for name, sign in (('damage', -1), ('healing', 1)):
                    labels[name].append(sign * delta > 0)
                    events[name].append(sign * predicted > 0 if predicted is not None else None)
    row = {'scenario': spec.name, 'category': spec.category, 'valid_attempt': spec.valid,
           'horizon': horizon, 'n_steps': n, 'terminal_censored': n < horizon,
           'milestone_observed': n >= spec.milestone, 'visual_mse': observation_mse(frames, ref.frames[:n]) if frames is not None else None,
           'outcome_accuracy': float(target_outcome(spec, readings[spec.milestone - 1]))
               if readings and n >= spec.milestone and target_outcome(spec, readings[spec.milestone - 1]) is not None else None,
           'state_accuracy': estimate(state_ok), 'transition_accuracy': estimate(transition_ok),
           'changed_transition_accuracy': estimate(changed_ok),
           'events': {name: binary(labels[name], events[name]) for name in labels},
           'reference_return': float(ref.rewards[:n].sum()), 'reward_mae': None,
           'reward_event': binary(np.abs(ref.rewards[:n]) > 1e-6, [None] * n),
           'termination': binary(ref.dones[:n], [None] * n), 'done_brier': None, 'action_following': None,
           'counterfactual_visual_margin': None, 'counterfactual_outcome_accuracy': None,
           'counterfactual_outcome_fields': 0, 'counterfactual_reward_delta_error': None,
           'persistence_success': None}
    if generated and generated['rewards'] is not None:
        r = generated['rewards'][:n]
        row['reward_mae'] = float(np.mean(np.abs(r - ref.rewards[:n])))
        row['reward_event'] = binary(np.abs(ref.rewards[:n]) > 1e-6, np.abs(r) > .1)
    if generated and generated['dones'] is not None:
        d = generated['dones'][:n]
        row['termination'] = binary(ref.dones[:n], d >= .5)
        row['done_brier'] = float(np.mean((d - ref.dones[:n]) ** 2))
    common = min(n, len(other.frames))
    if generated and alternate and common:
        own_error = observation_mse(frames[:common], ref.frames[:common])
        other_error = observation_mse(alternate['frames'][:common], ref.frames[:common])
        row['counterfactual_visual_margin'] = other_error - own_error
        # Meaningful only if the real engine's alternatives differ; invalid vs NOOP often does not.
        if observation_mse(ref.frames[:common], other.frames[:common]) > 1e-8:
            row['action_following'] = float(other_error > own_error + 1e-8) + .5 * float(abs(other_error - own_error) <= 1e-8)
        right = reader.read(alternate['frames'][common - 1])
        left = readings[common - 1]
        comparisons = []
        for key in HUD:
            expected_delta = ref.facts[common - 1][key] - other.facts[common - 1][key]
            if expected_delta:
                comparisons.append(float(np.sign(left[key] - right[key]) == np.sign(expected_delta))
                    if left[key] is not None and right[key] is not None else None)
        row['counterfactual_outcome_fields'] = len(comparisons)
        # Complete-case, never hide unreadable fields in a partial success score.
        if comparisons and all(value is not None for value in comparisons):
            row['counterfactual_outcome_accuracy'] = mean(comparisons)
        if generated['rewards'] is not None and alternate['rewards'] is not None:
            pred_delta = generated['rewards'][:common].sum() - alternate['rewards'][:common].sum()
            real_delta = ref.rewards[:common].sum() - other.rewards[:common].sum()
            row['counterfactual_reward_delta_error'] = float(abs(pred_delta - real_delta))
    if readings and n >= spec.milestone and spec.name in (
            'gather_wood', 'gather_stone', 'place_table', 'place_furnace',
            'consumed_resource_stays_gone', 'table_leave_return', 'craft_progression'):
        # Require continuous independently readable inventory persistence after the milestone.
        required = [key for key in spec.expected if key in HUD]
        checks = [readings[t][key] == ref.facts[t][key] if readings[t][key] is not None else None
                  for t in range(spec.milestone - 1, n) for key in required]
        if spec.expected_tile is not None:
            tile = readings[-1]['tiles']['3,5']
            checks.append(tile == spec.expected_tile if tile is not None else None)
        if checks and all(value is not None for value in checks):
            row['persistence_success'] = float(all(checks))
    return row


SCALARS = ('visual_mse', 'outcome_accuracy', 'reward_mae', 'done_brier', 'action_following',
           'counterfactual_visual_margin', 'counterfactual_outcome_accuracy',
           'counterfactual_reward_delta_error', 'persistence_success')


def aggregate(rows):
    groups = {}
    for row in rows:
        key = f"{row['category']}/h{row['horizon']}/valid={row['valid_attempt']}"
        groups.setdefault(key, []).append(row)
    out = {}
    for key, group in groups.items():
        entry = {'samples': len(group), 'steps': sum(r['n_steps'] for r in group),
                 **{metric: estimate([r[metric] for r in group]) for metric in SCALARS}}
        for event in ('inventory_gain', 'inventory_consumption', 'damage', 'healing', 'reward_event', 'termination'):
            counts = Counter()
            for row in group:
                metric = row[event] if event in ('reward_event', 'termination') else row['events'][event]
                counts.update({k: metric[k] for k in ('tp', 'fp', 'fn', 'tn', 'n', 'total', 'missing',
                                                     'positives_total', 'missing_positives', 'missing_negatives')})
            tp, fp, fn = (counts[k] for k in ('tp', 'fp', 'fn'))
            entry[event] = dict(counts, precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn) if tp + fn else None,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None)
        out[key] = entry
    macro = {}
    for horizon in sorted({r['horizon'] for r in rows}):
        by_category = {cat: [r for r in rows if r['horizon'] == horizon and r['category'] == cat]
                       for cat in sorted({r['category'] for r in rows})}
        macro[str(horizon)] = {metric: estimate([mean([r[metric] for r in group])
            for group in by_category.values()]) for metric in SCALARS}
    return {'groups': out, 'macro_by_horizon': macro}
