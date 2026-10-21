"""Small RGB-reader grid translation tracker. No actions or reference futures enter observation."""

VERSION = 'terrain-consensus-v1'
# Fixed reliability gates, chosen before scoring generated trajectories. These are not probabilities.
MIN_SUPPORT = 2
MIN_TYPES = 2
MIN_AGREEMENT = .8
MIN_MARGIN = .2
MAX_SHIFT = 2
# Grass is uninformative; plants can grow; tested tables/furnaces never localize themselves.
IGNORED = {0, 1, 2, 11, 12, 15, 16}
EXCLUDED = {(3, 4), (3, 5)}  # Player and interaction cell, in both frames.


def cells(reading):
    out = {}
    for key, value in reading['tiles'].items():
        point = tuple(map(int, key.split(',')))
        detail = reading.get('measurements', {}).get('tile:'+key, {})
        if value is not None and point not in EXCLUDED and not detail.get('entity'):
            out[point] = value
    return out


def estimate(previous, current, target_previous=(3,5)):
    """Align overlapping visible terrain; displacement is opposite to screen translation.

    New/non-overlapping terrain is never compared. Entity cells are excluded, even when their terrain
    was recoverable by ReaderV2. Disagreement reduces confidence, never directly fails a mechanic.
    """
    a, b = cells(previous), cells(current)
    candidates = []
    for dr in range(-MAX_SHIFT, MAX_SHIFT+1):
        for dc in range(-MAX_SHIFT, MAX_SHIFT+1):
            support, conflicts = [], []
            for (row,col), value in a.items():
                if (row,col)==target_previous:
                    continue
                dest = (row-dr,col-dc)
                other = b.get(dest)
                if other is None or value in (11,12,15,16) or other in (11,12,15,16):
                    continue
                if value in IGNORED and other in IGNORED:
                    continue
                item = dict(previous=[row,col], current=list(dest), before=value, after=other)
                (support if value == other else conflicts).append(item)
            n = len(support)+len(conflicts)
            candidates.append(dict(displacement=[dr,dc], camera_translation=[-dr,-dc],
                support=support, conflicts=conflicts, supporting_landmarks=len(support),
                landmark_types=len({p['before'] for p in support}),
                agreement=len(support)/n if n else 0.))
    candidates.sort(key=lambda c:(c['agreement'],c['supporting_landmarks']),reverse=True)
    eligible = [c for c in candidates if c['supporting_landmarks']>=MIN_SUPPORT and c['landmark_types']>=MIN_TYPES]
    best = eligible[0] if eligible else candidates[0]
    runner = eligible[1] if len(eligible)>1 else None
    margin = best['agreement']-(runner['agreement'] if runner else 0.)
    accepted = bool(eligible and best['agreement']>=MIN_AGREEMENT and margin>=MIN_MARGIN)
    reason = ('accepted_terrain_translation' if accepted else 'insufficient_distinct_landmarks' if not eligible
              else 'incoherent_landmark_motion' if best['agreement']<MIN_AGREEMENT else 'ambiguous_translation')
    return dict(version=VERSION, accepted=accepted, displacement=best['displacement'] if accepted else None,
                camera_translation=best['camera_translation'] if accepted else None, reason=reason,
                confidence=dict(agreement=best['agreement'], margin=margin,
                                supporting_landmarks=best['supporting_landmarks'], landmark_types=best['landmark_types']),
                support=best['support'],conflicts=best['conflicts'],
                ambiguities=[dict(displacement=c['displacement'],agreement=c['agreement'],
                                  supporting_landmarks=c['supporting_landmarks']) for c in eligible[1:]
                             if best['agreement']-c['agreement']<MIN_MARGIN],
                thresholds=dict(min_support=MIN_SUPPORT,min_types=MIN_TYPES,min_agreement=MIN_AGREEMENT,
                                min_margin=MIN_MARGIN,max_shift=MAX_SHIFT))


def track(readings, anchor=0):
    """Accumulate adjacent observations; recover after gaps only with independent anchor alignment."""
    records = [dict(timestep=t,position=None,reason='before_anchor') for t in range(len(readings))]
    records[anchor] = dict(timestep=anchor,position=[0,0],reason='visual_anchor',confidence=None)
    for t in range(anchor+1,len(readings)):
        previous = records[t-1]['position']
        target = (3-previous[0],5-previous[1]) if previous is not None else None
        step = estimate(readings[t-1],readings[t],target_previous=target)
        position = [a+b for a,b in zip(previous,step['displacement'])] if previous is not None and step['accepted'] else None
        recovery = None
        if position is None:
            recovery = estimate(readings[anchor],readings[t])
            if recovery['accepted']:position=recovery['displacement']
        source = recovery if recovery else step
        reason = ('localization_gap:'+source['reason'] if position is None and previous is None else source['reason'])
        records[t] = dict(timestep=t,position=position,reason=reason,confidence=source['confidence'],
                          adjacent=step,relocalization=recovery)
    return records


def displacement(readings, timestep, expected, anchor=0):
    from .atomic import scalar
    if timestep>=len(readings):
        return scalar('displacement','relative_displacement',list(expected),None,
                      reason='milestone_not_reached',milestone=timestep,confidence=None,motion_trace=[])
    records = track(readings[:timestep+1],anchor)
    record = records[timestep]
    return scalar('displacement','relative_displacement',list(expected),record['position'],
                  reason=record['reason'],milestone=timestep,motion=record,motion_trace=records,
                  confidence=record.get('confidence'),anchor_timestep=anchor)


def object_observation(reading, position, origin=(3,5), expected_tile=None, localization=None):
    """An object's anchor coordinate is fixed; player motion changes its visible screen coordinate."""
    point = [origin[0]-position[0],origin[1]-position[1]] if position is not None else None
    visible=point is not None and 0<=point[0]<7 and 0<=point[1]<9
    key=','.join(map(str,point)) if visible else None
    value=reading['tiles'].get(key) if visible else None
    detail=reading.get('measurements',{}).get('tile:'+key) if visible else None
    from craftax.craftax_classic.constants import BlockType
    labels={b.value:b.name.lower() for b in BlockType}
    reason='uncertain_localization' if position is None else 'target_out_of_view' if not visible else 'target_observed' if value is not None else 'target_unreadable'
    return dict(value=value,coordinate=point,reason=reason,pixel_evidence=detail,
        target_coordinate=point,estimated_player_position=position,
        localization_status='known' if position is not None else 'unknown',localization_evidence=localization,
        target_visibility='visible' if visible else 'unknown' if position is None else 'out_of_view',
        tile_reader_status='accepted' if value is not None else 'rejected' if visible else 'not_observed',
        tile_reader_reason=(detail or {}).get('reason',reason),expected_tile=expected_tile,observed_tile=value,
        expected_label=labels.get(expected_tile) if isinstance(expected_tile,int) else None,observed_label=labels.get(value),
        candidate_labels={k:labels.get((detail or {}).get(k)) for k in ('candidate','runner_up')},
        entity_occlusion={'status':'recognized','entity':detail['entity']} if detail and detail.get('entity') else
                         {'status':'not_detected' if value is not None else 'unresolved' if visible else 'not_observed'},
        confidence=dict(localization=(localization or {}).get('confidence'),tile=detail))
