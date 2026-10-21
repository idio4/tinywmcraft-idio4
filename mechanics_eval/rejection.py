"""Departure-only eligibility for the conditional fixed six-tile table excursion."""
from craftax.craftax_classic.constants import SOLID_BLOCKS
from .motion import track

VERSION = 'newly-revealed-blocker-v1'


def sampling_seed(first_seed, sample_index, step=1):
    if not 0 <= sample_index < 2**32 or not 0 < step < 2**32 or step % 2 == 0:
        raise ValueError('seed step must be odd and positive; sample index must be uint32')
    return (int(first_seed) + int(sample_index)*int(step)) % 2**32


def departure_eligibility(readings):
    """Only context + placement + six UP frames are inspected, never return or object outcomes.

    Reject only a stable solid first seen when its world-relative cell entered the viewport. Any
    earlier readable free appearance, departure error, localization gap or ambiguous collision prevents
    rejection. This conservative rule cannot turn an unreadable attempt into a better sampled one.
    """
    readings=readings[:8]
    if len(readings)<8:
        return dict(decision='ELIGIBLE',reason='incomplete_departure_evidence',witnesses=[],failure_evidence=[],ambiguities=[])
    trajectory=track(readings);history={};failures=[];ambiguities=[];blockers=[]
    solids=set(map(int,SOLID_BLOCKS))
    for t,reading in enumerate(readings):
        position=trajectory[t]['position']
        if t==1:
            initial_step=trajectory[t]['adjacent']['displacement']
            if initial_step is None:ambiguities.append(dict(timestep=t,reason='placement_localization_uncertain',localization=trajectory[t]))
            elif initial_step!=[0,0]:failures.append(dict(timestep=t,reason='movement_during_placement',measured_step=initial_step,localization=trajectory[t]))
        if t>=2:
            previous=trajectory[t-1]['position'];step=trajectory[t]['adjacent']['displacement']
            patch=readings[t-1].get('measurements',{}).get('tile:2,4',{})
            tile=readings[t-1]['tiles'].get('2,4');entity=patch.get('entity')
            evidence=dict(timestep=t,measured_step=step,tile=tile,entity=entity,tile_evidence=patch,
                          localization=trajectory[t])
            if step is None or previous is None or any(r['position'] is None for r in trajectory[:t]):
                ambiguities.append(dict(evidence,reason='departure_localization_uncertain'))
            elif step not in ([0,0],[-1,0]):
                failures.append(dict(evidence,reason='wrong_direction_or_teleport'))
            elif tile is None or entity:
                ambiguities.append(dict(evidence,reason='unreadable_collision' if tile is None else 'entity_collision_uncertain'))
            elif tile not in solids:
                world=(previous[0]-1,previous[1])
                if any(s['tile'] in solids for s in history.get(world,[]) if s['tile'] is not None):
                    failures.append(dict(evidence,reason='previously_visible_solid_changed_without_interaction'))
                if step==[0,0]:failures.append(dict(evidence,reason='frozen_on_visible_free_tile'))
            elif step==[-1,0]:
                failures.append(dict(evidence,reason='movement_through_visible_solid'))
            else:
                world=(previous[0]-1,previous[1]);seen=history.get(world,[])
                evidence.update(world_relative_coordinate=list(world),visibility_history=list(seen))
                if any(s['tile'] is not None and s['tile'] not in solids for s in seen):
                    failures.append(dict(evidence,reason='previously_visible_free_tile_became_solid'))
                elif (seen and seen[0]['timestep']>0 and len(seen)>=2 and
                      all(s['tile']==tile for s in seen)):
                    blockers.append(dict(evidence,reason='stable_solid_in_newly_revealed_cell'))
                else:
                    ambiguities.append(dict(evidence,reason='obstacle_not_proven_newly_revealed_and_stable'))
        # Even unreadable cells count as previously visible. Never equate unreadable with unseen.
        if position is not None:
            for row in range(7):
                for col in range(9):
                    if (row,col)==(3,4):continue
                    key=f'{row},{col}';world=(position[0]+row-3,position[1]+col-4)
                    history.setdefault(world,[]).append(dict(timestep=t,screen_coordinate=[row,col],
                        tile=reading['tiles'].get(key),reader=reading.get('measurements',{}).get('tile:'+key)))
    decision='REJECTED_BLOCKED_ROUTE' if blockers and not failures and not ambiguities else 'ELIGIBLE'
    reason=('positive_newly_revealed_blocker' if decision=='REJECTED_BLOCKED_ROUTE' else
            'preserve_departure_failure' if failures else 'preserve_ambiguous_departure' if ambiguities else 'unblocked_departure')
    return dict(version=VERSION,decision=decision,reason=reason,witnesses=blockers,
                failure_evidence=failures,ambiguities=ambiguities,departure_motion=trajectory)


def retry_selection(attempts, limit):
    """No pass/fail score is consulted when selecting a trajectory."""
    for index,attempt in enumerate(attempts[:limit]):
        if attempt['eligibility']['decision']!='REJECTED_BLOCKED_ROUTE':
            return dict(status='SELECTED_ELIGIBLE',selected_index=index,retries=index)
    exhausted=len(attempts)>=limit
    return dict(status='RETRY_BUDGET_EXHAUSTED' if exhausted else 'NEEDS_SAMPLE',
                selected_index=None,retries=max(0,min(len(attempts),limit)-1))
