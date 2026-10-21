"""First-step identity for an initially visible adjacent zombie, using RGB only.

Ordinary engine movement/spawn bounds rule out distant replacement zombies. An
unreadable reachable cell, viewport loss, occlusion or competing identity abstains.
This does not attempt identity reacquisition or general long-term entity tracking.
"""
from .motion import track

VERSION = 'bounded-visible-zombie-v1'


def target_zombie(readings, origin=(3,5)):
    initial = readings[0]
    key = ','.join(map(str,origin))
    known = initial.get('measurements',{}).get('tile:'+key,{}).get('entity') == 'zombie'
    # A competing zombie can enter the original target's reachable region in one
    # step. An unreadable nearby cell could hide that competitor; never guess.
    for r in range(7):
        for c in range(9):
            if (r,c) not in (origin,(3,4)) and abs(r-origin[0])+abs(c-origin[1]) <= 2:
                k=f'{r},{c}';detail=initial.get('measurements',{}).get('tile:'+k,{})
                if initial['tiles'].get(k) is None or detail.get('entity'):
                    known=False
    motion=track(readings[:2]);records=[]
    for t,reading in enumerate(readings):
        result=None;reachable=[];candidates=[];unclear=[];camera=motion[t] if t<2 else None
        reason='initial_target_or_identity_unreadable'
        if t>1:
            reason='one_step_identity_scope_exceeded'
        elif t==0 and known:
            result=False;reason='initial_target_visible'
        elif t==1 and known:
            before,now=motion[0]['position'],motion[1]['position']
            if before is None or now is None:
                reason='target_camera_localization_unknown'
            else:
                shift=[now[i]-before[i] for i in range(2)]
                reachable=sorted({(origin[0]+dr-shift[0],origin[1]+dc-shift[1])
                    for dr,dc in ((0,0),(1,0),(-1,0),(0,1),(0,-1))}-{(3,4)})
                for point in reachable:
                    k=','.join(map(str,point));detail=reading.get('measurements',{}).get('tile:'+k,{})
                    if not (0<=point[0]<7 and 0<=point[1]<9) or reading['tiles'].get(k) is None:
                        unclear.append(point)
                    elif detail.get('entity')=='zombie':
                        candidates.append(point)
                    elif detail.get('entity'):
                        unclear.append(point)
                if len(candidates)==1 and not unclear:
                    result=False;reason='unique_visible_target_continuation'
                elif not candidates and not unclear:
                    result=True;reason='entire_target_reachable_region_visibly_empty'
                else:
                    reason='target_occlusion_or_identity_ambiguity'
        evidence=dict(version=VERSION,reason=reason,timestep=t,initial_target_coordinate=list(origin),
            initial_identity_verified=bool(known),reachable_positions=[list(p) for p in reachable],
            zombie_candidates=[list(p) for p in candidates],unreadable_or_occluded=[list(p) for p in unclear],camera=camera)
        reading['target_zombie_killed']=result;reading['target_zombie_evidence']=evidence;records.append(evidence)
    return records
