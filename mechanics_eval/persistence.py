"""Visible object persistence, independent of a fixed excursion or future reference images."""
from .atomic import check, result, scalar
from .motion import track, object_observation


def visible_persistence(readings, expected_tile, anchor=1, absence=False):
    trajectory=track(readings,anchor);facts=[];observations=[]
    for t in range(anchor+1,len(readings)):
        evidence=object_observation(readings[t],trajectory[t]['position'],expected_tile=expected_tile,localization=trajectory[t])
        evidence['expected_presence']=not absence
        observation=dict(evidence,timestep=t,status='not_observed' if evidence['target_visibility']=='out_of_view' else 'observed')
        observations.append(observation)
        if observation['status']=='not_observed':continue
        value=evidence['value'];expected=True if absence else expected_tile
        observed=value!=expected_tile if absence and value is not None else value
        facts.append(dict(check('target_remains_absent' if absence else 'target_persists',expected,observed,evidence['reason']),
                          timestep=t,object_evidence=evidence,motion=trajectory[t]))
    raw=result('visible_object_persistence',facts) if facts else scalar('visible_object_persistence','visible_observation',True,None,reason='no_visible_observations')
    raw.update(motion_trace=trajectory,object_anchor_coordinate=[3,5],anchor_timestep=anchor,observations=observations,
               out_of_view_observations=sum(o['status']=='not_observed' for o in observations))
    return raw
