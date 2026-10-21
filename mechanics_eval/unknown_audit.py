"""Explain saved UNKNOWN/NOT_TESTED/NOT_APPLICABLE and missing fact observations without inference."""
from collections import Counter
import json
from pathlib import Path
from PIL import Image
from .evidence import load_case,dump_json,digest
from .motion import object_observation
from .visual import HUD,hud_patch


def diagnose(raw,stage,fact,t,readings,actions):
    name=stage.get('name',raw.get('mechanic',''));field=fact.get('field','')
    status=stage.get('status',raw.get('status'));reason=fact.get('reason') or stage.get('observation_reason') or stage.get('reason') or raw.get('reason')
    motion=fact.get('motion') or raw.get('motion')
    trace=raw.get('motion_trace',[])
    if t<len(trace):motion=trace[t]
    evidence=dict(component=None,root_cause=reason,observability=None)
    world_target=('object_persistence' in name or 'tracked_table' in field or 'target_persist' in field or 'target_remains' in field)
    screen_target=field in ('table_present','target_removed','harvest_observed') or field=='tile:3,5'
    if world_target or field.startswith('tile:') or screen_target:
        position=(motion or {}).get('position')
        original=fact.get('object_evidence') or raw.get('object_evidence') or {}
        if position is None and original.get('coordinate') is not None:
            position=[3-original['coordinate'][0],5-original['coordinate'][1]]
        if screen_target and not world_target:position=[0,0]
        expected=11 if 'table' in name or 'table' in field or name.endswith('after_return') or name.endswith('.visible') else fact.get('expected')
        if 'harvested_resource' in name or field in ('target_removed','harvest_observed'):
            expected={'absent':readings[0]['tiles'].get('3,5')}
        target=object_observation(readings[t],position,expected_tile=expected,localization=motion)
        if screen_target and not world_target:
            target.update(coordinate_basis='explicit_screen_cell',estimated_player_position=None,
                          localization_status='not_required_for_screen_relative_measurement')
        evidence['target']=target
        if not any(int(a) in (8,9) for a in actions) and readings[0]['tiles'].get('3,5')!=11 and ('table' in name or 'table' in field or name.endswith(('.after_return','.visible'))):
            evidence.update(component='branch_applicability',root_cause='branch_never_requested_object_placement',observability='not_applicable_scoring_logic')
            target.update(localization_status='not_applicable',target_visibility='not_applicable',tile_reader_status='not_applicable',
                          tile_reader_reason='branch_without_placement',expected_tile=None,observed_tile=None)
        elif status=='not_tested':
            evidence.update(component='dependency_gate',root_cause=stage.get('reason'),observability='prerequisite_not_established',
                            prerequisites=stage.get('prerequisites'))
        elif position is None:
            evidence.update(component='motion_tracker',root_cause=(motion or {}).get('reason') or reason or 'unknown_player_position',
                            observability='localization_uncertain_no_target_coordinate')
        elif target['target_visibility']=='out_of_view':
            evidence.update(component='visibility',root_cause='target_out_of_view',observability='genuinely_outside_rgb_viewport')
        elif target['value'] is None:
            evidence.update(component='tile_reader',root_cause=target['tile_reader_reason'],observability='visible_pixels_reader_abstained')
        else:
            evidence.update(component='scoring_logic',root_cause=reason,observability='target_readable_missing_fact_caused_by_logic')
    elif status in ('not_tested','not_applicable'):
        evidence.update(component='branch_applicability' if status=='not_applicable' else 'dependency_gate',root_cause=stage.get('reason') or reason,
                        observability='not_applicable_scoring_logic' if status=='not_applicable' else 'prerequisite_not_established',
                        prerequisites=stage.get('prerequisites'))
    elif reason in ('requires_stationary_noop_suffix','branch_not_noop','branch_without_harvest','harvest_not_observed_at_anchor'):
        evidence.update(component='branch_applicability' if reason!='harvest_not_observed_at_anchor' or not any(int(a)==5 for a in actions) else 'dependency_gate',
                        root_cause=reason,observability='branch_or_establishment_scoring_logic')
    elif reason=='stochastic_effect_not_exactly_scored':
        evidence.update(component='stochastic_target_policy',root_cause=reason,observability='exact_reference_outcome_not_identifiable',
                        observed_delta=raw.get('observed_delta'))
    elif 'reference_route_not_comparable'==reason:
        evidence.update(component='reference_comparability_gate',root_cause=reason,observability='geometry_readable_reference_route_incomparable',
                        measured_displacement=raw.get('measured_displacement'),route_comparability=raw.get('route_comparability'))
    elif 'displacement' in field or field=='legal_single_step':
        if motion is None:
            from .motion import track
            motion=track(readings[:t+1])[t]
            evidence['localization_recomputed_for_audit']=True
        evidence.update(component='motion_tracker',root_cause=(motion or {}).get('reason') or reason,observability='localization_uncertain',motion=motion)
        if fact.get('expected') is None or reason in ('entity_collision_uncertain','collision_tile_unreadable'):
            from craftax.craftax_classic.constants import DIRECTIONS
            direction=list(map(int,DIRECTIONS[int(actions[t-1])])) if t else [0,0]
            key=f'{3+direction[0]},{4+direction[1]}'
            evidence.update(component='collision_reader',root_cause=reason,observability='collision_or_entity_order_uncertain',
                collision_coordinate=key,reader=readings[max(0,t-1)].get('measurements',{}).get('tile:'+key))
    elif field in ('sleeping','direction','target_zombie_killed'):
        player = readings[t].get('player_evidence')
        target = readings[t].get('target_zombie_evidence')
        detail = target if field == 'target_zombie_killed' else player
        evidence.update(component='temporal_target_reader' if field == 'target_zombie_killed' else 'player_pose_reader',
                        root_cause=(detail or {}).get('reason') or reason,
                        observability='rgb_reader_abstained' if detail else 'reader_not_available_in_historical_measurements',
                        reader=detail)
    elif field == 'done':
        evidence.update(component='unsupported_terminal_observation',root_cause='no_general_terminal_signal_in_rgb',
                        observability='terminal_flag_not_identifiable_from_health_or_sprite_alone')
    else:
        hud=field.removesuffix('_created')
        if hud in HUD:
            endpoints=[dict(timestep=i,value=readings[i].get(hud),reader=readings[i].get('measurements',{}).get(hud))
                       for i in sorted({max(0,t-1),t})]
            causes=[e['reader']['digit']['reason'] for e in endpoints if e['value'] is None and e['reader']]
            evidence.update(component='hud_reader',root_cause=','.join(dict.fromkeys(causes)) or reason,
                            observability='visible_hud_reader_abstained',endpoints=endpoints)
        else:evidence.update(component='unsupported_rgb_reader',root_cause=reason or 'reader_not_implemented',observability='not_implemented_not_proven_fundamentally_unobservable')
    return evidence


def audit(results_files,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);(out/'frames').mkdir();records=[];sources=[]
    for suite,path in enumerate(map(Path,results_files)):
        report=json.loads(path.read_text());sources.append(dict(path=str(path.resolve()),sha256=digest(path),metric_version=report['metric_version']))
        for row in report['samples']:
            name=Path(row['evidence']).name;meta,arrays=load_case(row['evidence'])
            cache=path.parent/name/'measurements.json'
            if not cache.exists():
                cache=Path(report['measurement_replay']['source'])/name/'measurements.json'
            measured=json.loads(cache.read_text())
            for branch in range(2):
                readings=[measured['context_last']]+measured['generated'][branch]
                groups=[('primary',row['branch_outcomes'][branch],{})]+[
                    ('stage',s,s) for s in row.get('measurements_by_branch',[[],[]])[branch]]
                for stage in row.get('measurements_by_branch',[[],[]])[branch]:
                    for observation in stage.get('observations',[]):
                        if observation.get('status')=='not_observed':
                            raw=dict(stage,status='not_observed',facts=[dict(field='tile:tracked_object',
                                expected=observation.get('expected_tile'),observed=None,timestep=observation['timestep'],
                                reason=observation.get('reason'),object_evidence=observation)])
                            groups.append(('visibility',raw,stage))
                for source,raw,stage in groups:
                    facts=[f for f in raw.get('facts',[]) if f.get('observed') is None]
                    if not facts and raw['status'] in ('unknown','not_tested','not_applicable'):
                        facts=[dict(field='stage_applicability',expected=stage.get('expected'),observed=None,
                                    reason=stage.get('observation_reason') or stage.get('reason') or raw.get('reason'))]
                    for fact in facts:
                        t=int(fact.get('timestep',raw.get('milestone',stage.get('timestep',meta['spec']['milestone']))))
                        t=min(t,len(readings)-1)
                        explanation=diagnose(raw,stage,fact,t,readings,arrays[f'b{branch}_actions'])
                        record=dict(suite=suite,cohort=report.get('cohort',path.parent.name),scenario=row['scenario'],engine_seed=row['seed'],model_seed=meta['model_seed'],
                            branch=branch,source=source,mechanic=stage.get('name',raw['mechanic']),status=raw['status'],
                            timestep=t,expected=fact.get('expected'),observed=fact.get('observed'),field=fact['field'],
                            explanation=explanation,reader_measurements=str(cache.resolve()),rgb_evidence=str(Path(row['evidence']).resolve()))
                        if 'target' in explanation:
                            basename=f'{suite}-{name}-b{branch}-t{t}';frame=arrays['context'][-1] if t==0 else arrays[f'b{branch}_generated'][t-1]
                            frame_path=out/'frames'/f'{basename}.png'
                            if not frame_path.exists():Image.fromarray(frame).resize((378,378),Image.Resampling.NEAREST).save(frame_path)
                            record['frame']=str(frame_path.resolve());point=explanation['target']['target_coordinate']
                            if point is not None and explanation['target']['target_visibility']=='visible':
                                r,c=point;crop=out/'frames'/f'{basename}-target-{r}-{c}.png'
                                if not crop.exists():Image.fromarray(frame[r*7:r*7+7,c*7:c*7+7]).resize((140,140),Image.Resampling.NEAREST).save(crop)
                                record['target_crop']=str(crop.resolve())
                        if explanation.get('component')=='hud_reader':
                            field=fact['field'].removesuffix('_created');record['hud_crops']=[]
                            for endpoint in explanation['endpoints']:
                                i=endpoint['timestep'];frame=arrays['context'][-1] if i==0 else arrays[f'b{branch}_generated'][i-1]
                                crop=out/'frames'/f'{suite}-{name}-b{branch}-t{i}-hud-{field}.png'
                                if not crop.exists():Image.fromarray(hud_patch(frame,field)).resize((140,140),Image.Resampling.NEAREST).save(crop)
                                record['hud_crops'].append(dict(timestep=i,path=str(crop.resolve())))
                        records.append(record)
    tracked=[r for r in records if r['field']=='tile:tracked_table']
    summary=dict(sources=sources,records=len(records),record_semantics='Logical primary/stage facts, including aliases; not independent scenarios.',
        by_component=dict(Counter(r['explanation']['component'] for r in records)),
        by_root_cause=dict(Counter(r['explanation']['root_cause'] for r in records)),
        tracked_table_none=tracked)
    dump_json(out/'records.json',records);dump_json(out/'summary.json',summary)
    lines=['# Missing-observation audit','',f'{len(records)} logical fact/stage records; aliases are labelled and are not extra attempts.','',
           '| Cohort | Scenario | Engine seed | Model seed | Branch | t | Mechanic | Status | Component | Exact reason | RGB evidence |','|---|---|---:|---:|---:|---:|---|---|---|---|---|']
    for r in records:
        links=([f'[frame]({r["frame"]})'] if r.get('frame') else [])+([f'[target]({r["target_crop"]})'] if r.get('target_crop') else [])+[
            f'[HUD t={c["timestep"]}]({c["path"]})' for c in r.get('hud_crops',[])]
        e=r['explanation'];lines.append(f'| {r["cohort"]} | {r["scenario"]} | {r["engine_seed"]} | {r["model_seed"]} | {r["branch"]} | {r["timestep"]} | {r["mechanic"]} | {r["status"]} | {e["component"]} | {e["root_cause"]} | '+', '.join(links)+' |')
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    return summary
