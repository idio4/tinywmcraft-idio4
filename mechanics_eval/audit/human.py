"""Offline, lossless human review. Annotations never alter independent mechanics scores."""
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import hashlib
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image
from ..evidence import digest, dump_json, load_case
from ..gifs import ACTION_NAMES
from ..motion import track, object_observation
from ..visual import HUD, hud_patch

SCHEMA = 1
DECISIONS = ('CONFIRM_AUTOMATIC','READER_ERROR','SCORING_ERROR','RESOLVABLE_UNKNOWN',
             'GENUINELY_UNOBSERVABLE','NEEDS_FURTHER_REVIEW')
STATUSES = ('pass','fail','unknown','not_tested','not_applicable')
CURRENT = ('scores/existing/results.json','scores/focused/results.json',
           'scores/controlled/results.json','scores/table/results.json',
           'conditional/unconditional-results.json','conditional/conditional-results.json',
           'conditional/all-attempt-scores/results.json')


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def physical_hash(meta, arrays, branch):
    h=hashlib.sha256()
    h.update(json.dumps([meta['seed'],meta['model_seed'],branch]).encode())
    n=len(arrays[f'b{branch}_reference'])
    for a in (arrays['context'][-1:],arrays[f'b{branch}_generated'][:n],arrays[f'b{branch}_actions'][:n]):
        h.update(str((a.shape,str(a.dtype))).encode());h.update(a.tobytes())
    return h.hexdigest()


def family(name,category):
    if 'direction' in name:return 'facing_direction'
    if category=='Movement' or any(x in name for x in ('displacement','return_to_origin','action_consequences','fixed_route')):return 'movement'
    if 'table' in name or name in ('object_persistence.visible','object_persistence.after_return'):return 'object.crafting_table'
    if 'furnace' in name:return 'object.furnace'
    if any(x in name for x in ('harvested_resource','target_removal')):return 'object.harvested_resource'
    return category.lower().replace(' ','_')


def category_for(raw,row):
    if raw.get('category'):return raw['category']
    name=raw.get('name',raw['mechanic'])
    if name.startswith('inventory_delta'):return 'Inventory accounting'
    if 'persistence' in name:return 'Object persistence' if 'resource' in name or 'table' in name else 'Inventory persistence'
    return {'valid_crafting':'Crafting','invalid_crafting':'Crafting','crafting':'Crafting',
        'resource_collection':'Resource collection','invalid_interaction':'Resource collection','gathering':'Resource collection',
        'resource_accounting':'Inventory accounting','movement':'Movement','placement':'Placement',
        'combat':'Combat and survival','other':'Inventory persistence'}.get(row['category'],row['category'])


def records_for(row,branch):
    """Include diagnostic and compatibility records too; aliases are deduplicated below."""
    stages=row['measurements_by_branch'][branch]
    for i,raw in enumerate(stages):yield 'stage',i,raw
    yield 'primary',0,row['branch_outcomes'][branch]
    for name,raw in row.get('diagnostics_by_branch',[{},{}])[branch].items():yield 'diagnostic',name,raw


def compact(raw):
    keys=('mechanic','name','category','status','required','eligible','timestep','milestone','reason',
          'observation_reason','prerequisites','facts','expected','observed','object_evidence','observations',
          'anchor_timestep','object_anchor_coordinate','excursion_requirement','route_comparability','motion')
    result={k:deepcopy(raw[k]) for k in keys if k in raw}
    # Motion is represented once per frame below, not repeated in every fact/observation.
    for f in result.get('facts',[]):f.pop('motion',None)
    for o in result.get('observations',[]):o.pop('localization_evidence',None)
    if result.get('object_evidence'):result['object_evidence'].pop('localization_evidence',None)
    return result


def question_key(raw):
    facts=raw.get('facts',[])
    name=raw.get('name',raw['mechanic'])
    if name in ('object_persistence.after_return','movement.return_to_origin'):
        return [(name,int(raw.get('milestone',raw.get('timestep',13))))]
    # Same physical fact/time across primary and diagnostic aliases is one review question.
    if facts:
        return sorted({(f['field'],int(f.get('timestep',raw.get('milestone',raw.get('timestep',1))))) for f in facts})
    return [('applicability:'+raw.get('name',raw['mechanic']),int(raw.get('timestep',raw.get('milestone',1))))]


def read_sources(paths):
    missing=[];reports=[]
    for path in map(Path,paths):
        if not path.is_file():missing.append(str(path));continue
        report=json.loads(path.read_text());reports.append((path,report))
        for row in report['samples']:
            evidence=Path(row['evidence'])
            for name in ('metadata.json','trajectory.npz'):
                if not (evidence/name).is_file():missing.append(str(evidence/name))
            cache=path.parent/evidence.name/'measurements.json'
            if not cache.is_file():
                replay=report.get('measurement_replay') or {}
                cache=Path(replay.get('source',path.parent))/evidence.name/'measurements.json'
            if not cache.is_file():missing.append(str(cache))
            if evidence.parent.name=='evidence' and 'conditional-' in str(evidence.parent.parent):
                if not (evidence/'eligibility.json').is_file():missing.append(str(evidence/'eligibility.json'))
    if missing:raise FileNotFoundError('Required local review evidence is unavailable:\n'+'\n'.join(sorted(set(missing))))
    return reports


def build_items(paths,fail_limit=4,pass_limit=2):
    if min(fail_limit,pass_limit)<0:raise ValueError('Sample limits must be nonnegative')
    reports=read_sources(paths);events={};trajectories={};sources=[];unknown_records=[]
    for path,report in reports:
        source_id=identity(str(path.resolve()));cohort=report.get('cohort',path.parent.name)
        sources.append(dict(path=str(path.resolve()),sha256=digest(path),metric_version=report['metric_version']))
        for row in report['samples']:
            evidence=Path(row['evidence']);meta,arrays=load_case(evidence)
            eligibility_path=evidence/'eligibility.json'
            eligibility=json.loads(eligibility_path.read_text()) if eligibility_path.exists() else None
            cache=path.parent/evidence.name/'measurements.json'
            if not cache.exists():cache=Path(report['measurement_replay']['source'])/evidence.name/'measurements.json'
            readings=json.loads(cache.read_text())
            for branch in range(2):
                ph=physical_hash(meta,arrays,branch)
                if ph not in trajectories:
                    branch_reads=[readings['context_last']]+readings['generated'][branch]
                    trajectories[ph]=dict(meta=meta,arrays=arrays,branch=branch,readings=branch_reads,
                        motion=track(branch_reads[:len(arrays[f'b{branch}_reference'])+1]),evidence=evidence)
                for role,pointer,raw in records_for(row,branch):
                    name=raw.get('name',raw['mechanic']);category=category_for(raw,row)
                    group=family(name,category);event_id=identity([ph,group]);event=events.setdefault(event_id,dict(
                        id=event_id,evidence_hash=ph,scenario=row['scenario'],engine_seed=meta['seed'],model_seed=meta['model_seed'],
                        branch=branch,branch_label='selected' if branch==0 else 'NOOP alternative' if not np.any(arrays['b1_actions']) else 'action alternative',
                        category=category,family=group,scope=row.get('scope'),questions={},cohorts=set(),groups=set()))
                    event['cohorts'].add(cohort)
                    if eligibility and eligibility['decision']=='REJECTED_BLOCKED_ROUTE':event['cohorts'].add('REJECTED_ATTEMPT')
                    qid=identity([ph,group,question_key(raw)]);question=event['questions'].setdefault(qid,dict(
                        id=qid,mechanics=set(),records={},groups=set(),times=set()))
                    question['mechanics'].add(name)
                    record=compact(raw);record['category']=category
                    rid=identity(record);entry=question['records'].setdefault(rid,dict(id=rid,automatic=record,associations=[]))
                    assoc=dict(source=source_id,results=str(path.resolve()),case=evidence.name,cohort=cohort,role=role,pointer=pointer,
                        required=raw.get('required',False),metric_version=report['metric_version'],scorer_sha256=report['scorer_sha256'])
                    if assoc not in entry['associations']:entry['associations'].append(assoc)
                    times=[int(f.get('timestep',raw.get('milestone',raw.get('timestep',1)))) for f in raw.get('facts',[])]
                    question['times'].update(times or [int(raw.get('timestep',raw.get('milestone',1)))])
                    if raw['status']=='unknown':question['groups'].add('A');unknown_records.append((source_id,evidence.name,branch,role,str(pointer),qid))
                    if raw['status']=='not_tested' and (category in ('Object persistence','Movement','Inventory persistence') or 'prerequisite' in str(raw.get('reason'))):question['groups'].add('B')
                    conditional=report.get('cohort')=='CONDITIONAL_SELECTED_ELIGIBLE'
                    if raw['status']=='fail' and (name in ('movement.return_to_origin','object_persistence.visible') or
                            conditional and name in ('object_persistence.after_return','movement.fixed_route_fidelity')):question['groups'].add('C')
                    if raw['status']=='not_applicable':question['groups'].add('NA')
            # Eligibility is its own departure-only review, with return pixels deliberately unavailable.
            ep=evidence/'eligibility.json'
            if ep.exists():
                eligibility=json.loads(ep.read_text())
                if eligibility['decision']=='REJECTED_BLOCKED_ROUTE' or eligibility['reason']=='preserve_ambiguous_departure':
                    ph=physical_hash(meta,arrays,0);eid=identity([ph,'departure_eligibility'])
                    event=events.setdefault(eid,dict(id=eid,evidence_hash=ph,scenario=row['scenario'],engine_seed=meta['seed'],model_seed=meta['model_seed'],
                        branch=0,branch_label='selected',category='Rejection eligibility',family='departure_eligibility',scope='static',
                        questions={},cohorts=set(),groups=set()))
                    event['cohorts'].add('REJECTED_BLOCKED_ROUTE' if eligibility['decision']=='REJECTED_BLOCKED_ROUTE' else 'RETAINED_AMBIGUOUS')
                    qid=identity([ph,'departure_eligibility',[1,7]])
                    q=event['questions'].setdefault(qid,dict(id=qid,mechanics={'departure eligibility'},records={},groups={'D'},times=set(range(1,8))))
                    raw=dict(mechanic='departure eligibility',status=eligibility['decision'],reason=eligibility['reason'],required=False,
                             facts=[],eligibility=eligibility)
                    rid=identity(raw);record=q['records'].setdefault(rid,dict(id=rid,automatic=raw,associations=[]))
                    association=dict(cohort=cohort,results=str(path.resolve()),metric_version=report['metric_version'],
                        role='eligibility',required=False,scorer_sha256=report['scorer_sha256'])
                    if association not in record['associations']:record['associations'].append(association)
    # Deterministic round-robin by scenario + reason, then seed; limits apply per category.
    for status,limit,letter in [('fail',fail_limit,'C'),('pass',pass_limit,'D')]:
        candidates=defaultdict(lambda:defaultdict(list))
        for e in events.values():
            if e['family']=='departure_eligibility':continue
            for q in e['questions'].values():
                for r in q['records'].values():
                    a=r['automatic']
                    if a['status']==status:
                        if status=='fail' and 'C' in q['groups']:continue
                        stratum=(e['scenario'],a.get('reason',a.get('observation_reason','')),tuple(sorted(q['mechanics'])))
                        candidates[a.get('category',e['category'])][stratum].append((e['engine_seed'],e['model_seed'],e['branch'],q['id'],q))
        for strata in candidates.values():
            ordered=[sorted(v,key=lambda x:x[:-1]) for _,v in sorted(strata.items(),key=lambda x:str(x[0]))]
            chosen=set()
            while len(chosen)<limit and any(ordered):
                for bucket in ordered:
                    if bucket and len(chosen)<limit:
                        q=bucket.pop(0)[-1]
                        if q['id'] not in chosen:q['groups'].add(letter);chosen.add(q['id'])
    items=[]
    for e in events.values():
        questions=[]
        for q in e['questions'].values():
            if not q['groups']:continue
            q['records']=sorted(q['records'].values(),key=lambda x:x['id']);q['mechanics']=sorted(q['mechanics'])
            for record in q['records']:record['associations'].sort(key=lambda a:json.dumps(a,sort_keys=True))
            q['times']=sorted(q['times']);q['groups']=sorted(q['groups']);q['automatic_records_hash']=identity(q['records'])
            q['scorer_versions']=sorted({a['metric_version'] for r in q['records'] for a in r['associations']})
            q['focus_timestep']=min((int(f.get('timestep',r['automatic'].get('milestone',r['automatic'].get('timestep',1))))
                for r in q['records'] for f in r['automatic'].get('facts',[]) if f['status'] in ('fail','unknown')),default=min(q['times']))
            questions.append(q);e['groups'].update(q['groups'])
        if not questions:continue
        e['questions']=sorted(questions,key=lambda q:(q['groups']==['NA'],q['focus_timestep'],q['id']))
        e['groups']=sorted(e['groups']);e['cohorts']=sorted(e['cohorts']);items.append(e)
    items.sort(key=lambda e:(e['groups']==['NA'],e['scenario'],e['engine_seed'],e['model_seed'],e['branch'],e['family']))
    return items,trajectories,sources,unknown_records


def target_at(question,t,motion,reading):
    for record in question['records']:
        a=record['automatic']
        for f in a.get('facts',[]):
            if int(f.get('timestep',a.get('milestone',a.get('timestep',1))))==t and f.get('object_evidence'):return f['object_evidence']
        for o in a.get('observations',[]):
            if o['timestep']==t:return o
        if a.get('object_evidence') and int(a.get('milestone',a.get('timestep',1)))==t:return a['object_evidence']
    return None


def export_review(out,results=None,runs_root=Path('mechanics_eval/runs'),fail_limit=4,pass_limit=2,smoke_limit=None):
    out=Path(out)
    if out.exists():raise ValueError('Use a new human-review output directory')
    if smoke_limit is not None and smoke_limit<1:raise ValueError('smoke-limit must be positive')
    paths=results or [Path(runs_root)/p for p in CURRENT]
    items,trajectories,sources,unknown_records=build_items(paths,fail_limit,pass_limit)
    full_count=len(items)
    if smoke_limit is not None:
        selected=[]
        for group in ('departure_eligibility','object.crafting_table','movement','inventory_accounting','object.harvested_resource','facing_direction'):
            candidate=next((i for i in items if i['family']==group and i['groups']!=['NA'] and
                (group!='departure_eligibility' or any(r['automatic']['status']=='REJECTED_BLOCKED_ROUTE'
                    for q in i['questions'] for r in q['records']))),None)
            if candidate and len(selected)<smoke_limit:selected.append(candidate)
        for status in ('unknown','known'):
            candidate=next((i for i in items if i not in selected and any(
                r['automatic'].get('object_evidence',{}).get('localization_status')==status
                for q in i['questions'] for r in q['records'])),None)
            if candidate and len(selected)<smoke_limit:selected.append(candidate)
        items=(selected+[i for i in items if i not in selected])[:smoke_limit]
    out.mkdir(parents=True);(out/'assets').mkdir();exported={}
    for item in items:
        ph=item['evidence_hash'];tr=trajectories[ph];a=tr['arrays'];b=tr['branch'];n=len(a[f'b{b}_reference'])
        if ph not in exported:
            folder=out/'assets'/ph;folder.mkdir();frames=[]
            for t in range(n+1):
                g=a['context'][-1] if t==0 else a[f'b{b}_generated'][t-1]
                r=a['context'][-1] if t==0 else a[f'b{b}_reference'][t-1]
                gp=folder/f'generated-{t}.png';rp=folder/f'reference-{t}.png';Image.fromarray(g).save(gp);Image.fromarray(r).save(rp)
                frames.append(dict(timestep=t,generated=gp.relative_to(out).as_posix(),reference=rp.relative_to(out).as_posix(),
                    action=0 if t==0 else int(a[f'b{b}_actions'][t-1]),action_name='NOOP incoming marker' if t==0 else ACTION_NAMES[int(a[f'b{b}_actions'][t-1])],
                    real_terminal=bool(t and a[f'b{b}_done_reference'][t-1]),motion=tr['motion'][t]))
            exported[ph]=dict(frames=frames,context_length=len(a['context']),evidence_hash=ph,
                source_trajectory_sha256=tr['meta']['trajectory_sha256'],source=str(tr['evidence'].resolve()),
                omitted_post_terminal_padding=len(a[f'b{b}_generated'])-n)
        item['trajectory']=exported[ph]
        if item['family']=='departure_eligibility':
            item['trajectory']=dict(exported[ph],frames=exported[ph]['frames'][:8],review_interval=[0,7],
                restriction='Departure-only review. Return/persistence frames and scores are excluded from this item.')
        for q in item['questions']:
            q['visuals']={};fields=set()
            q['blocker_visuals']=[]
            for record in q['records']:
                for witness in record['automatic'].get('eligibility',{}).get('witnesses',[]):
                    history=[]
                    for observation in witness['visibility_history']:
                        t=observation['timestep'];row,col=observation['screen_coordinate']
                        frame=a['context'][-1] if t==0 else a[f'b{b}_generated'][t-1]
                        name=f'{q["id"]}-blocker-{t}-{row}-{col}.png'
                        Image.fromarray(frame[row*7:row*7+7,col*7:col*7+7]).save(out/'assets'/ph/name)
                        history.append(dict(timestep=t,screen_coordinate=[row,col],crop=f'assets/{ph}/{name}'))
                    q['blocker_visuals'].append(dict(world_relative_coordinate=witness['world_relative_coordinate'],
                        collision_timestep=witness['timestep'],measured_step=witness['measured_step'],history=history))
            for r in q['records']:
                for f in r['automatic'].get('facts',[]):
                    field=f['field'].removesuffix('_created')
                    if field in HUD:fields.add(field)
            # An object-relative trace anchors at placement/harvest, independently of the tested object.
            for t in range(len(item['trajectory']['frames'])):
                frame=a['context'][-1] if t==0 else a[f'b{b}_generated'][t-1];v={'hud':{}}
                for field in sorted(fields):
                    patch=hud_patch(frame,field);name=f'{q["id"]}-hud-{field}-{t}.png';Image.fromarray(patch).save(out/'assets'/ph/name)
                    v['hud'][field]=dict(path=f'assets/{ph}/{name}',reader=tr['readings'][t].get('measurements',{}).get(field),value=tr['readings'][t].get(field))
                target=target_at(q,t,tr['motion'][t],tr['readings'][t])
                if target:
                    v['target']=target;point=target.get('coordinate',target.get('target_coordinate'))
                    # Failed localization never borrows a nominal/reference coordinate.
                    if point is not None and target.get('target_visibility')=='visible' and target.get('localization_status')=='known':
                        row,col=point
                        for suffix,r0,c0,r1,c1 in [('target',row,col,row+1,col+1),('surround',max(0,row-1),max(0,col-1),min(7,row+2),min(9,col+2))]:
                            name=f'{q["id"]}-{suffix}-{t}.png';Image.fromarray(frame[r0*7:r1*7,c0*7:c1*7]).save(out/'assets'/ph/name);v[suffix+'_crop']=f'assets/{ph}/{name}'
                # Explicit screen-space facts (HUD/tile creation) require no camera inference.
                if target is None:
                    for r in q['records']:
                        for f in r['automatic'].get('facts',[]):
                            if f['field'].startswith('tile:') and ',' in f['field'] and int(f.get('timestep',r['automatic'].get('milestone',1)))==t:
                                row,col=map(int,f['field'][5:].split(','));v['screen_coordinate']=[row,col]
                                name=f'{q["id"]}-screen-{t}.png';Image.fromarray(frame[row*7:row*7+7,col*7:col*7+7]).save(out/'assets'/ph/name);v['target_crop']=f'assets/{ph}/{name}'
                if fields or target or v.get('target_crop'):q['visuals'][str(t)]=v
    groups={g:sum(g in i['groups'] for i in items) for g in ('A','B','C','D','NA')}
    manifest=dict(schema_version=SCHEMA,kind='craftax_level1_human_review',items=items,sources=sources,
        groups=groups,total_unique_items=len(items),unique_questions=sum(len(i['questions']) for i in items),
        counts_semantics='Groups overlap. Related questions share a visual item; aliases are not independent annotations. NA is optional and hidden by default.',
        full_export=smoke_limit is None,full_item_count=full_count,sampling=dict(other_failures_per_category=fail_limit,passes_per_category=pass_limit),
        unknown_source_records=len(unknown_records),unknown_question_ids=sorted({r[-1] for r in unknown_records}),
        annotations_status='No human labels supplied; viewing does not verify a case.')
    manifest['manifest_id']=identity([(i['id'],[(q['id'],q['automatic_records_hash']) for q in i['questions']]) for i in items])
    dump_json(out/'review_manifest.json',manifest)
    template=Path(__file__).with_name('human_review.html').read_text()
    payload=json.dumps(manifest,separators=(',',':'),allow_nan=False).replace('<','\\u003c')
    (out/'index.html').write_text(template.replace('/*__MANIFEST__*/',payload))
    (out/'README.md').write_text(README)
    dump_json(out/'labels-template.json',dict(schema_version=SCHEMA,manifest_id=manifest['manifest_id'],labels=[]))
    return manifest


def validate_labels(manifest,document):
    if not isinstance(document,dict):raise ValueError('Human labels must be a JSON object')
    if document.get('schema_version')!=SCHEMA:raise ValueError('Unsupported human-label schema_version')
    if document.get('manifest_id')!=manifest['manifest_id']:raise ValueError('Labels belong to a different review manifest')
    if not isinstance(document.get('labels'),list):raise ValueError('labels must be an array')
    questions={(i['id'],q['id']):(i,q) for i in manifest['items'] for q in i['questions']};seen=set();valid=[]
    errors=[]
    for index,label in enumerate(document.get('labels',[])):
        if not isinstance(label,dict):raise ValueError(f'Label {index}: must be an object')
        key=(label.get('item_id'),label.get('question_id'))
        reason=None
        if key not in questions:reason='unknown review item/question ID'
        elif key in seen:reason='duplicate item/question label'
        else:
            i,q=questions[key]
            if label.get('evidence_hash')!=i['evidence_hash']:reason='evidence hash mismatch'
            elif label.get('automatic_records_hash')!=q['automatic_records_hash'] or label.get('scorer_versions')!=q['scorer_versions']:reason='automatic records/scorer version mismatch'
            elif label.get('decision') not in DECISIONS:reason='invalid human decision'
            elif label.get('original_automatic')!={r['id']:r['automatic']['status'] for r in q['records']}:reason='original automatic verdict mismatch'
            elif not isinstance(label.get('reviewer'),str) or not label['reviewer'].strip():reason='reviewer identifier required'
            elif not isinstance(label.get('comment',''),str):reason='comment must be text'
            elif label.get('observation_agreement','not_assessed') not in ('agree','disagree','unreadable','not_assessed'):reason='invalid observation agreement'
            elif not isinstance(label.get('human_verdicts'),dict) or any(k not in {r['id'] for r in q['records']} or v not in STATUSES for k,v in label['human_verdicts'].items()):reason='invalid policy record/verdict'
        if reason:errors.append(f'Label {index}: {reason}')
        else:seen.add(key);valid.append(label)
    if errors:raise ValueError('Incompatible labels; no import performed:\n'+'\n'.join(errors))
    return valid


def summarize_review(manifest_path,labels_path,out):
    manifest=json.loads(Path(manifest_path).read_text());labels=json.loads(Path(labels_path).read_text());valid=validate_labels(manifest,labels)
    by_key={(l['item_id'],l['question_id']):l for l in valid};rows=[];confusion=Counter();agreement=Counter();policies=defaultdict(list)
    policy_totals=defaultdict(set)
    confirmed=Counter();reader_errors=Counter();unknown=Counter();rejections=Counter();returned=[];unresolved=[]
    rejection_decisions=Counter()
    for i in manifest['items']:
        for q in i['questions']:
            label=by_key.get((i['id'],q['id']));statuses={r['automatic']['status'] for r in q['records']}
            for record in q['records']:
                for association in record['associations']:
                    policy_totals[(association['cohort'],'required' if association['required'] else 'diagnostic')].add(q['id'])
            if 'REJECTED_BLOCKED_ROUTE' in statuses:
                if not label:rejection_decisions['unreviewed']+=1
                elif label.get('human_observed')=='invalid_blocker':rejection_decisions['overturned']+=1
                elif label.get('human_observed')=='legitimate_blocker' or label['decision']=='CONFIRM_AUTOMATIC':rejection_decisions['confirmed']+=1
                else:rejection_decisions['uncertain_or_unresolved']+=1
            row=dict(item_id=i['id'],question_id=q['id'],scenario=i['scenario'],category=i['category'],mechanics='; '.join(q['mechanics']),
                engine_seed=i['engine_seed'],model_seed=i['model_seed'],branch=i['branch_label'],automatic_statuses='; '.join(sorted(statuses)),
                decision=label['decision'] if label else 'UNREVIEWED',human_observed=label.get('human_observed') if label else None,
                comment=label.get('comment','') if label else '',human_verdicts=label.get('human_verdicts',{}) if label else {})
            rows.append(row)
            if label:
                agreement[label.get('observation_agreement','not_assessed')]+=1
                if label['decision']=='READER_ERROR':reader_errors[i['category']+' / '+', '.join(q['mechanics'])]+=1
                if 'unknown' in statuses:unknown[label['decision']]+=1
                if label['decision']=='CONFIRM_AUTOMATIC':
                    # One physical question per status, never one count per measurement alias.
                    for status in statuses & {'pass','fail','unknown'}:confirmed[status]+=1
                distinct_pairs=set()
                for r in q['records']:
                    auto=r['automatic']['status'];human=label.get('human_verdicts',{}).get(r['id'])
                    if human is None and label['decision']=='CONFIRM_AUTOMATIC' and auto in STATUSES:human=auto
                    if human and label['decision'] not in ('GENUINELY_UNOBSERVABLE','NEEDS_FURTHER_REVIEW'):
                        distinct_pairs.add((auto,human))
                        for a in r['associations']:
                            policies[(a['cohort'],'required' if a['required'] else 'diagnostic')].append((q['id'],auto,human))
                # Incompatible policies are shown separately; do not pick one arbitrary verdict.
                if len(distinct_pairs)==1:confusion[next(iter(distinct_pairs))]+=1
                elif len(distinct_pairs)>1:confusion['MULTIPLE_POLICIES','SEPARATE']+=1
                if i['family']=='departure_eligibility':
                    value=label.get('human_observed')
                    if value in ('legitimate_blocker','invalid_blocker','uncertain'):
                        prefix='rejected' if 'REJECTED_BLOCKED_ROUTE' in statuses else 'retained_ambiguous'
                        rejections[prefix+':'+value]+=1
                    else:rejections['no_explicit_blocker_label']+=1
            if not label or label['decision']!='CONFIRM_AUTOMATIC':
                unresolved.append(dict(item_id=i['id'],question_id=q['id'],reason=row['decision']))
            for r in q['records']:
                if r['automatic'].get('name')=='object_persistence.after_return' and r['automatic']['status']=='fail' and any(a['cohort']=='CONDITIONAL_SELECTED_ELIGIBLE' for a in r['associations']):
                    returned.append(dict(item_id=i['id'],question_id=q['id'],engine_seed=i['engine_seed'],model_seed=i['model_seed'],human_conclusion=row['decision'],human_observed=row['human_observed']))
                    break
    reviewed_items=sum(all((i['id'],q['id']) in by_key for q in ([q for q in i['questions'] if q['groups']!=['NA']] or i['questions'])) for i in manifest['items'])
    touched=len({l['item_id'] for l in valid});binary=sum(v for (a,h),v in confusion.items() if a in ('pass','fail') and h in ('pass','fail'))
    fp=confusion['pass','fail'];fn=confusion['fail','pass']
    human_fail=sum(v for (a,h),v in confusion.items() if a in ('pass','fail') and h=='fail');human_pass=sum(v for (a,h),v in confusion.items() if a in ('pass','fail') and h=='pass')
    policy_stats=[]
    for cohort,role in sorted(policy_totals):
        entries=policies[cohort,role]
        entries=sorted(set(entries));by_question=defaultdict(set)
        for q,a,h in entries:by_question[q].add((a,h))
        pairs=[next(iter(v)) for v in by_question.values() if len(v)==1]
        policy_stats.append(dict(cohort=cohort,role=role,total_review_questions=len(policy_totals[cohort,role]),unique_questions=len(by_question),
            comparable_single_policy_questions=len(pairs),verdict_agreements=sum(a==h for a,h in pairs),
            false_positive_passes=sum(a=='pass' and h=='fail' for a,h in pairs),false_negative_fails=sum(a=='fail' and h=='pass' for a,h in pairs),
            unique_question_verdicts=[dict(question_id=q,automatic=a,human=h) for q,a,h in entries]))
    stats=dict(schema_version=SCHEMA,manifest_id=manifest['manifest_id'],total_unique_items=len(manifest['items']),
        main_review_items=sum(any(g!='NA' for g in i['groups']) for i in manifest['items']),
        optional_only_items=sum(i['groups']==['NA'] for i in manifest['items']),
        fully_reviewed_items=reviewed_items,partially_reviewed_items=touched-reviewed_items,untouched_items=len(manifest['items'])-touched,
        unique_questions=sum(len(i['questions']) for i in manifest['items']),reviewed_questions=len(valid),unreviewed_questions=len(rows)-len(valid),
        group_counts=manifest['groups'],confirmed_automatic=dict(confirmed),observation_agreement=dict(agreement),
        category_progress={category:dict(total=sum(i['category']==category for i in manifest['items']),
            touched=sum(i['category']==category and any((i['id'],q['id']) in by_key for q in i['questions']) for i in manifest['items']))
            for category in sorted({i['category'] for i in manifest['items']})},
        observation_agreement_denominator=agreement['agree']+agreement['disagree'],reader_errors_by_mechanic=dict(reader_errors),unknown_decisions=dict(unknown),
        physical_question_verdict_confusion=[dict(automatic=a,human=h,count=v) for (a,h),v in confusion.items()],
        binary_verdict_denominator=binary,false_positive_passes=fp,false_negative_fails=fn,
        false_positive_rate=fp/human_fail if human_fail else None,false_negative_rate=fn/human_pass if human_pass else None,
        rejection_judgments=dict(rejections),rejection_decisions=dict(rejection_decisions),returned_table_failures=returned,unresolved=unresolved,
        policies=policy_stats,
        limitations=['Selected review subset only; no accuracy claim for unreviewed frames.',
            'Group counts overlap; item completion requires all non-optional questions. NA is optional.',
            'Agreement requires explicit agree/disagree; verdict errors require explicit human verdict or CONFIRM_AUTOMATIC.',
            'PASS is the positive verdict. Binary confusion excludes UNKNOWN/NOT_TESTED/NA and multiple-policy questions.',
            'Policy strata overlap physically; never add cohort/required/diagnostic denominators.',
            'Human annotations do not modify automatic scores.'])
    out=Path(out);out.mkdir(parents=True,exist_ok=False);dump_json(out/'report.json',stats)
    with (out/'questions.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['item_id'],lineterminator='\n');writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)
    with (out/'statistics.csv').open('w') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(['metric','value'])
        writer.writerows((k,v) for k,v in stats.items() if isinstance(v,(int,float)) or v is None)
        for name in ('confirmed_automatic','observation_agreement','unknown_decisions','rejection_decisions','reader_errors_by_mechanic'):
            writer.writerows((name+'.'+k,v) for k,v in stats[name].items())
    with (out/'policies.csv').open('w') as f:
        fields=['cohort','role','total_review_questions','unique_questions','comparable_single_policy_questions','verdict_agreements','false_positive_passes','false_negative_fails']
        writer=csv.DictWriter(f,fields,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(policy_stats)
    lines=['# Level 1 human review','',f'{len(manifest["items"])} unique visual items; {reviewed_items} fully reviewed, {touched-reviewed_items} partially reviewed, {len(manifest["items"])-touched} untouched.',
        f'{len(valid)} / {len(rows)} physical questions annotated. Viewing alone does not annotate.',
        '',f'Explicit observation agreement: {agreement["agree"]} agree / {agreement["disagree"]} disagree; {agreement["unreadable"]} unreadable; {agreement["not_assessed"]} not assessed.',
        f'Confirmed automatic statuses: {dict(confirmed)}. A question with different policies may confirm more than one status.',
        f'Binary false-positive PASS: {fp}; false-negative FAIL: {fn}; eligible binary denominator: {binary}.',
        f'Reader errors by mechanic: {dict(reader_errors)}',f'UNKNOWN decisions: {dict(unknown)}',f'Rejection decisions: {dict(rejection_decisions)}; explicit blocker judgments: {dict(rejections)}',
        '', '## Six conditional after-return table failures','', '| Engine seed | Model seed | Human conclusion | Observed |','|---|---|---|---|']
    lines.extend(f'| {r["engine_seed"]} | {r["model_seed"]} | {r["human_conclusion"]} | {r["human_observed"]} |' for r in returned)
    lines.extend(['','## Cohort and required/diagnostic policy strata (overlap)','',
        '| Cohort | Role | Review subset total | Questions with verdicts | Comparable questions | Agreements | False PASS | False FAIL |',
        '|---|---|---:|---:|---:|---:|---:|---:|'])
    lines.extend(f'| {p["cohort"]} | {p["role"]} | {p["total_review_questions"]} | {p["unique_questions"]} | {p["comparable_single_policy_questions"]} | {p["verdict_agreements"]} | {p["false_positive_passes"]} | {p["false_negative_fails"]} |' for p in policy_stats)
    lines.extend(['','## Limits','']+['- '+s for s in stats['limitations']]+['','Detailed unresolved questions and overlapping policy strata are in report.json; every physical question is in questions.csv.'])
    text='\n'.join(lines)+'\n';(out/'report.md').write_text(text)
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Human review summary</title><style>body{font:16px system-ui;max-width:1100px;margin:32px auto}pre{white-space:pre-wrap}</style><pre>'+html.escape(text)+'</pre>')
    return stats


README = '''# Craftax Level 1 human review

Open `index.html` in a desktop browser. No server, internet, Python background process or external
libraries are needed. All images are lossless PNGs exported from saved RGB, at their original resolution.

1. Enter your reviewer identifier. Group A contains every current UNKNOWN, B prerequisite-blocked
   cases, C required failures plus stratified failures, and D positive/rejection controls. Groups overlap.
   NOT_APPLICABLE is hidden by default; select group NA to audit it. Filters include reviewed/partial/
   unreviewed/flagged, status, category, branch, cohort and search.
2. Each navigation item groups related physical events in one recording. Choose each related question
   to annotate its own mechanic. Measurement aliases and unconditional/conditional policy variants remain
   associated with that question. Fully reviewed means all non-optional questions have saved labels.
3. Blind mode shows generated RGB and the rule, hiding automatic observations/verdicts and reference
   futures until you save an initial label. Diagnostic mode exposes them immediately and is recorded.
   Judge local physics in generated terrain; reference futures are optional, not a pixel target.
4. Use ←/→ for frames, Space for playback, [ / ] for visual items. Avoid shortcuts while typing.
   Click timeline actions or milestone buttons; zoom preserves pixels. Turn on landmark overlays only
   in diagnostic/revealed view. Unknown localization has no fabricated target crop.
5. Set a decision, optionally enter an observed fact, an explicit observation-agreement judgment,
   comments and policy-specific human verdicts. Save explicitly. Simply displaying a case never labels it.
   A genuine unknown needs no number or Boolean. Rejection review is restricted to departure frames;
   enter legitimate_blocker / invalid_blocker / uncertain without consulting later persistence outcomes.
6. Export labels JSON often: it is the authoritative saved artifact. Browser storage is only a best-effort
   convenience on file://. Import JSON to resume on another browser/session. Imports validate all IDs,
   evidence hashes, automatic-record hashes and scorer versions atomically; incompatibilities are reported.
   Import replaces current progress only after explicit confirmation. Keep your original JSON backup.

`labels-template.json` contains an empty version-1 document, never synthetic human labels. Each saved
label records item/question IDs, evidence/automatic hashes, scorer versions, reviewer, decision,
human-observed fact, comment, optional explicit policy verdicts, timestamp, mode and initial blind label.
Automatic scores are immutable. Groups and policy strata overlap; do not add their counts.

After reviewing, run from the repository root (replace the labels path and use a fresh report directory):

```
python -m mechanics_eval.cli summarize-human-review \\
  --manifest mechanics_eval/runs/human-review-level1/review_manifest.json \\
  --labels /absolute/path/to/human_labels.json \\
  --out mechanics_eval/runs/human-review-level1/report
```

The report writes Markdown, HTML, JSON and CSV, separates required/diagnostic and cohort policies,
and leaves unreviewed observations explicitly unreviewed. It does not rescore the benchmark.
'''
