"""Offline pixel-art trajectory comparisons; verdicts come only from mechanics scores."""
import html
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
import multiprocessing
import os
from pathlib import Path
import shutil
import time

import numpy as np
from .evidence import load_case, dump_json, digest
from .gif_render import ACTION_NAMES, comparison_frames, render_case_gifs as _render_case_gifs


def suite_design(rows, name):
    if any(not {'scenario','seed','model_sample'} <= r.keys() for r in rows):
        return '<p>'+html.escape(name)+': scenario/seed design unavailable in legacy results.</p>'
    scenarios = sorted({r['scenario'] for r in rows})
    seeds = sorted({r['seed'] for r in rows})
    samples = sorted({r['model_sample'] for r in rows})
    balanced = len(rows) == len(scenarios)*len(seeds)*len(samples) == len({
        (r['scenario'],r['seed'],r['model_sample']) for r in rows})
    equation = (f'{len(scenarios)} scenario types × {len(seeds)} engine seeds × '
                f'{len(samples)} model sample(s) = {len(rows)} attempts') if balanced else f'{len(rows)} attempts (uneven sampling)'
    return ('<p><strong>'+html.escape(name)+': '+equation+'</strong>; '
            f'{2*len(rows)} paired branch recordings. Engine seeds: '+html.escape(str(seeds))+'.</p>'
            '<details><summary>Scenario types and static/dynamic split</summary><p>'+html.escape(', '.join(scenarios))+
            '</p><p>'+', '.join(f'{scope}: {len({r["scenario"] for r in rows if r.get("scope")==scope})} types / '
                              f'{sum(r.get("scope")==scope for r in rows)} attempts' for scope in ('static','dynamic'))+'</p></details>')


GIF_MODES = ('all','fail','unknown','nonpass','none')


def gif_enabled(mode,status):
    return mode=='all' or mode==status or (mode=='nonpass' and status in ('fail','unknown'))



def export_gifs(evidence_dirs, results_files, out, gifs='all', gif_workers=4):
    """Export every result, selectively rendering media from saved RGB only."""
    if gifs not in GIF_MODES:raise ValueError('Unknown GIF setting: '+str(gifs))
    if gif_workers<1:raise ValueError('gif-workers must be positive')
    if len(evidence_dirs)!=len(results_files):raise ValueError('Provide one results JSON per evidence directory')
    if Path(out).exists():raise ValueError('Use a new report directory')
    started=time.perf_counter()
    cases=sum(len(json.loads((Path(p)/'manifest.json').read_text())['cases']) for p in evidence_dirs)
    workers=min(gif_workers,os.cpu_count() or 1,max(1,cases)) if gifs!='none' else 0
    # Never fork a process holding a JAX runtime or loaded checkpoint.
    executor=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) if workers>1 else nullcontext(None)
    print(f'[gallery] exporting {cases} cases | GIFs {gifs} | workers {workers}',flush=True)
    with executor as pool:
        target,count=_export_gifs(evidence_dirs,results_files,out,gifs,pool)
    elapsed=round(time.perf_counter()-started,3)
    dump_json(Path(out)/'export.json',dict(elapsed_s=elapsed,gif_mode=gifs,gif_workers=workers,gifs_written=count,cases=cases))
    print(f'[gallery] complete | {count} GIFs | elapsed {elapsed:.1f}s',flush=True)
    return target


def _export_gifs(evidence_dirs, results_files, out, gifs, pool):
    from .gallery import compact_stage, compact_verdict, compact_selection, summary, write_page
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    data=dict(metric_version='saved mechanics scores',cards=[],cohorts=[],downloads={'Gallery data':'gallery.json','Export timings':'export.json'},
              source_note='Local full export. Results, eligibility and denominators are independent of the GIF setting.')
    index=[];pending=[];gif_count=0
    for i,(evidence,results_file) in enumerate(zip(map(Path,evidence_dirs),map(Path,results_files))):
        report=json.loads(results_file.read_text());manifest=json.loads((evidence/'manifest.json').read_text())
        if report.get('evidence_manifest_sha256') and report['evidence_manifest_sha256']!=digest(evidence/'manifest.json'):
            raise ValueError('Results do not match this evidence manifest')
        data['metric_version']=report.get('metric_version',data['metric_version'])
        suite_name=manifest.get('config',{}).get('suite',f'suite{i}')
        names={r.get('scenario') for r in report['samples']}
        label={'existing':'Existing diagnostic','focused':'Focused mechanics'}.get(suite_name,suite_name)
        if names=={'move_free','move_blocked','turn_loop','noop','table_leave_return'}:label='Controlled movement'
        elif names=={'place_table'}:label='Table placement'
        rows={Path(r['evidence']).name:r for r in report['samples']}
        suite_dir=out/f'suite{i}';suite_dir.mkdir()
        dump_json(suite_dir/'results.json',report,compact=True);shutil.copy2(evidence/'manifest.json',suite_dir/'manifest.json')
        data['downloads'][f'{label} results JSON']=f'suite{i}/results.json'
        conditional=report.get('benchmark_cohorts')
        if conditional:
            if data.get('conditional'):raise ValueError('Export one conditional experiment per gallery')
            data['conditional']=conditional
            subsets={'first':[r for r in rows.values() if r.get('conditional_attempt',{}).get('unconditional_first_sample')],
                     'selected':[r for r in rows.values() if r.get('conditional_attempt',{}).get('selected_conditional')],
                     'rejected':[r for r in rows.values() if r.get('conditional_attempt',{}).get('decision')=='REJECTED_BLOCKED_ROUTE'],
                     'attempts':list(rows.values())}
            for key,label in [('first','Unconditional first samples'),('selected','Conditional selected eligible'),('rejected','Rejected blocked routes · inspection only'),('attempts','All retry attempts · inspection only')]:
                data['cohorts'].append(dict(id=key,label=label,summary=summary(subsets[key],ordinary=key in ('first','rejected','attempts')),
                    description='Selected-action attempts only. Retry/inspection views are not ordinary benchmark success estimates. Full-route fidelity is required only in the conditional selected cohort.'))
            for filename in ('unconditional-results.json','conditional-results.json','summary.json'):
                dump_json(suite_dir/filename,json.loads((results_file.parent/filename).read_text()),compact=True)
        else:
            cid=f'ordinary:{i}'
            data['cohorts'].append(dict(id=cid,label=f'{label} · {len(rows)} attempts',summary=summary(list(rows.values())),
                description=f'{len(rows)} ordinary attempts in this corpus; one selected branch each. Alternative branches, supplements and retries are not added. '+
                            f'{len({r.get("scenario",Path(r["evidence"]).name) for r in rows.values()})} scenario types. '+
                            (f'{len({r["seed"] for r in rows.values()})} engine seeds; see each scenario for draws per seed.' if all('seed' in r for r in rows.values()) else 'Seed design unavailable in legacy result metadata.')))
        for name in manifest['cases']:
            meta,arrays=load_case(evidence/name);row=rows[name]
            if 'branch_outcomes' not in row:raise ValueError('Score independent branches before export')
            case_dir=suite_dir/name;case_dir.mkdir()
            requests=[]
            for filename in ('metadata.json','sampling.json','trajectory.npz'):shutil.copy2(evidence/name/filename,case_dir/filename)
            dump_json(case_dir/'result.json',row,compact=True)
            relative=case_dir.relative_to(out).as_posix()
            attempt=row.get('conditional_attempt');cohorts=[f'ordinary:{i}'] if not attempt else ['attempts']
            if attempt:
                if attempt['unconditional_first_sample']:cohorts.append('first')
                if attempt['selected_conditional']:cohorts.append('selected')
                if attempt['decision']=='REJECTED_BLOCKED_ROUTE':cohorts.append('rejected')
                shutil.copy2(evidence/name/'eligibility.json',case_dir/'eligibility.json')
            for branch in range(2):
                label='selected' if branch==0 else ('noop' if not np.any(arrays['b1_actions']) else 'alternative')
                primary=row['branch_outcomes'][branch]
                card=dict(case=name,scenario=row.get('scenario',meta['spec']['name']),category=row.get('category',meta['category']),
                    scope=row.get('scope','dynamic' if meta['spec']['zombie'] else 'static'),branch=label,
                    seed=meta['seed'],model_seed=meta['model_seed'],actions=arrays[f'b{branch}_actions'].tolist(),context_length=len(arrays['context']),
                    valid_steps=len(arrays[f'b{branch}_reference']),generated_steps=len(arrays[f'b{branch}_generated']),cohorts=list(cohorts),atomic_status=primary['status'],
                    counterfactual_status=row['counterfactual']['status'],counterfactual=compact_verdict(row['counterfactual']),views={},
                    links={'Mechanics JSON':relative+'/result.json','Metadata':relative+'/metadata.json','Saved RGB':relative+'/trajectory.npz'})
                if attempt:
                    card['links']['Departure-only eligibility proof']=relative+'/eligibility.json'
                    card['selection']=compact_selection(attempt,json.loads((evidence/name/'eligibility.json').read_text()))
                for view in (['current','ordinary'] if attempt else ['current']):
                    scenario=row.get('ordinary_scenario_results_by_branch' if view=='ordinary' else 'scenario_results_by_branch',row['branch_outcomes'])[branch]
                    stages=row.get('ordinary_measurements_by_branch' if view=='ordinary' else 'measurements_by_branch',[[],[]])[branch]
                    status=scenario['status'];failure=scenario.get('first_failure')
                    gif=None
                    if view=='ordinary' and not attempt['selected_conditional'] and card['views']['current']['status']==status:
                        gif=card['views']['current']['gif']
                    elif gif_enabled(gifs,status):
                        display={key:meta[key] for key in ('seed','model_seed','score_scope','ordinary_overall') if key in meta}
                        display['spec']=dict(name=meta['spec']['name']+' / '+primary.get('mechanic',meta['spec']['name']),milestone=primary.get('milestone',meta['spec']['milestone']))
                        if attempt:display['score_scope']=('unconditional first' if attempt['unconditional_first_sample'] else 'unfiltered attempt') if view=='ordinary' or not attempt['selected_conditional'] else 'conditional'
                        filename=label+('-first' if view=='ordinary' else '')+'.gif'
                        requests.append((filename,display,branch,primary['status'],status))
                        gif=relative+'/'+filename
                    card['views'][view]=dict(status=status,gif=gif,stages=[compact_stage(s) for s in stages],
                        failure=(f'{failure["stage"]} at t={failure["timestep"]}: {failure["reason"]}' if failure else None))
                data['cards'].append(card)
                index.append(dict(case=name,suite=suite_name,branch=label,status=card['views']['current']['status'],atomic_status=primary['status'],
                    selected_status=row['outcome']['status'],counterfactual_status=row['counterfactual']['status'],gif=card['views']['current']['gif'],
                    ordinary_gif=card['views'].get('ordinary',{}).get('gif'),gif_setting=gifs,cohorts=list(cohorts),
                    valid_steps=len(arrays[f'b{branch}_reference']),unscored_generated_padding_omitted=len(arrays[f'b{branch}_generated'])-len(arrays[f'b{branch}_reference'])))
            for test in row.get('consistency_tests',[]):
                branch=test['branch'];status=test['status'];rule=test['rule'];gif=None
                if gif_enabled(gifs,status):
                    display={key:meta[key] for key in ('seed','model_seed','score_scope','ordinary_overall') if key in meta}
                    display['spec']=dict(name=rule+' / '+name,milestone=len(arrays[f'b{branch}_reference']))
                    target=case_dir/f'consistency-{rule}.gif'
                    requests.append((target.name,display,branch,status,None))
                    gif=target.relative_to(out).as_posix()
                data['cards'].append(dict(case=name+' / '+rule,scenario=row.get('scenario',name),category='consistency',scope='static',branch='consistency',
                    seed=meta['seed'],model_seed=meta['model_seed'],cohorts=list(cohorts),atomic_status=status,counterfactual_status='not_applicable',
                    views={'current':dict(status=status,gif=gif,stages=[],failure=None)},links={'Mechanics JSON':relative+'/result.json'}))
                index.append(dict(case=name,suite=suite_name,branch='consistency',mechanic=rule,scope='static',status=status,gif=gif,
                    valid_steps=len(arrays[f'b{branch}_reference']),unscored_generated_padding_omitted=len(arrays[f'b{branch}_generated'])-len(arrays[f'b{branch}_reference'])))
            if requests:
                gif_count+=len(requests)
                if pool is None:_render_case_gifs(case_dir,requests)
                else:pending.append(pool.submit(_render_case_gifs,case_dir,requests))
    last_progress=time.perf_counter();completed=0
    for future in as_completed(pending):
        completed+=future.result()
        now=time.perf_counter()
        if now-last_progress>=30:
            print(f'[gallery] {completed}/{gif_count} GIFs complete',flush=True)
            last_progress=now
    if not data['cohorts']:raise ValueError('No result corpora supplied')
    data['design']=dict(contexts=sorted({c['context_length'] for c in data['cards'] if 'context_length' in c}),
        generated_steps=sorted({c['generated_steps'] for c in data['cards'] if 'generated_steps' in c}),
        engine_seeds=len({c['seed'] for c in data['cards']}))
    data['default_cohort']=next((c['id'] for c in data['cohorts'] if c['label'].startswith('Focused mechanics')),data['cohorts'][0]['id'])
    import csv
    with (out/'mechanics.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['cohort','scope','measurement_scope','kind','name','scenario','category','required','attempts','eligible','passed','failed','unknown','not_tested','not_applicable','pass_rate','decision_coverage','eligibility_coverage','engine_seed_count','model_sample_count','samples_per_seed_min','samples_per_seed_max','instances','applicable'])
        writer.writeheader()
        for cohort in data['cohorts']:
            for scope,score in cohort['summary'].items():
                frequency=score.get('mechanics',{})
                for kind,entries in (('scenario',frequency.get('scenarios',[])),('atomic',frequency.get('atomic',[]))):
                    for row in entries:
                        entry=dict(row,cohort=cohort['id'],scope=scope,measurement_scope=row.get('scope'),kind=kind)
                        writer.writerow({key:val for key,val in entry.items() if key in writer.fieldnames})
    data['downloads']['Mechanic frequencies CSV']='mechanics.csv'
    dump_json(out/'index.json',index);write_page(out,data)
    return out/'index.html',gif_count
