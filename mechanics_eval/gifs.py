"""Offline pixel-art trajectory comparisons; verdicts come only from mechanics scores."""
import html
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
from PIL import Image, ImageDraw
from .evidence import load_case, dump_json, digest

ACTION_NAMES = ['NOOP','LEFT','RIGHT','UP','DOWN','DO','SLEEP','PLACE_STONE',
                'PLACE_TABLE','PLACE_FURNACE','PLACE_PLANT','MAKE_WOOD_PICKAXE',
                'MAKE_STONE_PICKAXE','MAKE_IRON_PICKAXE','MAKE_WOOD_SWORD',
                'MAKE_STONE_SWORD','MAKE_IRON_SWORD']


def comparison_frames(meta, arrays, branch, status, scale=4, overall_status=None):
    """Include the last real conditioning frame, then all valid steps through terminal.

    Model padding beyond engine termination is deliberately excluded, never shown as a reset.
    """
    real, generated = arrays[f'b{branch}_reference'], arrays[f'b{branch}_generated']
    if len(generated) < len(real):
        raise ValueError('generated rollout shorter than reference')
    start = arrays['context'][-1]
    images = []
    size = 63 * scale
    extra = 16 if overall_status is not None else 0
    for t in range(len(real) + 1):
        canvas = Image.new('RGB', (max(660, 2 * size + 24), size + 108 + extra), '#eeeeee')
        draw = ImageDraw.Draw(canvas)
        action = 'NOOP (incoming marker)' if t == 0 else ACTION_NAMES[int(arrays[f'b{branch}_actions'][t-1])]
        terminal = t > 0 and bool(arrays[f'b{branch}_done_reference'][t-1])
        draw.text((10,5), meta['spec']['name'], fill='black')
        draw.text((10,21), f"engine seed={meta['seed']}  model seed={meta['model_seed']}  context={len(arrays['context'])}", fill='black')
        draw.text((10,37), f"t={t} action={action}  {'SELECTED' if branch == 0 else 'ALTERNATIVE'} branch", fill='black')
        draw.text((10,53), f"Branch mechanics at t={meta['spec']['milestone']}: {status.upper()}" + (' | REAL TERMINAL; stop here' if terminal else ''), fill='black')
        if overall_status is not None:
            label='Scenario overall'+(' ['+meta['score_scope']+']' if meta.get('score_scope') else '')+': '+overall_status.upper()
            if meta.get('ordinary_overall') is not None:label+=' | unconditional first: '+meta['ordinary_overall'].upper()
            draw.text((10,69),label,fill='black')
        for x, title, frame in [(8,'Real Craftax',start if t == 0 else real[t-1]),
                                 (size+16,'World model',start if t == 0 else generated[t-1])]:
            draw.text((x,75+extra),title,fill='black')
            canvas.paste(Image.fromarray(np.asarray(frame)).resize((size,size),Image.Resampling.NEAREST),(x,96+extra))
        images.append(canvas)
    return images



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


def export_gifs(evidence_dirs, results_files, out, gifs='all'):
    """Export every result, selectively rendering media from saved RGB only."""
    from .gallery import compact_stage, compact_verdict, compact_selection, summary, write_page
    if gifs not in GIF_MODES:raise ValueError('Unknown GIF setting: '+str(gifs))
    if len(evidence_dirs)!=len(results_files):raise ValueError('Provide one results JSON per evidence directory')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    data=dict(metric_version='saved mechanics scores',cards=[],cohorts=[],downloads={'All evidence (ZIP)':'evidence.zip','Gallery data':'gallery.json'},
              source_note='Local full export. Results, eligibility and denominators are independent of the GIF setting.')
    index=[]
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
        shutil.copy2(results_file,suite_dir/'results.json');shutil.copy2(evidence/'manifest.json',suite_dir/'manifest.json')
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
                shutil.copy2(results_file.parent/filename,suite_dir/filename)
        else:
            cid=f'ordinary:{i}'
            data['cohorts'].append(dict(id=cid,label=f'{label} · {len(rows)} attempts',summary=summary(list(rows.values())),
                description=f'{len(rows)} ordinary attempts in this corpus; one selected branch each. Alternative branches, supplements and retries are not added. '+
                            f'{len({r.get("scenario",Path(r["evidence"]).name) for r in rows.values()})} scenario types. '+
                            (f'{len({r["seed"] for r in rows.values()})} engine seeds; {len({r.get("model_sample",0) for r in rows.values()})} model sample(s) per state.' if all('seed' in r for r in rows.values()) else 'Seed design unavailable in legacy result metadata.')))
        for name in manifest['cases']:
            meta,arrays=load_case(evidence/name);row=rows[name]
            if 'branch_outcomes' not in row:raise ValueError('Score independent branches before export')
            case_dir=suite_dir/name;case_dir.mkdir()
            for filename in ('metadata.json','sampling.json','trajectory.npz'):shutil.copy2(evidence/name/filename,case_dir/filename)
            dump_json(case_dir/'result.json',row)
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
                        display=dict(meta,spec=dict(meta['spec'],name=meta['spec']['name']+' / '+primary.get('mechanic',meta['spec']['name']),milestone=primary.get('milestone',meta['spec']['milestone'])))
                        if attempt:display['score_scope']=('unconditional first' if attempt['unconditional_first_sample'] else 'unfiltered attempt') if view=='ordinary' or not attempt['selected_conditional'] else 'conditional'
                        filename=label+('-first' if view=='ordinary' else '')+'.gif'
                        frames=comparison_frames(display,arrays,branch,primary['status'],overall_status=status)
                        frames[0].save(case_dir/filename,save_all=True,append_images=frames[1:],duration=250,loop=0,optimize=False,disposal=2)
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
                    display=dict(meta,spec=dict(meta['spec'],name=rule+' / '+name,milestone=len(arrays[f'b{branch}_reference'])))
                    frames=comparison_frames(display,arrays,branch,status)
                    target=case_dir/f'consistency-{rule}.gif';frames[0].save(target,save_all=True,append_images=frames[1:],duration=250,loop=0,optimize=False,disposal=2)
                    gif=target.relative_to(out).as_posix()
                data['cards'].append(dict(case=name+' / '+rule,scenario=row.get('scenario',name),category='consistency',scope='static',branch='consistency',
                    seed=meta['seed'],model_seed=meta['model_seed'],cohorts=list(cohorts),atomic_status=status,counterfactual_status='not_applicable',
                    views={'current':dict(status=status,gif=gif,stages=[],failure=None)},links={'Mechanics JSON':relative+'/result.json'}))
                index.append(dict(case=name,suite=suite_name,branch='consistency',mechanic=rule,scope='static',status=status,gif=gif,
                    valid_steps=len(arrays[f'b{branch}_reference']),unscored_generated_padding_omitted=len(arrays[f'b{branch}_generated'])-len(arrays[f'b{branch}_reference'])))
    if not data['cohorts']:raise ValueError('No result corpora supplied')
    data['design']=dict(contexts=sorted({c['context_length'] for c in data['cards'] if 'context_length' in c}),
        generated_steps=sorted({c['generated_steps'] for c in data['cards'] if 'generated_steps' in c}),
        engine_seeds=len({c['seed'] for c in data['cards']}))
    data['default_cohort']=next((c['id'] for c in data['cohorts'] if c['label'].startswith('Focused mechanics')),data['cohorts'][0]['id'])
    dump_json(out/'index.json',index);write_page(out,data)
    with zipfile.ZipFile(out/'evidence.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob('*')):
            if path.is_file() and path.name!='evidence.zip':archive.write(path,path.relative_to(out))
    return out/'index.html'
