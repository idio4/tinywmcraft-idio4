"""Reproducible departure-only rejection sampling; ordinary and conditional cohorts stay separate."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import hashlib
import jax
import numpy as np
from .evidence import digest,dump_json,load_case,save_case
from .reader_v2 import ReaderV2
from .rollouts import ModuleAdapter,reference_pair
from .scenarios import Engine,Scenario,catalog
from .rejection import departure_eligibility,retry_selection,sampling_seed,VERSION


def cohort(report,rows,conditional):
    from .dependencies import summarize_measurements
    from .milestone import summarize
    result=deepcopy({k:v for k,v in report.items() if k not in ('samples','v1_comparison_same_rgb')})
    summary=summarize_measurements(rows)
    result.update(samples=rows,fixed_route_required=conditional,measurement_aggregate=summary,
                  scenario_aggregate=summary['overall_scenarios'],aggregate=summarize(rows) if rows else None,
                  aggregate_by_scope={'static':summarize(rows)} if rows else {},
                  reference_reader_validation_scope='all retained attempts, same independent reader',
                  dependency_aggregate_by_scope={'static':summary} if rows else {},
                  aggregate_semantics='single-primary compatibility metric; use scenario_aggregate for cohort outcomes')
    return result


def conditional_row(row,readings):
    from .dependencies import build
    meta,arrays=load_case(row['evidence']);result=deepcopy(row)
    for branch in range(2):
        n=len(arrays[f'b{branch}_reference']);trajectory=[readings['context_last']]+readings['generated'][branch][:n]
        stages,scenario=build(meta['spec'],trajectory,arrays[f'b{branch}_actions'],row['branch_outcomes'][branch],
            row['diagnostics_by_branch'][branch],meta['reference_truth'][str(branch)]['facts'],
            meta['reference_truth'][str(branch)]['initial'],conditional=True)
        for stage in stages:stage.update(scope='static',measurement_id=f'{Path(row["evidence"]).name}/b{branch}/'+stage['name'])
        result['measurements_by_branch'][branch]=stages;result['scenario_results_by_branch'][branch]=scenario
    result['scenario_result']=result['scenario_results_by_branch'][0]
    return result


def finalize(out):
    from .milestone import score_run
    out=Path(out);sampling=json.loads((out/'sampling.json').read_text())
    report=score_run(out/'evidence',out/'all-attempt-scores',out/'readings')
    rows={Path(r['evidence']).name:r for r in report['samples']};first=[];selected=[];seed_results=[]
    selected_by_case={};trial_by_case={}
    for seed in sampling['seeds']:
        first.append(rows[seed['attempts'][0]['case']])
        for trial in seed['attempts']:trial_by_case[trial['case']]=trial
        index=seed['selection']['selected_index'];entry=deepcopy(seed)
        if index is not None:
            name=seed['attempts'][index]['case'];read=json.loads((out/'readings'/name/'measurements.json').read_text())
            row=conditional_row(rows[name],read);selected.append(row);selected_by_case[name]=row
            entry['outcome']=row['scenario_result'];entry['mechanics']={s['name']:s['status'] for s in row['measurements_by_branch'][0]}
        else:entry.update(outcome=None,mechanics=None)
        seed_results.append(entry)
    ordinary=cohort(report,first,False);ordinary['cohort']='UNCONDITIONAL_FIRST_SAMPLE'
    conditional=cohort(report,selected,True);conditional['cohort']='CONDITIONAL_SELECTED_ELIGIBLE'
    exhausted=sum(s['selection']['status']=='RETRY_BUDGET_EXHAUSTED' for s in seed_results)
    summary=dict(kind='conditional_fixed_six_tile_experiment',eligibility_version=VERSION,
        selection_rule='first attempt not positively rejected; no mechanics outcome is consulted',
        environment_seeds=len(seed_results),max_samples_per_seed=sampling['max_samples_per_seed'],
        total_generated_attempts=sum(len(s['attempts']) for s in seed_results),
        first_samples_reused=len(seed_results),newly_generated_attempts=sum(len(s['attempts'])-1 for s in seed_results),
        rejected_blocked_routes=sum(t['decision']=='REJECTED_BLOCKED_ROUTE' for s in seed_results for t in s['attempts']),
        selected_eligible=len(selected),retry_budget_exhausted=exhausted,
        selected_ambiguous_departures=sum(s['selection']['selected_index'] is not None and
            s['attempts'][s['selection']['selected_index']]['reason']=='preserve_ambiguous_departure' for s in seed_results),
        selected_preserved_failures=sum(s['selection']['selected_index'] is not None and
            s['attempts'][s['selection']['selected_index']]['reason']=='preserve_departure_failure' for s in seed_results),
        eligibility_interpretation='ELIGIBLE means retained without retry, not certified obstacle-free; ambiguous evidence remains in the cohort.',
        selection_coverage=len(selected)/len(seed_results) if seed_results else None,
        unconditional=ordinary['scenario_aggregate'],conditional=conditional['scenario_aggregate'],
        conditional_atomic=conditional['measurement_aggregate']['per_mechanic'],seeds=seed_results)
    ordinary['sampling_summary']='First sample for each engine seed, without any eligibility filtering.'
    conditional['sampling_summary']=summary
    dump_json(out/'unconditional-results.json',ordinary);dump_json(out/'conditional-results.json',conditional)
    dump_json(out/'summary.json',summary)
    gallery=deepcopy(report);gallery.update(benchmark_cohorts=summary,
        dependency_aggregate_by_scope=conditional['dependency_aggregate_by_scope'],
        scenario_aggregate=conditional['scenario_aggregate'])
    for i,row in enumerate(gallery['samples']):
        name=Path(row['evidence']).name;trial=trial_by_case[name]
        current=deepcopy(selected_by_case.get(name,row))
        current['ordinary_scenario_results_by_branch']=row['scenario_results_by_branch']
        current['ordinary_measurements_by_branch']=row['measurements_by_branch']
        current['conditional_attempt']=dict(trial,unconditional_first_sample=trial['sample_index']==0,
            selected_conditional=name in selected_by_case,
            score_scope='conditional_full_route' if name in selected_by_case else 'ordinary_unfiltered')
        gallery['samples'][i]=current
    dump_json(out/'gallery-results.json',gallery)
    return out/'summary.json'


def run(args):
    out=Path(args.out)
    if out.exists():raise ValueError('Use a new conditional output directory')
    if not 1<=args.max_samples<2**32:raise ValueError('max-samples must be between 1 and 2**32-1')
    sampling_seed(0,0,args.seed_step)
    source=Path(args.first_evidence);first_manifest=json.loads((source/'manifest.json').read_text())
    if first_manifest['status']!='complete':raise ValueError('First-sample evidence must be complete')
    if digest(args.model)!=first_manifest['provenance']['checkpoint_sha256']:
        raise ValueError('Retry checkpoint differs from the unconditional checkpoint')
    config=first_manifest['config']
    if args.first_measurements:
        header=json.loads((Path(args.first_measurements)/'results.json').read_text())
        if header['evidence_manifest_sha256']!=digest(source/'manifest.json'):
            raise ValueError('First measurements belong to different evidence')
    if args.module!=config['module'] or args.fp32!=config['fp32']:
        raise ValueError('Use the same model module and precision as the first samples')
    candidates=[]
    for name in first_manifest['cases']:
        meta=json.loads((source/name/'metadata.json').read_text())
        if meta['spec']['name']=='table_leave_return' and meta['model_sample']==0:candidates.append((meta['seed'],name))
    if not candidates or len({s for s,_ in candidates})!=len(candidates):
        raise ValueError('Expected exactly one first table-return sample per engine seed')
    out.mkdir();evidence=out/'evidence';evidence.mkdir();cache=out/'readings';cache.mkdir()
    engine=Engine();reader=ReaderV2(engine,engine.construct(catalog()[0],100000,8));adapter=None
    manifest=dict(schema_version=3,status='in_progress',cases=[],config=dict(suite='conditional_table_return',
        scenario=['table_leave_return'],context=1,horizon=16,max_samples=args.max_samples,seed_step=args.seed_step,
        module=args.module,fp32=args.fp32,retry_batch_size=1,first_evidence=str(source.resolve())),
        provenance=dict(checkpoint_sha256=digest(args.model),backend=jax.default_backend(),
                        first_manifest_sha256=digest(source/'manifest.json'),rejection_sha256=digest(Path(__file__).with_name('rejection.py')),
                        collector_sha256=digest(__file__),sources={str(p):digest(p) for p in sorted(Path(__file__).resolve().parent.glob('*.py'))}))
    sampling=dict(max_samples_per_seed=args.max_samples,seed_schedule='(recorded first model seed + sample_index * odd seed_step) mod 2**32',
                  seed_step=args.seed_step,seeds=[])
    dump_json(evidence/'manifest.json',manifest)
    for environment_seed,first_name in sorted(candidates):
        first_meta,first_arrays=load_case(source/first_name)
        if len(first_arrays['context'])!=1 or first_arrays['context_actions'].tolist()!=[0]:
            raise ValueError('Conditional experiment requires one real frame and NOOP incoming marker')
        if first_arrays['b0_actions'].tolist()!=[8]+[3]*6+[4]*6+[0]*3:
            raise ValueError('Conditional experiment requires the original fixed 16-frame sequence')
        refs=reference_pair(engine,Scenario(**first_meta['spec']),environment_seed,1,16)
        for branch in range(2):
            np.testing.assert_array_equal(refs[branch].context,first_arrays['context'])
            np.testing.assert_array_equal(refs[branch].frames,first_arrays[f'b{branch}_reference'])
        context_hash=hashlib.sha256(first_arrays['context'].tobytes()).hexdigest()
        trials=[]
        for sample_index in range(args.max_samples):
            model_seed=sampling_seed(first_meta['model_seed'],sample_index,args.seed_step)
            name=f'table_leave_return_s{environment_seed}_m{sample_index}';target=evidence/name
            if sample_index==0:
                shutil.copytree(source/first_name,target)
            else:
                if adapter is None:
                    adapter=ModuleAdapter(args.module,args.model,args.fp32)
                    manifest['provenance']['module_sha256']=digest(adapter.module.__file__)
                    if manifest['provenance']['module_sha256']!=first_manifest['provenance']['module_sha256']:
                        raise ValueError('Retry model module differs from first-sample module')
                predictions=[]
                for ref in refs:
                    pred=adapter.rollout(ref.context[None],ref.context_actions[None],ref.actions[None],model_seed,True)
                    predictions.append({k:v[0] if v is not None else None for k,v in pred.items()})
                save_case(target,Scenario(**first_meta['spec']),refs,predictions,environment_seed,model_seed,sample_index,first_meta['category'])
                meta=json.loads((target/'metadata.json').read_text())
                meta['replay']=dict(context_length=1,engine_seed=environment_seed,model_seed=model_seed,module=args.module,
                    checkpoint_sha256=manifest['provenance']['checkpoint_sha256'],batch_size_actual=1,batch_size_configured=1,fp32=args.fp32,
                    sampling_settings=first_manifest['provenance']['sampling_settings'])
                dump_json(target/'metadata.json',meta)
            meta,arrays=load_case(target)
            assert hashlib.sha256(arrays['context'].tobytes()).hexdigest()==context_hash
            saved=None
            if sample_index==0 and args.first_measurements:
                saved=json.loads((Path(args.first_measurements)/first_name/'measurements.json').read_text())
                if saved['reader']!=reader.profile:raise ValueError('First reader profile differs; omit --first-measurements to reread RGB')
                if saved['rgb_evidence']!=str((source/first_name/'trajectory.npz').resolve()):
                    raise ValueError('First cached RGB source mismatch')
            if saved is None:
                saved=dict(context_last=reader.read(arrays['context'][-1]),
                    generated=[[reader.read(frame) for frame in arrays[f'b{b}_generated']] for b in range(2)],reader=reader.profile)
            saved['rgb_evidence']=str((target/'trajectory.npz').resolve())
            (cache/name).mkdir();dump_json(cache/name/'measurements.json',saved)
            eligibility=departure_eligibility([saved['context_last']]+saved['generated'][0])
            dump_json(target/'eligibility.json',eligibility)
            trial=dict(case=name,sample_index=sample_index,engine_seed=environment_seed,model_seed=model_seed,
                decision=eligibility['decision'],reason=eligibility['reason'],conditioning_sha256=context_hash,
                evidence=str(target.resolve()),eligibility_evidence=str((target/'eligibility.json').resolve()),
                eligibility={'decision':eligibility['decision'],'reason':eligibility['reason']})
            trials.append(trial);manifest['cases'].append(name);dump_json(evidence/'manifest.json',manifest)
            print(f'[{environment_seed}:{sample_index}] {eligibility["decision"]} {eligibility["reason"]}',flush=True)
            if eligibility['decision']!='REJECTED_BLOCKED_ROUTE':break
        selection=retry_selection(trials,args.max_samples)
        sampling['seeds'].append(dict(engine_seed=environment_seed,attempts=trials,selection=selection))
        dump_json(out/'sampling.json',sampling)
    manifest['status']='complete';dump_json(evidence/'manifest.json',manifest)
    dump_json(cache/'cache_manifest.json',dict(reader=reader.profile,evidence_manifest_sha256=digest(evidence/'manifest.json')))
    return finalize(out)
