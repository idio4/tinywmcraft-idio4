"""Three-valued mechanics outcomes and observable long-horizon invariants, replayed from saved RGB."""
from collections import Counter
import csv
import json
from pathlib import Path
import numpy as np
from .evidence import digest, dump_json, load_case
from .reader_v2 import ReaderV2
from .scenarios import Engine, catalog
from .visual import HUD
from .audit import confusion


def conjunction(statuses):
    statuses = list(statuses)
    if 'fail' in statuses:
        return 'fail'
    return 'pass' if statuses and all(s == 'pass' for s in statuses) else 'unknown'


def fact(name, expected, observed, reason='unreadable_or_unimplemented'):
    return {'field': name, 'expected': expected, 'observed': observed,
            'status': 'unknown' if observed is None else 'pass' if observed == expected else 'fail',
            'reason': reason if observed is None else 'accepted_match' if observed == expected else 'accepted_contradiction'}


def outcome(spec, read, expected=None, target_tile=None):
    expected = spec['expected'] if expected is None else expected
    target_tile = spec['expected_tile'] if target_tile is None else target_tile
    facts = []
    for key, value in expected.items():
        facts.append(fact(key, value, read.get(key), 'extractor_not_implemented' if key not in HUD else
                          read['measurements'][key]['digit']['reason']))
        if key in HUD:
            facts.append(fact('item:' + key, 'empty' if value == 0 else key, read['items'][key]))
    if target_tile is not None:
        facts.append(fact('tile:3,5', target_tile, read['tiles']['3,5'],
                          read['measurements']['tile:3,5']['reason']))
    return {'status': conjunction(f['status'] for f in facts), 'facts': facts,
            'complete': all(f['observed'] is not None for f in facts),
            'facts_accepted': sum(f['observed'] is not None for f in facts), 'facts_total': len(facts)}


def paired(spec, first, second, truth_first, truth_second, tile_first=None, tile_second=None):
    facts = []
    for key in spec['expected']:
        if key in HUD:
            delta = truth_first[key] - truth_second[key]
            a, b = first.get(key), second.get(key)
            if delta:
                facts.append(fact('delta:' + key, delta, a - b if a is not None and b is not None else None))
    if spec['expected_tile'] is not None and tile_first != tile_second:
        # Object/tile transition is categorical, not an arithmetic subtraction of block IDs.
        facts.append(fact('paired_target_tile', [tile_first, tile_second],
                          [first['tiles']['3,5'], second['tiles']['3,5']]
                          if first['tiles']['3,5'] is not None and second['tiles']['3,5'] is not None else None))
    if not facts:
        return {'status': 'not_identifiable', 'facts': [], 'complete': False}
    left = outcome(spec, first)
    expected_other = {k: truth_second[k] for k in spec['expected']}
    right = outcome(spec, second, expected_other, tile_second if spec['expected_tile'] is not None else None)
    return {'status': conjunction([left['status'], right['status'], *[f['status'] for f in facts]]),
            'facts': facts, 'branch_outcomes': [left['status'], right['status']],
            'complete': left['complete'] and right['complete'] and all(f['observed'] is not None for f in facts)}


RECIPES = {'wood_pickaxe': {'wood': 1}, 'stone_pickaxe': {'wood': 1, 'stone': 1},
           'iron_pickaxe': {'wood': 1, 'stone': 1, 'iron': 1, 'coal': 1}}


def invariants(readings, context, actions, limit, spec):
    """Compare generated evidence with generated evidence, never a distant recorded future."""
    checks = []
    for t in range(limit):
        before, after = context if t == 0 else readings[t-1], readings[t]
        for key in HUD:
            if key == 'health':
                continue
            a, b = before[key], after[key]
            if int(actions[t]) == 0:
                checks.append(dict(fact('noop_inventory:' + key, a, b) if a is not None else
                                   fact('noop_inventory:' + key, None, None), step=t+1))
            elif 'pickaxe' in key or 'sword' in key:
                checks.append(dict(fact('tool_not_lost:' + key, True, b >= a if a is not None and b is not None else None), step=t+1))
        for tool, costs in RECIPES.items():
            a, b = before[tool], after[tool]
            if a is not None and b is not None and b > a:
                for resource, cost in costs.items():
                    x, y = before[resource], after[resource]
                    observed = x - y if x is not None and y is not None else None
                    checks.append(dict(fact('observed_crafting_cost:' + tool + ':' + resource,
                                            cost * (b-a), observed), step=t+1))
    # Screen-local persistence only after confidently observing creation/removal and no subsequent movement.
    if spec['expected_tile'] is not None and spec['category'] in ('placement','gathering'):
        milestone = spec['milestone'] - 1
        if milestone < limit and all(int(a) == 0 for a in actions[milestone+1:limit]):
            established = readings[milestone]['tiles']['3,5']
            if established is not None:
                for t in range(milestone+1, limit):
                    checks.append(dict(fact('observed_tile_persistence', established, readings[t]['tiles']['3,5']), step=t+1))
    return {'status': conjunction(c['status'] for c in checks),
            'checks': len(checks), 'accepted': sum(c['observed'] is not None for c in checks),
            'violations': [c for c in checks if c['status'] == 'fail'],
            'unknown': [c for c in checks if c['status'] == 'unknown'],
            'complete': bool(checks) and all(c['observed'] is not None for c in checks)}


def rates(rows, key='status'):
    count = Counter(row[key] for row in rows)
    n = len(rows); p, f, u = (count[s] for s in ('pass','fail','unknown'))
    return {'n': n, 'pass': p, 'fail': f, 'unknown': u, 'not_identifiable': count['not_identifiable'],
            'success_rate_lower': p/n if n else None, 'success_rate_upper': (p+u)/n if n else None,
            'measurement_coverage': (p+f)/n if n else None,
            'success_given_measured': p/(p+f) if p+f else None,
            'complete_measurement_coverage': sum(r.get('complete',False) for r in rows)/n if n else None}


def summarize(rows):
    groups = {}
    for group in sorted({r['category'] for r in rows}):
        selected = [r for r in rows if r['category'] == group]
        identifiable = [r['counterfactual'] for r in selected if r['counterfactual']['status'] != 'not_identifiable']
        groups[group] = {'cases': len(selected), 'counterfactual_not_identifiable': len(selected)-len(identifiable),
                         'outcomes': rates([r['outcome'] for r in selected]),
                         'counterfactual': rates(identifiable)}
    macro = {key: float(np.mean([v['outcomes'][key] for v in groups.values()]))
             for key in ('success_rate_lower','success_rate_upper','measurement_coverage','complete_measurement_coverage')}
    cf_groups = [g['counterfactual'] for g in groups.values() if g['counterfactual']['n']]
    cf_macro = {key: float(np.mean([g[key] for g in cf_groups])) if cf_groups else None
                for key in ('success_rate_lower','success_rate_upper','measurement_coverage','complete_measurement_coverage')}
    cf_macro.update(identifiable_categories=len(cf_groups), total_categories=len(groups))
    return {'groups': groups, 'macro': macro, 'counterfactual_macro': cf_macro,
            'overall': rates([r['outcome'] for r in rows])}


def evaluation_targets(spec, group, reference_facts):
    """Reference-only targets for deterministic short effects; never a labeler for generated RGB."""
    from .focused import GROUPS
    if group == 'combat':
        # Attacking never consumes a sword/tool. A readable loss fails even when kills are unknown.
        tools = {key: reference_facts[key] for key in HUD if 'sword' in key or 'pickaxe' in key}
        return dict(spec, expected={**spec['expected'], **tools})
    if group not in GROUPS:
        return spec
    stochastic = {'sapling'} if spec['name']=='consumed_resource_stays_gone' or (
        spec['name']=='gather_empty' and spec['front']==2) else set()
    return dict(spec, expected={key:reference_facts[key] for key in HUD
                                if key!='health' and key not in stochastic})


def score_run(evidence, out, measurement_source=None, conditional=False):
    evidence, out = Path(evidence), Path(out)
    if out.exists():
        raise ValueError('Use a new score directory')
    manifest = json.loads((evidence / 'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('Evidence collection is incomplete')
    engine = Engine()
    reader = ReaderV2(engine, engine.construct(catalog()[0], 100000, 8))
    replay_hashes={}
    if measurement_source is not None:
        measurement_source=Path(measurement_source)
        header_path=measurement_source/('results.json' if (measurement_source/'results.json').exists() else 'cache_manifest.json')
        header=json.loads(header_path.read_text())
        if header['reader']!=reader.profile or header['evidence_manifest_sha256']!=digest(evidence/'manifest.json'):
            raise ValueError('Cached measurements do not match the current RGB reader and evidence')
    out.mkdir(parents=True)
    rows = []
    reference_records = {'digits': [], 'items': [], 'tiles': []}
    comparisons = []
    for name in manifest['cases']:
        meta, arrays = load_case(evidence / name)
        spec = meta['spec']; milestone = spec['milestone'] - 1
        if measurement_source is None:
            read = [[reader.read(frame) for frame in arrays[f'b{branch}_generated']] for branch in range(2)]
            ctx = reader.read(arrays['context'][-1])
        else:
            cached_path=measurement_source/name/'measurements.json';cached=json.loads(cached_path.read_text())
            if cached['reader']!=reader.profile or cached['rgb_evidence']!=str((evidence/name/'trajectory.npz').resolve()):
                raise ValueError('Cached measurement source mismatch: '+name)
            if any(len(cached['generated'][b])!=len(arrays[f'b{b}_generated']) for b in range(2)):
                raise ValueError('Cached trajectory length mismatch: '+name)
            read=cached['generated'];ctx=cached['context_last'];replay_hashes[name]=digest(cached_path)
        truth = meta['reference_truth']
        reached = milestone < min(len(read[0]), len(truth['0']['facts']), len(truth['1']['facts']))
        from .atomic import short_tests, contrast, consistency
        atomic_branches=[]; diagnostic_branches=[]; measurement_branches=[]; scenario_branches=[]
        for branch in range(2):
            n=len(arrays[f'b{branch}_reference'])
            trajectory=[ctx]+read[branch][:n]
            if spec['name'] == 'attack_zombie':
                from .target_reader import target_zombie
                target_zombie(trajectory)
            selected, diagnostics=short_tests(spec,trajectory,arrays[f'b{branch}_actions'],
                truth[str(branch)]['facts'],truth[str(branch)]['initial'],truth[str(branch)]['tiles'])
            atomic_branches.append(selected);diagnostic_branches.append(diagnostics)
            from .dependencies import build
            measurements,scenario_result=build(spec,trajectory,arrays[f'b{branch}_actions'],selected,diagnostics,
                                               truth[str(branch)]['facts'],truth[str(branch)]['initial'],conditional=conditional)
            for item in measurements:
                item.update(scope='dynamic' if spec['zombie'] else 'static',
                            measurement_id=f'{evidence.name}/{name}/b{branch}/'+item['name'])
            measurement_branches.append(measurements);scenario_branches.append(scenario_result)
        result,other=atomic_branches
        counter=contrast(result,other)
        row = {'scenario': spec['name'], 'category': meta['category'], 'seed': meta['seed'],
               'scope':'dynamic' if spec['zombie'] else 'static',
               'model_sample': meta['model_sample'], 'milestone_observed': reached,
               'primary_mechanic':result['mechanic'],
               'outcome':result,'branch_outcomes':atomic_branches,'counterfactual':counter,
               'diagnostics_by_branch':diagnostic_branches,'consistency_tests':[],
               'measurements_by_branch':measurement_branches,'scenario_results_by_branch':scenario_branches,
               'scenario_result':scenario_branches[0],
               'reward_mae_short':float(np.abs(arrays['b0_rewards'][:milestone+1]-arrays['b0_reward_reference'][:milestone+1]).mean())
                   if 'b0_rewards' in arrays else None,
               'evidence':str((evidence/name).resolve())}
        from .metrics import target_outcome
        from .scenarios import Scenario
        strict = reader.strict.read(arrays['b0_generated'][milestone]) if reached else None
        old = target_outcome(Scenario(**spec), strict) if strict else None
        comparisons.append({'scenario': spec['name'], 'seed':meta['seed'], 'v1_complete_case':old,
                            'v2_status':result['status'],'v2_complete':result['complete']})
        # Audit on reference RGB only; generated RGB receives no hidden labels.
        for branch in range(2):
            for t, frame in enumerate(arrays[f'b{branch}_reference']):
                rr = reader.read(frame); facts = truth[str(branch)]['facts'][t]
                for field in HUD:
                    reference_records['digits'].append({'truth':facts[field], 'prediction':rr[field]})
                    reference_records['items'].append({'truth':field if facts[field]>0 else 'empty','prediction':rr['items'][field]})
                if not spec['zombie'] and spec['name'] != 'sleep':
                    for key, value in rr['tiles'].items():
                        reference_records['tiles'].append({'truth':truth[str(branch)]['tiles'][t][key], 'prediction':value})
        case_out = out / name; case_out.mkdir()
        dump_json(case_out/'measurements.json', {'context_last':ctx, 'generated':read,
                   'reader':reader.profile,'rgb_evidence':str((evidence/name/'trajectory.npz').resolve())})
        dump_json(case_out/'outcomes.json', row)
        rows.append(row)
    validation = {k:confusion(v) for k,v in reference_records.items()}
    reference_ok = all(v['correct'] == v['accepted'] for v in validation.values())
    from .dependencies import summarize_measurements
    measurement_summary=summarize_measurements(rows)
    from . import motion
    report = {'schema_version':5, 'metric_version':'level1-atomic-v4-rgb-pose-target', 'kind':'milestone_mechanics_results',
              'fixed_route_required':conditional,
              'measurement_replay':{'source':str(measurement_source.resolve()),'file_sha256':replay_hashes} if measurement_source is not None else None,
              'status':'provisional_pending_generated_human_audit' if reference_ok else 'failed_reference_reader_audit',
              'generated_human_validation':'pending', 'scorer_sha256':digest(__file__),
              'atomic_scorer_sha256':digest(Path(__file__).with_name('atomic.py')),
              'dependency_scorer_sha256':digest(Path(__file__).with_name('dependencies.py')), 'reader':reader.profile,
              'persistence_scorer_sha256':digest(Path(__file__).with_name('persistence.py')),
              'motion_tracker':{'version':motion.VERSION,'implementation_sha256':digest(motion.__file__),
                                'min_support':motion.MIN_SUPPORT,'min_types':motion.MIN_TYPES,
                                'min_agreement':motion.MIN_AGREEMENT,'min_margin':motion.MIN_MARGIN,
                                'max_shift':motion.MAX_SHIFT,'uses_actions_or_reference_futures':False},
              'evidence_manifest_sha256':digest(evidence/'manifest.json'), 'samples':rows,
              'aggregate':summarize(rows), 'aggregate_semantics':'legacy single-primary atomic compatibility metric, not scenario overall',
              'measurement_aggregate':measurement_summary,
              'scenario_aggregate':measurement_summary['overall_scenarios'],
              'dependency_aggregate_by_scope':{scope:summarize_measurements([r for r in rows if r['scope']==scope])
                                            for scope in ('static','dynamic') if any(r['scope']==scope for r in rows)},
              'aggregate_by_scope':{scope:summarize([r for r in rows if r['scope']==scope])
                                    for scope in ('static','dynamic') if any(r['scope']==scope for r in rows)},
              'reference_reader_validation':validation,
              'v1_comparison_same_rgb':comparisons,
              'limitations':['Grid motion abstains on insufficient, repeated or incoherent landmarks; maximum adjacent shift is two tiles.',
                            'Zombie identity is limited to an unambiguous first-step visible reachable region; general visual termination is unsupported. Player pose abstains on corruption and noisy night lighting.',
                            'Generated-reader accuracy requires independent labels; clean-reference accuracy is not sufficient.',
                            'Long-horizon invariants compare generated observations with each other, not recorded future pixels.']}
    dump_json(out/'results.json', report)
    columns = ['scenario','category','seed','status','complete','facts_accepted','facts_total','counterfactual']
    with (out/'samples.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,columns);writer.writeheader()
        for r in rows:
            writer.writerow({k:r[k] for k in ('scenario','category','seed')} | {k:r['outcome'][k]
                             for k in ('status','complete','facts_accepted','facts_total')} |
                             {'counterfactual':r['counterfactual']['status']})
    text=['# Dependency-aware mechanics milestone','','Reader: '+reader.profile['version'],
          'Status: '+report['status']+'. Generated-image human validation is pending.',
          'Pass/fail are conclusive measurements; unknown remains in the denominator. Complete-fact coverage is reported separately.','',
          '| Category | N | Pass | Fail | Unknown | Coverage | Success bounds |','|---|---:|---:|---:|---:|---:|---:|']
    for name, group in report['aggregate']['groups'].items():
        s=group['outcomes'];text.append(f"| {name} | {s['n']} | {s['pass']} | {s['fail']} | {s['unknown']} | {s['measurement_coverage']:.1%} | {s['success_rate_lower']:.1%}–{s['success_rate_upper']:.1%} |")
    text += ['', 'Overall scenario results: '+json.dumps(report['scenario_aggregate']),
             'Each overall uses only its explicitly required stages. Atomic NOT_TESTED entries are excluded from eligible denominators.',
             '', 'Legacy single-primary macro averages: '+json.dumps(report['aggregate']['macro']), '',
             'Reader validation on generated images is reported separately; this table does not establish reader reliability.',
             'All frames, crops, actions and reward predictions are saved in the evidence directory; per-field reasons and distances are in each measurements.json.']
    (out/'summary.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({'atomic_primary':report['aggregate']['overall'],
                      'scenarios':report['scenario_aggregate']},indent=2))
    return report
