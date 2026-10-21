"""Small dependency layer over existing RGB-only atomic measurements."""
from .atomic import scalar, result, consistency, delta, TOOLS, movement

CATEGORIES=('Crafting','Resource collection','Placement','Inventory accounting',
            'Inventory persistence','Object persistence','Movement','Combat and survival')


def measurement(raw, name, category, required=False, prerequisites=(), end=1):
    """Normalize an atomic result without recomputing its verdict."""
    facts=raw['facts']
    times=sorted({f.get('timestep',raw.get('milestone',end)) for f in facts})
    bad=min((f for f in facts if f['status']=='fail'),key=lambda f:f.get('timestep',raw.get('milestone',end)),default=None)
    unknown=min((f for f in facts if f['status']=='unknown'),key=lambda f:f.get('timestep',raw.get('milestone',end)),default=None)
    chosen=bad or unknown
    return dict(raw,name=name,category=category,required=required,eligible=True,
                prerequisites=list(prerequisites),timesteps=times,
                timestep=chosen.get('timestep',raw.get('milestone',end)) if chosen else (times[-1] if times else end),
                expected=[{'field':f['field'],'timestep':f.get('timestep',raw.get('milestone',end)),'value':f['expected']} for f in facts],
                observed=[{'field':f['field'],'timestep':f.get('timestep',raw.get('milestone',end)),'value':f['observed']} for f in facts],
                reason='observation_unknown' if raw['status']=='unknown' else
                       bad['reason'] if bad else 'observations_match',
                observation_reason=unknown['reason'] if unknown else None)


def prerequisite(name,status,**evidence):
    return dict(mechanic=name,status=status,**evidence)


def dependent(raw,name,category,parents,condition,required=True,end=16):
    prereqs=[prerequisite(p['name'],p['status']) for p in parents]
    if condition is not None:prereqs.append(condition)
    failed=any(p['status']=='fail' or p.get('reason')=='prerequisite_failed' for p in parents)
    failed=failed or (condition is not None and condition['status']=='fail')
    blocked=failed or any(p['status']!='pass' for p in parents) or (condition is not None and condition['status']!='pass')
    entry=measurement(raw,name,category,required,prereqs,end)
    if blocked:
        entry.update(status='not_tested',eligible=False,complete=False,facts_accepted=0,
                     reason='prerequisite_failed' if failed else 'prerequisite_unknown',
                     observation_reason=None)
        # No hypothetical persistence observations are counted when ineligible.
        entry['facts']=[];entry['observed']=[]
    return entry


def not_applicable(name,category,reason,required=False,end=16):
    entry=measurement(scalar(name,'applicability',True,None,reason=reason),name,category,required,end=end)
    entry.update(status='not_applicable',eligible=False,complete=False,reason=reason,
                 observation_reason=None,facts=[],facts_total=0,facts_accepted=0,observed=[],expected=[])
    if category=='Object persistence':
        entry['object_evidence']=dict(target_coordinate=None,estimated_player_position=None,
            localization_status='not_applicable',target_visibility='not_applicable',tile_reader_status='not_applicable',
            tile_reader_reason=reason,expected_tile=None,observed_tile=None,entity_occlusion={'status':'not_applicable'},confidence=None)
    return entry


def presence(field,value,timestep,expected=None):
    holds=None if value is None else value>0 if expected is None else value==expected
    return prerequisite('visible_presence', 'unknown' if holds is None else 'pass' if holds else 'fail',
                        field=field,timestep=timestep,expected='positive_quantity' if expected is None else expected,
                        observed=value)


def overall(stages):
    required=[s for s in stages if s['required'] and s['status']!='not_applicable']
    first=min((s for s in required if s['status']=='fail'),key=lambda s:s.get('timestep',0),default=None)
    blocked_failure=min((s for s in required if s['status']=='not_tested' and s['reason']=='prerequisite_failed'),
                        key=lambda s:s.get('timestep',0),default=None)
    failure=min((s for s in (first,blocked_failure) if s is not None),key=lambda s:s.get('timestep',0),default=None)
    status='fail' if failure else 'pass' if required and all(s['status']=='pass' for s in required) else 'unknown'
    fact=min((f for f in failure.get('facts',[]) if f['status']=='fail'),
             key=lambda f:f.get('timestep',failure['timestep']),default=None) if failure else None
    return dict(status=status,required_stages=[s['name'] for s in required],
                first_failing_stage=failure['name'] if failure else None,
                first_failure=dict(stage=failure['name'],timestep=failure['timestep'],reason=failure['reason'],fact=fact,
                                   prerequisites=failure['prerequisites']) if failure else None)


def atomic_rates(stages):
    applicable=[s for s in stages if s['status']!='not_applicable']
    eligible=[s for s in applicable if s['eligible']]
    p,f,u=(sum(s['status']==status for s in eligible) for status in ('pass','fail','unknown'))
    n=len(eligible);total=len(stages)
    return dict(total=total,applicable=len(applicable),not_applicable=total-len(applicable),eligible=n,not_tested=len(applicable)-n,**{'pass':p},fail=f,unknown=u,
                success_rate=p/n if n else None,success_rate_upper=(p+u)/n if n else None,
                measurement_coverage=(p+f)/n if n else None,eligibility_coverage=n/len(applicable) if applicable else None)


def summarize_measurements(rows):
    stages=[s for r in rows for s in r['measurements_by_branch'][0]]
    groups={}
    for name in sorted({s['name'] for s in stages}):
        selected=[s for s in stages if s['name']==name]
        groups[name]=dict(category=selected[0]['category'],**atomic_rates(selected))
    categories={}
    for category in CATEGORIES:
        selected=[s for s in stages if s['category']==category]
        if selected:
            categories[category]={role:atomic_rates([s for s in selected if s['required']==required])
                                  for role,required in [('required',True),('diagnostic',False)]}
    from .milestone import rates
    by_scenario={}
    for scenario in sorted({r['scenario'] for r in rows}):
        selected=[r for r in rows if r['scenario']==scenario]
        by_scenario[scenario]=dict(overall=rates([dict(r['scenario_result'],complete=r['scenario_result']['status']!='unknown') for r in selected]),
            mechanics={name:atomic_rates([s for r in selected for s in r['measurements_by_branch'][0] if s['name']==name])
                       for name in sorted({s['name'] for r in selected for s in r['measurements_by_branch'][0]})})
    return dict(per_mechanic=groups,by_category=categories,by_scenario=by_scenario,
                overall_scenarios=rates([dict(r['scenario_result'],complete=r['scenario_result']['status']!='unknown') for r in rows]))


def build(spec, readings, actions, primary, diagnostics, reference, initial, conditional=False):
    """Declare only relevant required stages; reuse the existing measurements once."""
    stages=[];used=set();m=spec['milestone'];end=len(readings)-1;name=spec['name']
    def add(raw,label,category,required=False):
        entry=(not_applicable(label,category,raw['reason'],required,end=m) if raw['status']=='not_applicable' else
               measurement(raw,label,category,required,end=m));stages.append(entry)
        used.add(raw['mechanic']);return entry
    def retain_diagnostics():
        for key in sorted(diagnostics,key=lambda k:0 if k.startswith('ingredient_consumption:') else 1):
            raw=diagnostics[key]
            if raw['mechanic'] in used:continue
            if any(raw['facts']==s['facts'] and raw.get('milestone',m)==s.get('milestone',m) for s in stages):continue
            category='Placement' if key=='target_tile' else 'Movement' if key=='direction' else 'Inventory accounting'
            label=('inventory_accounting.' if category=='Inventory accounting' else 'placement.' if category=='Placement' else 'movement.')+key.replace(':','.').replace('@t','_t')
            add(raw,label,category)
    def retain(parent,field,t):
        raw=consistency('crafted_tool_persistence',readings,actions,field=field,anchor=t)
        stage=dependent(raw,'inventory_persistence.'+field,'Inventory persistence',[parent],
                        presence(field,readings[t].get(field) if t<len(readings) else None,t),end=end)
        stages.append(stage);return stage
    def tile_persistence(parent,original,t=1,absence=False):
        from .persistence import visible_persistence
        anchor=readings[t]['tiles'].get('3,5') if t<len(readings) else None
        condition=prerequisite('visible_harvest', 'unknown' if anchor is None else 'pass' if anchor!=original else 'fail',
                               field='tile:3,5',timestep=t,expected='not '+str(original),observed=anchor) if absence else presence('tile:3,5',anchor,t,original)
        raw=visible_persistence(readings,original,t,absence)
        stage=dependent(raw,'object_persistence.'+('harvested_resource' if absence else 'crafting_table' if original==11 else 'furnace' if original==12 else 'placed_object'),
                        'Object persistence',[parent],condition,end=end)
        stages.append(stage);return stage
    creation_keys=[k for k in diagnostics if k.startswith('tool_creation:')]
    if creation_keys:
        for key in creation_keys:
            raw=diagnostics[key];tool=key.split(':',1)[1].split('@')[0];t=raw['milestone']
            creation=add(raw,'crafting.'+tool+'_creation','Crafting',True)
            # Invalid-crafting scenarios intend rejection, not subsequent retention of a nonexistent tool.
            expected=raw['facts'][0]['expected']
            if expected and all(int(a)==0 for a in actions[t:]):retain(creation,tool,t)
            elif not expected:
                stages.append(not_applicable('inventory_persistence.'+tool,'Inventory persistence','no_tool_creation_expected',end=end))
    elif name in ('gather_wood','gather_stone'):
        resource='wood' if name=='gather_wood' else 'stone'
        add(primary,'resource_collection.'+resource+'_acquisition','Resource collection',True)
        original=readings[0]['tiles'].get('3,5');tile=readings[1]['tiles'].get('3,5') if end>=1 else None
        expected_removed=diagnostics['target_tile']['facts'][0]['expected']!=spec['front']
        removal=scalar('resource_removal','target_removed',expected_removed,tile!=original if tile is not None and original is not None else None,milestone=1)
        removed=add(removal,'resource_collection.target_removal','Resource collection',True)
        if expected_removed:
            # Removal, not inventory gain, is the necessary prerequisite for absence persistence.
            tile_persistence(removed,spec['front'],absence=True)
        else:stages.append(not_applicable('object_persistence.harvested_resource','Object persistence','no_harvest_expected',end=end))
    elif name=='consumed_resource_stays_gone':
        original=readings[0]['tiles'].get('3,5');tile=readings[1]['tiles'].get('3,5') if end>=1 else None
        expected_removed=diagnostics['target_tile']['facts'][0]['expected']!=spec['front']
        removal=add(scalar('target_removal','target_removed',expected_removed,tile!=original if tile is not None and original is not None else None,milestone=1),
                    'resource_collection.target_removal','Resource collection',True)
        if expected_removed:tile_persistence(removal,spec['front'],absence=True)
        else:stages.append(not_applicable('object_persistence.harvested_resource','Object persistence','no_harvest_expected',end=end))
        used.add(primary['mechanic'])  # superseded by the same removal + persistence stages, not counted twice.
    elif name=='table_leave_return':
        from .motion import displacement, track, object_observation
        from .persistence import visible_persistence
        from .atomic import check
        placement=add(diagnostics['table_creation@t1'],'placement.crafting_table','Placement',True)
        applicable=bool(int(actions[0])==8 and diagnostics['table_creation@t1']['facts'][0]['expected'])
        if not applicable:
            # The alternative branch makes a categorical creation prediction, not a promise to retain an absent table.
            action_result=movement(readings,actions,reference,min(13,end),[0,0])
            if 'local_action_consequences' in action_result:
                add(action_result['local_action_consequences'],'movement.action_consequences','Movement',True)
            for label,category in [('movement.fixed_route_fidelity','Movement'),('movement.displacement','Movement'),
                                   ('movement.return_to_origin','Movement'),('object_persistence.visible','Object persistence'),
                                   ('object_persistence.after_return','Object persistence')]:
                stages.append(not_applicable(label,category,'branch_without_table_excursion',end=end))
            retain_diagnostics()
            return stages,overall(stages)
        if end<13 or len(reference)<13:
            add(scalar('return_to_origin','relative_displacement',[0,0],None,
                       reason='milestone_not_reached',milestone=13),'movement.return_to_origin','Movement',True)
            return stages,overall(stages)
        departure=movement(readings,actions,reference,7,[reference[6]['row_delta']-reference[0]['row_delta'],
                    reference[6]['col_delta']-reference[0]['col_delta']],anchor=1)
        leave=add(departure,'movement.displacement','Movement',True)
        route=movement(readings,actions,reference,13,[reference[12]['row_delta']-reference[0]['row_delta'],
                    reference[12]['col_delta']-reference[0]['col_delta']],anchor=1)
        add(route['local_action_consequences'],'movement.action_consequences','Movement',True)
        raw=displacement(readings,13,[0,0],anchor=1)
        positions=[r['position'] for r in departure['motion_trace'][2:]]
        away=any(p is not None and p!=[0,0] for p in positions)
        departure_condition=prerequisite('observed_departure','pass' if away else 'unknown' if None in positions else 'fail',
                                         observed=positions)
        # A shorter but visibly completed trip still permits an independent object check.
        parents=[] if raw['motion']['position']==[0,0] else [leave]
        returned=dependent(raw,'movement.return_to_origin','Movement',parents,departure_condition,end=13);stages.append(returned)
        full_trace=track(readings[:14]);facts=[]
        for t in range(1,14):
            expected=[-min(t-1,6)+max(t-7,0),0]
            facts.append(dict(check('relative_displacement',expected,full_trace[t]['position'],full_trace[t]['reason']),
                              timestep=t,motion=full_trace[t]))
        fidelity=add(result('fixed_route_fidelity',facts,milestone=13,motion_trace=full_trace),
                     'movement.fixed_route_fidelity','Movement',conditional)
        trace=track(readings[:14],anchor=1)
        evidence=object_observation(readings[13],trace[13]['position'],expected_tile=11,localization=trace[13])
        raw=scalar('table_visibility','tile:tracked_table',11,evidence['value'],reason=evidence['reason'],
                   milestone=13,motion_trace=trace,object_anchor_coordinate=[3,5],object_evidence=evidence,
                   confidence=trace[13].get('confidence'))
        raw['excursion_requirement']='fixed_six_tile_route' if conditional else 'observed_excursion'
        stages.append(dependent(raw,'object_persistence.after_return','Object persistence',[placement,returned]+([fidelity] if conditional else []),
                               presence('tile:3,5',readings[1]['tiles'].get('3,5'),1,11),end=13))
        stages.append(dependent(visible_persistence(readings,11),'object_persistence.visible','Object persistence',[placement],
                               presence('tile:3,5',readings[1]['tiles'].get('3,5'),1,11),end=end))
        used.add(primary['mechanic'])
    elif name.startswith('place_'):
        placed=add(primary,'placement.target_object','Placement',True)
        # Only successful placement is an intended creation/persistence sequence.
        if spec['valid'] and spec['expected_tile'] is not None:
            if int(actions[0]) in (8,9):tile_persistence(placed,spec['expected_tile'])
            else:stages.append(not_applicable('object_persistence.'+('crafting_table' if spec['expected_tile']==11 else 'furnace'),
                                             'Object persistence','branch_without_placement',end=end))
    elif name in ('move_free','move_blocked','turn_loop'):
        add(primary,'movement.'+primary['mechanic'],'Movement',True)
        if 'local_action_consequences' in primary:
            add(primary['local_action_consequences'],'movement.action_consequences','Movement',True)
        if name=='turn_loop' and len(reference)>=2:
            add(movement(readings,actions,reference,2,[reference[1]['row_delta'],reference[1]['col_delta']]),
                'movement.displacement','Movement',True)
    else:
        category='Combat and survival' if spec['zombie'] or name=='sleep' else 'Movement' if spec['category']=='movement' else 'Inventory accounting' if name=='noop' else 'Resource collection'
        label=('survival.' if name=='sleep' else 'combat.' if spec['zombie'] else 'movement.' if category=='Movement' else 'inventory_accounting.' if name=='noop' else 'resource_collection.')+primary['mechanic']
        add(primary,label,category,True)
        if name=='noop':
            from .motion import displacement
            add(displacement(readings,m,[reference[m-1]['row_delta'],reference[m-1]['col_delta']]),
                'movement.displacement','Movement',True)
    # Retain diagnostics, but skip aliases of the already measured primary/creation stage.
    retain_diagnostics()
    # Full NOOP inventory accounting can share the counterfactual recording without generating a new test.
    if name=='gather_wood' and all(int(a)==0 for a in actions):
        add(consistency('noop_inventory_unchanged',readings,actions),
            'inventory_accounting.noop_inventory_unchanged','Inventory accounting')
    return stages,overall(stages)
