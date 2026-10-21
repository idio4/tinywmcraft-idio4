"""Independent mechanics targets. Only RGB readings supply generated facts."""
from .visual import HUD

INVENTORY = [key for key in HUD if key != 'health']
TOOLS = {11:'wood_pickaxe',12:'stone_pickaxe',13:'iron_pickaxe',
         14:'wood_sword',15:'stone_sword',16:'iron_sword'}


def result(name, facts, **metadata):
    from .milestone import conjunction
    return dict(mechanic=name, status=conjunction(f['status'] for f in facts), facts=facts,
                complete=bool(facts) and all(f['observed'] is not None for f in facts),
                facts_accepted=sum(f['observed'] is not None for f in facts),facts_total=len(facts),**metadata)


def check(name, expected, observed, reason='unreadable_or_unimplemented'):
    from .milestone import fact
    return fact(name,expected,observed,reason)


def delta(before, after, field):
    a,b=before.get(field),after.get(field)
    return b-a if a is not None and b is not None else None


def scalar(name, field, expected, observed, reason='unreadable_or_unimplemented', **metadata):
    return result(name,[check(field,expected,observed,reason)],**metadata)


def inapplicable(name,reason,milestone):
    return dict(mechanic=name,status='not_applicable',reason=reason,milestone=milestone,
                facts=[],complete=False,facts_accepted=0,facts_total=0)


def movement(readings, actions, reference, timestep, expected, anchor=0):
    """Score geometry afterward; changed visible collision conditions can invalidate a fixed route."""
    from .motion import displacement
    raw=displacement(readings,timestep,expected,anchor)
    if timestep>=len(readings):return raw
    from craftax.craftax_classic.constants import DIRECTIONS, SOLID_BLOCKS
    solids=set(map(int,SOLID_BLOCKS));facts=[];comparability=[]
    for t in range(anchor+1,timestep+1):
        action=int(actions[t-1]);direction=list(map(int,DIRECTIONS[action]))
        observed=raw['motion_trace'][t]['adjacent']['displacement']
        # Unknown collisions allow a stop or the commanded single step, never a teleport.
        facts.append(dict(check('legal_single_step',True,
                          observed in ([0,0],direction) if observed is not None else None,
                          'local_motion_unreadable'),timestep=t,measured_displacement=observed))
        expected_local=[0,0];reason='local_motion_unreadable'
        if any(direction):
            key=f'{3+direction[0]},{4+direction[1]}'
            tile=readings[t-1]['tiles'].get(key)
            entity=readings[t-1].get('measurements',{}).get('tile:'+key,{}).get('entity')
            expected_local=None if tile is None or entity else [0,0] if tile in solids else direction
            reason='entity_collision_uncertain' if entity else 'collision_tile_unreadable' if tile is None else reason
        previous=reference[t-2] if t>1 else dict(row_delta=0,col_delta=0)
        engine=[reference[t-1][k]-previous[k] for k in ('row_delta','col_delta')]
        # Different generated terrain is acceptable. Do not call a physically valid stop a failed move.
        if observed is not None and observed!=engine and (expected_local is None or observed==expected_local):
            comparability.append(dict(timestep=t,reason=reason if expected_local is None else 'generated_collision_conditions_differ',
                                      reference_step=engine,visible_expected_step=expected_local,observed_step=observed))
        facts.append(dict(check('relative_displacement',expected_local,observed if expected_local is not None else None,reason),
                          timestep=t,measured_displacement=observed,motion=raw['motion_trace'][t]))
    raw['local_action_consequences']=result('action_consequences',facts,milestone=timestep)
    raw['route_comparability']=comparability
    raw['measured_displacement']=raw['motion']['position']
    if comparability:
        raw.update(status='unknown',complete=False,facts_accepted=0)
        raw['facts']=[check('relative_displacement',list(expected),None,'reference_route_not_comparable')]
    return raw


def same_view(anchor, current):
    """Compatibility gate, now backed by temporal terrain consensus rather than exact map identity."""
    from .motion import estimate
    motion=estimate(anchor,current)
    return motion['accepted'] and motion['displacement']==[0,0]


def consistency(rule, readings, actions, field=None, anchor=0, original_tile=None):
    """One rule, one verdict. readings[0] is the real initial observation.

    Persistence is conditional on independently seeing the object/harvest established.
    An absent/unreadable prerequisite is UNKNOWN, never a failed creation/kill in disguise.
    """
    common=dict(rule=rule,target_field=field,anchor_timestep=anchor,action_sequence=[int(a) for a in actions])
    def unknown(reason):
        return scalar(rule,rule,True,None,reason=reason,**common)
    if len(readings)<2 or len(actions)!=len(readings)-1:
        return unknown('insufficient_valid_trajectory')
    if rule in ('noop_inventory_unchanged','crafted_tool_persistence') and any(int(a)!=0 for a in actions[anchor:]):
        return unknown('requires_stationary_noop_suffix')
    if rule=='noop_inventory_unchanged':
        facts=[]
        for t in range(1,len(readings)):
            for key in INVENTORY:
                a,b=readings[0].get(key),readings[t].get(key)
                facts.append(dict(check(key,a,b if a is not None else None),timestep=t))
        return result(rule,facts,**common)
    if anchor>=len(readings)-1:
        return unknown('no_observation_after_anchor')
    first=readings[anchor]
    if rule=='crafted_tool_persistence':
        established=first.get(field)
        if established is None or established<=0:
            return unknown('tool_not_observed_at_anchor')
        return result(rule,[dict(check(field,established,r.get(field)),timestep=t)
                           for t,r in enumerate(readings[anchor+1:],anchor+1)],**common)
    if rule=='placed_table_persistence':
        established=first['tiles'].get('3,5')
        if established!=11:
            return unknown('table_not_observed_at_anchor')
        return result(rule,[dict(check('target_table',11,r['tiles'].get('3,5') if same_view(first,r) else None,
                           'unreadable_tile_or_unverified_stationary_view'),timestep=t)
                           for t,r in enumerate(readings[anchor+1:],anchor+1)],**common)
    if rule=='harvested_resource_persistence':
        initial=readings[0]['tiles'].get('3,5');established=first['tiles'].get('3,5')
        if initial is None or initial!=original_tile or established is None or established==initial:
            return unknown('harvest_not_observed_at_anchor')
        return result(rule,[dict(check('harvested_resource_stays_absent',True,
                           r['tiles'].get('3,5')!=initial if r['tiles'].get('3,5') is not None and same_view(first,r) else None,
                           'unreadable_tile_or_unverified_stationary_view'),timestep=t)
                           for t,r in enumerate(readings[anchor+1:],anchor+1)],**common)
    raise ValueError(rule)


def short_tests(spec, readings, actions, reference, initial, tiles):
    """Return a designated atomic primary plus individually named diagnostics.

    Reference data define only expected consequences. They never fill an unreadable RGB fact.
    """
    name=spec['name'];m=spec['milestone'];diagnostics={}
    if m>=len(readings) or m>len(reference):
        return scalar(name,'milestone',True,None,reason='milestone_not_reached'),diagnostics
    def quantity(field,t=m):
        previous=initial if t==1 else reference[t-2]
        expected=reference[t-1][field]-previous[field]
        if field=='sapling' and int(actions[t-1])==5 and (name=='consumed_resource_stays_gone' or (name=='gather_empty' and spec['front']==2)):
            return scalar('inventory_delta:'+field+'@t'+str(t),field,expected,None,
                          reason='stochastic_effect_not_exactly_scored',milestone=t,
                          observed_delta=delta(readings[t-1],readings[t],field),reference_delta=expected)
        return scalar('inventory_delta:'+field+'@t'+str(t),field,expected,
                      delta(readings[t-1],readings[t],field),milestone=t)
    # Retain inventory effects, but never conjoin them into another target's verdict.
    for field in INVENTORY:
        diagnostics['inventory_delta:'+field]=quantity(field)
    if spec['expected_tile'] is not None:
        diagnostics['target_tile']=scalar('target_tile','tile:3,5',tiles[m-1]['3,5'],
                                         readings[m]['tiles'].get('3,5'),milestone=m)
    creations=[]
    for t,action in enumerate(spec['actions'][:m],1):
        if int(action) in TOOLS:
            tool=TOOLS[int(action)];key='tool_creation:'+tool+'@t'+str(t)
            effect=quantity(tool,t)['facts'][0]
            entry=scalar(key,tool+'_created',effect['expected']>0,
                         effect['observed']>0 if effect['observed'] is not None else None,milestone=t)
            diagnostics[key]=entry;creations.append(entry)
            # Each ingredient gets its own consumption result; unrelated stocks remain inventory diagnostics.
            from .milestone import RECIPES
            for ingredient in RECIPES.get(tool,{}):
                entry=quantity(ingredient,t);entry['mechanic']='ingredient_consumption:'+ingredient+'@t'+str(t)
                diagnostics[entry['mechanic']]=entry
    for field in INVENTORY:
        if 'pickaxe' in field or 'sword' in field:
            before,after=readings[m-1].get(field),readings[m].get(field)
            if before is not None and before>0:
                diagnostics['tool_consistency:'+field]=scalar('tool_consistency:'+field,field,
                    True,after>=before if after is not None else None,milestone=m)
    if name=='attack_zombie':
        # Target kill identity is required; total entity population is not a valid proxy when spawning differs.
        primary=scalar('target_zombie_killed','target_zombie_killed',reference[m-1]['zombies']<initial['zombies'],
                       readings[m].get('target_zombie_killed'),reason=readings[m].get('target_zombie_evidence',{}).get('reason','target_entity_reader_unimplemented'),milestone=m,
                       visual_evidence=readings[m].get('target_zombie_evidence'))
    elif name=='death':
        primary=scalar('player_termination','done',reference[m-1]['done'],readings[m].get('done'),milestone=m)
        diagnostics['health_change']=quantity('health')
    elif name=='take_damage':
        primary=quantity('health');primary['mechanic']='health_damage'
    elif creations:
        primary=creations[-1]  # progression claims only final-tool creation; earlier creations remain separate tests.
    elif name in ('gather_wood','gather_stone','gather_stone_no_tool','gather_empty'):
        field='stone' if 'stone' in name else 'wood'
        primary=quantity(field);primary['mechanic']=field+'_acquisition'
    elif name=='consumed_resource_stays_gone':
        if not any(int(a)==5 for a in actions[:m]):
            return inapplicable('harvested_resource_persistence','branch_without_harvest',m),diagnostics
        # First interaction establishes harvest; repeated DO may collect saplings, unrelated to tree absence.
        first=readings[1]['tiles'].get('3,5');original=readings[0]['tiles'].get('3,5')
        facts=[check('harvest_observed',True,None if first is None or original is None else first!=original)]
        for t in range(2,m+1):
            value=readings[t]['tiles'].get('3,5')
            facts.append(dict(check('tree_stays_absent',True,value!=original if value is not None and original is not None and same_view(readings[1],readings[t]) else None,
                                    'unreadable_tile_or_unverified_stationary_view'),timestep=t))
        # Missing initial harvest is a prerequisite failure, not a failed persistence rule.
        if first is None or original is None or first==original:
            primary=scalar('harvested_resource_persistence','harvest_established',True,None,
                           reason='harvest_not_observed_at_anchor',milestone=m)
        else:primary=result('harvested_resource_persistence',facts,milestone=m)
    elif name=='table_leave_return':
        value=readings[1]['tiles'].get('3,5')
        primary=scalar('table_creation','table_present',tiles[0]['3,5']==11,
                       value==11 if value is not None else None,milestone=1)
        diagnostics['table_creation@t1']=primary
    elif name.startswith('place_'):
        primary=diagnostics['target_tile'];primary['mechanic']='target_object_placement'
    elif name=='noop':
        primary=(consistency('noop_inventory_unchanged',readings[:m+1],actions[:m]) if all(int(a)==0 for a in actions[:m]) else
                 inapplicable('noop_inventory_unchanged','branch_not_noop',m));primary['milestone']=m
    elif name in ('move_free','move_blocked'):
        primary=movement(readings,actions,reference,m,[reference[m-1]['row_delta'],reference[m-1]['col_delta']])
    elif name=='turn_loop':
        primary=movement(readings,actions,reference,m,[reference[m-1]['row_delta'],reference[m-1]['col_delta']])
        primary['mechanic']='return_to_origin'
        diagnostics['direction']=scalar('direction','direction',reference[m-1]['direction'],readings[m].get('direction'),milestone=m)
    elif name=='sleep':
        primary=scalar('sleep_entry','sleeping',reference[m-1]['sleeping'],readings[m].get('sleeping'),milestone=m,
                       visual_evidence=readings[m].get('player_evidence'))
    else:raise ValueError('No atomic definition for '+name)
    return primary,diagnostics


def contrast(first,second):
    """Only the primary atomic target enters the paired comparison."""
    from .milestone import conjunction
    expected_a=[f['expected'] for f in first['facts']];expected_b=[f['expected'] for f in second['facts']]
    if 'not_applicable' in (first['status'],second['status']) or expected_a==expected_b or first['mechanic'] in ('noop_inventory_unchanged','harvested_resource_persistence'):
        return dict(status='not_identifiable',facts=[],complete=False,mechanic=first['mechanic'])
    return dict(result(first['mechanic'],first['facts']+second['facts']),
                status=conjunction([first['status'],second['status']]),
                branch_outcomes=[first['status'],second['status']])
