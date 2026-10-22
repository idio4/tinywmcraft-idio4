"""Presentation of saved verdicts. No RGB interpretation or inference occurs here."""
import html
import json
from pathlib import Path


def compact_stage(stage):
    keys=('name','category','required','eligible','status','timestep','reason','observation_reason',
          'expected','observed','prerequisites')
    value={k:stage[k] for k in keys if k in stage}
    # Keep directly useful object evidence, rather than embedding full motion traces in HTML.
    if stage.get('object_evidence'):
        value['object_evidence']={k:v for k,v in stage['object_evidence'].items() if k in
            ('target_coordinate','estimated_player_position','localization_status','target_visibility',
             'tile_reader_status','tile_reader_reason','observed_tile','expected_tile','entity_occlusion')}
    return value


def compact_selection(attempt,eligibility):
    value={k:attempt[k] for k in ('decision','reason','sample_index','unconditional_first_sample','selected_conditional')}
    for key in ('witnesses','failure_evidence','ambiguities'):
        value[key]=[{k:e[k] for k in ('timestep','reason','measured_step','tile','entity','world_relative_coordinate') if k in e}
                    for e in eligibility.get(key,[])]
    return value


def compact_verdict(verdict):
    value={k:verdict[k] for k in ('mechanic','status','reason','milestone','branch_outcomes') if k in verdict}
    value['facts']=[{k:f[k] for k in ('field','expected','observed','status','reason','timestep') if k in f}
                    for f in verdict.get('facts',[])]
    return value


def summary(rows, ordinary=False):
    """Count existing selected-branch verdicts; gallery/media filters never enter this function."""
    from .dependencies import atomic_rates, CATEGORIES
    import numpy as np
    from .milestone import rates
    from .performance import summarize as mechanic_summary, scenario_verdict, atomic_stages
    result={}
    for scope in ('all','static','dynamic'):
        selected=[r for r in rows if scope=='all' or r.get('scope','static')==scope]
        scenarios=[scenario_verdict(r, ordinary) for r in selected]
        stages=[s for r in selected for s in atomic_stages(r, ordinary)]
        result[scope]=dict(scenarios=rates([dict(s,complete=s['status']!='unknown') for s in scenarios]),
            required=atomic_rates([s for s in stages if s['required']]),
            diagnostic=atomic_rates([s for s in stages if not s['required']]),
            categories={c:{role:atomic_rates([s for s in stages if s['category']==c and s['required']==required])
                for role,required in [('required',True),('diagnostic',False)]} for c in CATEGORIES if any(s['category']==c for s in stages)})
        groups=[g['required'] for g in result[scope]['categories'].values() if g['required']['eligible']]
        result[scope]['category_macro']={k:float(np.mean([g[k] for g in groups])) if groups else None
            for k in ('success_rate','measurement_coverage','eligibility_coverage')}
    for scope in result:
        scoped = [r for r in rows if scope == "all" or r.get("scope", "static") == scope]
        result[scope]["mechanics"] = mechanic_summary(scoped, ordinary=ordinary)
    return result


def write_page(out, data):
    out=Path(out);cards=[]
    for i,card in enumerate(data['cards']):
        views=[]
        for view,content in card['views'].items():
            status=content['status'];gif=content.get('gif')
            media=(f'<img loading="lazy" src="{html.escape(gif,quote=True)}" alt="Real Craftax and generated {html.escape(card["case"],quote=True)} trajectory">'
                   f'<a download href="{html.escape(gif,quote=True)}">Download GIF ↗</a>') if gif else '<div class="no-media">GIF not exported for this view.<br><small>All results and evidence metadata remain available.</small></div>'
            views.append(f'<section data-view="{view}"'+(' hidden' if view=='ordinary' else '')+'>'
                f'<div class="verdict-line"><span class="badge {status}">{status.upper()}</span><span>Scenario overall: {status.upper()}</span></div>'
                f'<p class="failure">{html.escape(content.get("failure") or "No confirmed required-stage failure.")}</p>'
                +media+f'<details class="stages" data-card="{i}" data-stage-view="{view}"><summary>Required atomic stages & diagnostics <span>+</span></summary><div class="stage-content"></div></details></section>')
        selection=card.get('selection')
        selection_html=(f'<p class="card-meta">Departure selection: {html.escape(selection["decision"])} · {html.escape(selection["reason"])} · attempt {selection["sample_index"]} · retained: {selection["selected_conditional"]}</p><details class="stages"><summary>Departure-selection evidence</summary><pre>{html.escape(json.dumps(selection,indent=2))}</pre></details>') if selection else ''
        links=' · '.join(f'<a href="{html.escape(url,quote=True)}">{html.escape(label)}</a>' for label,url in card.get('links',{}).items())
        cards.append(f'<article id="case-{i}" data-card="{i}" data-first-status="{card["views"].get("ordinary",card["views"]["current"])["status"]}">'
            f'<div class="card-kicker">{html.escape(card["category"].replace("_"," "))} / {card["scope"]}</div>'
            f'<h3>{html.escape(card["case"])}</h3><p class="card-meta">{html.escape(card["branch"])} branch · engine {card["seed"]} · model {card["model_seed"]}</p>'
            f'<p class="card-meta">Atomic target: {html.escape(card["atomic_status"].upper())} · paired contrast: {html.escape(card["counterfactual_status"].upper())}</p>'
            +selection_html+''.join(views)+(f'<details class="stages"><summary>Paired atomic contrast</summary><pre>{html.escape(json.dumps(card["counterfactual"],indent=2))}</pre></details>' if card.get('counterfactual') else '')+f'<details class="stages"><summary>Actions and sampling</summary><pre>{html.escape(json.dumps({k:card.get(k) for k in ("actions","seed","model_seed","context_length","valid_steps")},indent=2))}</pre></details><footer>{links}</footer></article>')
    document=json.dumps(data,separators=(',',':'),ensure_ascii=True,allow_nan=False)
    payload=document.replace('<','\\u003c')
    template=Path(__file__).with_name('gallery.html').read_text()
    page=template.replace('__CARDS__','\n'.join(cards)).replace('__DATA__',payload)
    page=page.replace('__CHECKPOINT__',html.escape(data.get('checkpoint_name','saved checkpoint'),quote=True))
    (out/'index.html').write_text(page)
    (out/'gallery.json').write_text(document+'\n')
    return out/'index.html'
