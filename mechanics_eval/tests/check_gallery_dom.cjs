// Offline DOM-behavior unit check. No browser, network, server or npm packages.
// This does not certify rendering, file-protocol behavior or mobile layout.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const file=process.argv[2];if(!file)throw Error('Usage: node mechanics_eval/tests/check_gallery_dom.cjs path/index.html');
const source=fs.readFileSync(file,'utf8'),ids=new Map();
class Element {
 constructor(tag,attrs={}){this.tag=tag;this.attrs=attrs;this.children=[];this.ownText='';this.events={};this.style={};this.value=attrs.value||'';this.hidden='hidden' in attrs;this.dataset={};for(const [k,v] of Object.entries(attrs)){if(k==='id')ids.set(v,this);if(k.startsWith('data-'))this.dataset[k.slice(5).replace(/-([a-z])/g,(_,c)=>c.toUpperCase())]=v}this.className=attrs.class||''}
 append(...nodes){for(const n of nodes){this.children.push(n);if(this.tag==='select'&&this.children.length===1)this.value=n.value}}
 replaceChildren(...nodes){this.children=[];this.ownText='';this.append(...nodes)}
 set textContent(text){this.children=[];this.ownText=String(text)}
 get textContent(){return this.ownText+this.children.map(n=>n.textContent).join('')}
 get childNodes(){return this.children}
 setAttribute(k,v){this.attrs[k]=String(v);if(k==='class')this.className=String(v);if(k.startsWith('data-'))this.dataset[k.slice(5).replace(/-([a-z])/g,(_,c)=>c.toUpperCase())]=String(v)}
 addEventListener(event,fn){(this.events[event]??=[]).push(fn)}
 dispatch(event){for(const fn of this.events[event]||[])fn({target:this})}
 querySelectorAll(selector){const all=[];function match(n){const tag=selector.match(/^[\w-]+/)?.[0],cls=selector.match(/\.([\w-]+)/)?.[1],attr=selector.match(/\[([\w-]+)\]/)?.[1];return(!tag||n.tag===tag)&&(!cls||n.className.split(' ').includes(cls))&&(!attr||attr in n.attrs)}function walk(n){for(const c of n.children){if(match(c))all.push(c);walk(c)}}walk(this);return all}
 querySelector(selector){return this.querySelectorAll(selector)[0]||null}
}
const root=new Element('document'),stack=[root],prefix=source.split('<script id="gallery-data"')[0];
// Parse the actual generated markup; scripts and JSON are processed separately.
for(const token of prefix.matchAll(/<\/?([a-zA-Z][\w-]*)([^>]*)>|([^<]+)/g)){
 if(token[3]){stack.at(-1).ownText+=token[3];continue}
 const [full,tag,tail]=token;if(full.startsWith('</')){while(stack.length>1&&stack.at(-1).tag!==tag)stack.pop();if(stack.length>1)stack.pop();continue}
 const attrs={};for(const m of tail.matchAll(/([\w-]+)(?:="([^"]*)")?/g))attrs[m[1]]=m[2]??'';
 const n=new Element(tag,attrs);stack.at(-1).append(n);if(!['meta','link','input','img','br','hr'].includes(tag))stack.push(n);
}
const payload=source.match(/<script id="gallery-data" type="application\/json">([\s\S]*?)<\/script>/)[1],data=JSON.parse(payload);
const script=new Element('script',{id:'gallery-data'});script.textContent=payload;root.append(script);
const document={getElementById:id=>ids.get(id),createElement:tag=>new Element(tag),querySelectorAll:s=>root.querySelectorAll(s)};
const scripts=[...source.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)];
const context=vm.createContext({document,console});vm.runInContext(scripts.at(-1)[1],context,{timeout:10000});
const get=id=>ids.get(id),articles=root.querySelectorAll('article[data-card]');assert.equal(articles.length,data.cards.length);
const initial=data.cohorts.find(c=>c.id===(get('cohort')?.value));
assert(initial && initial.summary.all.mechanics,'per-mechanic summary must be included in the gallery');
const checkMechanics=()=>{const scope=get('scope').value,cohort=data.cohorts.find(c=>c.id===get('cohort').value);
 const expected=cohort.summary[scope].mechanics.scenarios.length;
 const bodyRows=get('mechanics').querySelectorAll('tr[data-mechanic]');
 assert.equal(bodyRows.length,expected,'visible mechanic frequencies must match this cohort and scope');
 for(const row of cohort.summary[scope].mechanics.scenarios){
  assert.equal(row.attempts,row.passed+row.failed+row.unknown);
  const actual=bodyRows.find(n=>n.dataset.mechanic===row.name&&n.dataset.scope===row.scope);assert(actual);
  assert.equal(actual.children[1].textContent,`${row.passed}/${row.attempts}`);
  assert.equal(actual.children[2].textContent,(100*row.pass_rate).toFixed(1)+'%');
  assert(actual.children[3].textContent.includes(`${row.failed} fail · ${row.unknown} unknown`));
  assert(actual.children[3].textContent.includes(`${row.engine_seed_count} engine seeds`));
 }
 const atoms=get('mechanics').querySelectorAll('tr[data-atomic]');
 assert.equal(atoms.length,cohort.summary[scope].mechanics.atomic.length);
 for(const row of cohort.summary[scope].mechanics.atomic){
  const actual=atoms.find(n=>n.dataset.atomic===row.name&&n.dataset.scenario===row.scenario&&n.dataset.required===String(row.required)&&n.dataset.scope===row.scope);assert(actual);
  assert.equal(actual.children[1].textContent,`${row.passed}/${row.eligible}`);
  assert.equal(actual.children[2].textContent,row.pass_rate==null?'—':(100*row.pass_rate).toFixed(1)+'%');
  assert(actual.children[3].textContent.includes(`eligibility ${row.eligible}/${row.applicable}`));
 }
};

let checks=0;function verify(){const cohort=get('cohort').value,scope=get('scope').value;const expected=data.cards.filter(c=>{const view=c.views[cohort!=='selected'&&c.views.ordinary?'ordinary':'current'];return c.cohorts.includes(cohort)&&(scope==='all'||scope===c.scope)&&(get('status').value==='all'||get('status').value===view.status)&&(get('category').value==='all'||get('category').value===c.category)&&(get('branch').value==='all'||get('branch').value===c.branch)&&(get('scenario').dataset.exact?(c.scenario===get('scenario').dataset.exact&&c.scope===get('scenario').dataset.exactScope):c.case.toLowerCase().includes(get('scenario').value.trim().toLowerCase()))});assert.equal(articles.filter(n=>!n.hidden).length,expected.length);assert.equal(get('empty').hidden,expected.length!==0);const stats=data.cohorts.find(c=>c.id===cohort).summary[scope].scenarios;for(const [i,k] of ['pass','fail','unknown'].entries())assert.equal(get('stats').children[i].children[1].textContent,String(stats[k]));assert.equal(get('stats').children[3].children[1].textContent,stats.measurement_coverage==null?'—':(100*stats.measurement_coverage).toFixed(1)+'%');checks++}
function set(id,value,event='change'){get(id).value=value;get(id).dispatch(event);verify();if(id==='cohort'||id==='scope')checkMechanics()}
verify();checkMechanics();
for(const cohort of data.cohorts){set('cohort',cohort.id);for(const scope of ['all','static','dynamic'])set('scope',scope);set('scope','all');const summary=get('stats').textContent;for(const status of ['pass','fail','unknown','not_applicable','all']){set('status',status);assert.equal(get('stats').textContent,summary)}for(const branch of ['selected','noop','alternative','consistency','all'])set('branch',branch);for(const category of [...new Set(data.cards.map(c=>c.category))])set('category',category);set('category','all');set('scenario','gather','input');set('scenario','__NO_MATCH__','input');get('reset').dispatch('click');verify()}
let clickChecks=0;
for(const cohort of data.cohorts){
 set('cohort',cohort.id);set('scope','all');
 const rows=get('mechanics').querySelectorAll('tr[data-mechanic]');
 for(const row of rows){
  const scenario=row.dataset.mechanic,scope=row.dataset.scope,summaryBefore=get('stats').textContent,scopeBefore=get('scope').value;
  row.querySelector('button').dispatch('click');verify();checkMechanics();
  assert.equal(get('scenario').value,scenario);assert.equal(get('branch').value,'selected');
  assert.equal(get('scope').value,scopeBefore);assert.equal(get('stats').textContent,summaryBefore,'mechanic clicks must not alter summary denominators');
  const expected=data.cards.filter(c=>c.cohorts.includes(cohort.id)&&c.scenario===scenario&&c.scope===scope&&c.branch==='selected');
  assert(expected.length>0);assert.equal(articles.filter(n=>!n.hidden).length,expected.length);
  for(const n of articles.filter(n=>!n.hidden))assert.equal(data.cards[Number(n.dataset.card)].scenario,scenario);
  get('reset').dispatch('click');verify();assert(!get('scenario').dataset.exact);
  clickChecks++;
 }
}
let details=0;for(const n of root.querySelectorAll('details.stages[data-stage-view]')){n.open=true;n.dispatch('toggle');assert(n.querySelector('.stage-content').textContent.length>0);details++}
assert(!source.includes('fetch('));assert(!/<script[^>]+src=/.test(source));assert(source.includes('@media(max-width:720px)'));
console.log(JSON.stringify({page:file,cards:articles.length,filter_checks:checks,mechanic_click_checks:clickChecks,expanded_stage_views:details,no_external_scripts:true,method:'offline simulated DOM execution; real browser/visual check pending'},null,2));
