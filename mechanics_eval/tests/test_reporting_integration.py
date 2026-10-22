"""Standalone synthetic reporting fixtures; no checkpoint or historical corpus required."""
import csv
import json
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from mechanics_eval.evidence import digest, dump_json
from mechanics_eval.gifs import export_gifs, gif_enabled
from mechanics_eval.performance import summarize, write_csv, markdown
from mechanics_eval.tests.test_gallery import Links
from mechanics_eval.tests.test_performance import case


def fixture(root, conditional=False):
    evidence=root/'evidence';evidence.mkdir()
    entries=[('gather_stone','pass',0,0,'static'),('gather_stone','unknown',0,1,'static'),
             ('gather_stone_no_tool','fail',1,0,'static'),('attack_zombie','pass',2,0,'dynamic'),
             ('gather_stone','fail',3,0,'dynamic')]
    if conditional:
        entries=[('table_leave_return','fail',0,0,'static'),('table_leave_return','fail',0,1,'static'),
                 ('table_leave_return','unknown',1,0,'static')]
    rows=[];names=[]
    for scenario,status,seed,sample,scope in entries:
        name=f'{scenario}_s{seed}_m{sample}';names.append(name)
        target=evidence/name;target.mkdir()
        row=case(scenario,status,seed,stage_status=status,scope=scope)
        row.update(model_sample=sample,evidence=str(target.resolve()),outcome={'status':status},
                   branch_outcomes=[{'status':status},{'status':'pass'}],counterfactual={'status':'not_identifiable'})
        required=row['measurements_by_branch'][0][0]
        required.update(timestep=1,reason='synthetic',expected=[],observed=[],prerequisites=[])
        row['measurements_by_branch'][0].append(dict(required,name='inventory.optional',required=False,status='fail'))
        if conditional:
            ordinary='pass' if sample==1 else status
            row['ordinary_scenario_results_by_branch']=[{'status':ordinary},{'status':'pass'}]
            row['ordinary_measurements_by_branch']=[[dict(required,status=ordinary)],[]]
            row['conditional_attempt']=dict(decision='REJECTED_BLOCKED_ROUTE' if seed==0 and sample==0 else 'ELIGIBLE',
                reason='synthetic',sample_index=sample,unconditional_first_sample=sample==0,
                selected_conditional=sample==1 or seed==1)
            dump_json(target/'eligibility.json',{})
        frames=np.zeros((2,63,63,3),np.uint8);frames[1]=40
        arrays={'context':frames[:1], 'context_actions':np.array([0],np.int32)}
        for branch in range(2):
            arrays.update({f'b{branch}_reference':frames,f'b{branch}_generated':frames,
                           f'b{branch}_actions':np.array([2,0] if branch==0 else [0,0]),
                           f'b{branch}_done_reference':np.array([False,False])})
        np.savez_compressed(target/'trajectory.npz',**arrays)
        dump_json(target/'metadata.json',dict(seed=seed,model_seed=100+seed,model_sample=sample,category='crafting',
            spec=dict(name=scenario,milestone=1,zombie='attack' if scope=='dynamic' else ''),
            trajectory_sha256=digest(target/'trajectory.npz')))
        dump_json(target/'sampling.json',{})
        rows.append(row)
    dump_json(evidence/'manifest.json',dict(status='complete',cases=names,config=dict(suite='focused')))
    report={'samples':rows,'mechanic_performance':summarize(rows)}
    if conditional:
        counts=lambda p,f,u:dict(n=p+f+u,pass_rate=p/(p+f+u),measurement_coverage=(p+f)/(p+f+u),
                               **{'pass':p,'fail':f,'unknown':u})
        report['benchmark_cohorts']=dict(unconditional=counts(0,1,1),conditional=counts(0,1,1),
            total_generated_attempts=3,first_samples_reused=2,newly_generated_attempts=1,
            rejected_blocked_routes=1,selected_eligible=2)
        for filename in ('unconditional-results.json','conditional-results.json','summary.json'):
            dump_json(root/filename,{})
    result=root/'results.json';dump_json(result,report)
    return evidence,result


class ReportingIntegrationTests(unittest.TestCase):
    def assert_export(self, out):
        data=json.loads((out/'gallery.json').read_text())
        page=(out/'index.html').read_text()
        payload=re.search(r'<script id="gallery-data" type="application/json">(.*?)</script>',page,re.S)[1]
        self.assertEqual(json.loads(payload),data)
        self.assertNotIn('seed-6',page)
        self.assertNotIn('Six conditional table-loss observations',page)
        self.assertFalse((out/'evidence.zip').exists())
        parser=Links();parser.feed(page)
        for url in parser.urls:
            if not url.startswith(('https:','http:','#')):
                self.assertTrue((out/url).is_file(),url)
        with (out/'mechanics.csv').open() as stream:
            csv_rows=list(csv.DictReader(stream))
        expected=[]
        for cohort in data['cohorts']:
            for scope,summary in cohort['summary'].items():
                for kind,entries in (('scenario',summary['mechanics']['scenarios']),('atomic',summary['mechanics']['atomic'])):
                    for row in entries:
                        expected.append(dict(row,cohort=cohort['id'],scope=scope,measurement_scope=row['scope'],kind=kind))
        self.assertEqual(len(csv_rows),len(expected))
        for actual,row in zip(csv_rows,expected):
            for key,value in actual.items():
                self.assertEqual(value,'' if row.get(key) is None else str(row.get(key)),key)
        return data

    def test_all_gif_modes_keep_frequencies_links_and_denominators(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            with patch('mechanics_eval.rollouts.ModuleAdapter',side_effect=AssertionError('no inference')), \
                 patch('mechanics_eval.milestone.score_run',side_effect=AssertionError('no rescoring')):
                for conditional in (False,True):
                    source=root/str(conditional);source.mkdir();evidence,result=fixture(source,conditional)
                    before=digest(result);summaries=None
                    for mode in ('all','fail','unknown','nonpass','none'):
                        out=source/mode;export_gifs([evidence],[result],out,mode,gif_workers=1)
                        data=self.assert_export(out)
                        if summaries is None:summaries=data['cohorts']
                        self.assertEqual(data['cohorts'],summaries)
                        for card in data['cards']:
                            for view in card['views'].values():
                                self.assertEqual(bool(view['gif']),gif_enabled(mode,view['status']))
                        if mode=='none':self.assertFalse(list(out.rglob('*.gif')))
                    self.assertEqual(digest(result),before)

    def test_conditional_cohorts_and_interactive_controls(self):
        if not shutil.which('node'):self.skipTest('Node.js unavailable')
        with TemporaryDirectory() as temp:
            root=Path(temp)
            for conditional in (False,True):
                source=root/str(conditional);source.mkdir();evidence,result=fixture(source,conditional)
                out=source/'report';export_gifs([evidence],[result],out,'none')
                data=self.assert_export(out)
                if conditional:
                    cohorts={c['id']:c for c in data['cohorts']}
                    for name,n in [('first',2),('selected',2),('rejected',1),('attempts',3)]:
                        self.assertEqual(cohorts[name]['summary']['all']['scenarios']['n'],n)
                    self.assertEqual(cohorts['attempts']['summary']['all']['mechanics']['scenarios'][0]['passed'],1)
                    self.assertEqual(cohorts['selected']['summary']['all']['mechanics']['scenarios'][0]['passed'],0)
                check=subprocess.run(['node',str(Path(__file__).with_name('check_gallery_dom.cjs')),str(out/'index.html')],
                                     capture_output=True,text=True)
                self.assertEqual(check.returncode,0,check.stdout+check.stderr)
                report=json.loads(check.stdout)
                self.assertGreater(report['filter_checks'],0)
                self.assertGreater(report['mechanic_click_checks'],0)

    def test_scorer_csv_and_markdown_use_json_counts(self):
        rows=[case('craft_stone_pickaxe','pass',i) for i in range(9)]+[case('craft_stone_pickaxe','unknown',9)]
        performance=summarize(rows)
        with TemporaryDirectory() as temp:
            target=Path(temp)/'mechanics.csv';write_csv(target,performance)
            with target.open() as stream:records=list(csv.DictReader(stream))
        self.assertEqual(records[0]['passed'],'9')
        self.assertEqual(records[0]['attempts'],'10')
        self.assertEqual(records[0]['pass_rate'],'0.9')
        self.assertIn('9/10 | 90.0% | 0 | 1 | 90.0%', '\n'.join(markdown(performance)))

    def test_conditional_subset_recomputes_even_when_no_attempt_is_retained(self):
        from mechanics_eval.conditional import cohort
        report={'mechanic_performance':summarize([case('table_leave_return','pass',0)])}
        selected=cohort(report,[],True)
        self.assertEqual(selected['mechanic_performance'],{'scenarios':[],'atomic':[]})
        self.assertEqual(selected['scenario_aggregate']['n'],0)
