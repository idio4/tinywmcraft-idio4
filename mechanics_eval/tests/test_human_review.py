"""Offline review contracts; human labels below are synthetic test fixtures only."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image
from mechanics_eval.evidence import digest,dump_json
from mechanics_eval.audit.human import build_items,export_review,validate_labels,summarize_review


def stage(name,status='unknown',field='relative_displacement',t=2,**extra):
    return dict(name=name,mechanic=name,category='Movement',status=status,required=True,timestep=t,
                facts=[dict(field=field,expected=[0,0],observed=None,status=status,timestep=t,reason='test_unknown')],**extra)


class HumanReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.root=Path(self.tmp.name);self.paths=[]
        self.results=[]
        for n in range(18):
            case=self.root/'raw'/f'case{n}';case.mkdir(parents=True)
            frames=np.random.default_rng(n).integers(0,255,(3,63,63,3),dtype=np.uint8)
            arrays=dict(context=frames[:1],context_actions=np.array([0]),b0_generated=frames[1:],b1_generated=frames[1:],
                        b0_reference=frames[1:],b1_reference=frames[1:],b0_actions=np.array([8,3]),b1_actions=np.array([0,0]),
                        b0_done_reference=np.array([False,True]),b1_done_reference=np.array([False,True]))
            np.savez_compressed(case/'trajectory.npz',**arrays)
            dump_json(case/'metadata.json',dict(seed=n,model_seed=100+n,spec={'name':'table_leave_return'},trajectory_sha256=digest(case/'trajectory.npz')))
            obj=stage('object_persistence.after_return','fail','tile:tracked_table',category_unused=True)
            obj.update(category='Object persistence',object_evidence=dict(value=2,coordinate=[3,5],target_coordinate=[3,5],
                localization_status='known',target_visibility='visible',estimated_player_position=[0,0],tile_reader_reason='test'))
            if n==0:obj['object_evidence'].update(coordinate=None,target_coordinate=None,localization_status='unknown',target_visibility='unknown')
            rows=[stage('movement.displacement'),obj]
            if n>=6:
                rows=[stage('movement.direction',field='direction')]
                dump_json(case/'eligibility.json',dict(decision='REJECTED_BLOCKED_ROUTE',reason='positive_newly_revealed_blocker',
                    witnesses=[dict(timestep=2,world_relative_coordinate=[-4,0],measured_step=[0,0],visibility_history=[dict(timestep=1,screen_coordinate=[2,4],tile=5),dict(timestep=2,screen_coordinate=[2,4],tile=5)])]))
            row=dict(scenario='table_leave_return',category='placement',seed=n,scope='static',evidence=str(case),
                measurements_by_branch=[rows,[stage('object_persistence.after_return','not_applicable','applicability')]],
                branch_outcomes=[deepcopy(rows[0]),stage('alternative','unknown',field='wood')],
                diagnostics_by_branch=[{'hud':stage('inventory_delta','unknown',field='wood')},{}])
            self.results.append(row)
        for name,cohort in [('conditional','CONDITIONAL_SELECTED_ELIGIBLE'),('ordinary','UNCONDITIONAL_FIRST_SAMPLE')]:
            folder=self.root/name;folder.mkdir();rows=deepcopy(self.results)
            if name=='ordinary':
                for row in rows:
                    for b in row['measurements_by_branch']:
                        for r in b:r['required']=False
            report=dict(samples=rows,cohort=cohort,metric_version='test-v1',scorer_sha256='unchanged')
            path=folder/'results.json';dump_json(path,report);self.paths.append(path)
            for n,row in enumerate(rows):
                cache=folder/f'case{n}';cache.mkdir()
                reading=dict(tiles={},measurements={},wood=None)
                dump_json(cache/'measurements.json',dict(context_last=reading,generated=[[reading,reading],[reading,reading]]))

    def tearDown(self):self.tmp.cleanup()

    def label(self,manifest,item=None,question=None,decision='CONFIRM_AUTOMATIC'):
        i=item or manifest['items'][0];q=question or i['questions'][0]
        return dict(item_id=i['id'],question_id=q['id'],evidence_hash=i['evidence_hash'],automatic_records_hash=q['automatic_records_hash'],
            scorer_versions=q['scorer_versions'],original_automatic={r['id']:r['automatic']['status'] for r in q['records']},
            reviewer='synthetic-test',decision=decision,human_observed=None,comment='',human_verdicts={},observation_agreement='not_assessed')

    def test_all_unknowns_both_branches_and_diagnostics_are_included(self):
        items,_,_,unknown=build_items(self.paths,0,0)
        qs={q['id'] for i in items for q in i['questions']}
        self.assertTrue({r[-1] for r in unknown}<=qs)
        self.assertTrue(any(i['branch']==1 and 'A' in i['groups'] for i in items))
        self.assertTrue(any('inventory_delta' in q['mechanics'] for i in items for q in i['questions']))

    def test_all_six_return_failures_and_twelve_rejections(self):
        items,*_=build_items(self.paths,0,0)
        failed={i['engine_seed'] for i in items for q in i['questions'] for r in q['records'] if r['automatic'].get('name')=='object_persistence.after_return' and r['automatic']['status']=='fail'}
        self.assertEqual(failed,set(range(6)))
        self.assertEqual(sum(i['family']=='departure_eligibility' for i in items),12)

    def test_reused_sources_and_policy_variants_share_visual_items(self):
        items,*_=build_items(self.paths)
        keys=[(i['evidence_hash'],i['family']) for i in items];self.assertEqual(len(keys),len(set(keys)))
        q=next(q for i in items for q in i['questions'] if 'movement.displacement' in q['mechanics'])
        self.assertEqual({a['cohort'] for r in q['records'] for a in r['associations']},{'UNCONDITIONAL_FIRST_SAMPLE','CONDITIONAL_SELECTED_ELIGIBLE'})
        self.assertEqual(len(q['records']),2)  # required policy differs; primary aliases join one record.

    def test_stable_ids_and_record_hashes_independent_of_source_order(self):
        a,*_=build_items(self.paths);b,*_=build_items(list(reversed(self.paths)))
        signature=lambda items:{i['id']:{q['id']:q['automatic_records_hash'] for q in i['questions']} for i in items}
        self.assertEqual(signature(a),signature(b))

    def test_lossless_frames_alignment_terminal_and_known_unknown_crops(self):
        before={p:digest(p) for p in self.paths};m=export_review(self.root/'export',self.paths,fail_limit=0,pass_limit=0)
        self.assertEqual(before,{p:digest(p) for p in self.paths})
        for item in m['items']:
            frames=item['trajectory']['frames'];self.assertEqual([f['action'] for f in frames],[0,8,3] if item['branch']==0 else [0,0,0])
            self.assertEqual(len(frames),3);self.assertTrue(frames[-1]['real_terminal'])
            with np.load(Path(item['trajectory']['source'])/'trajectory.npz') as a:
                np.testing.assert_array_equal(np.asarray(Image.open(self.root/'export'/frames[2]['generated'])),a[f'b{item["branch"]}_generated'][1])
            for q in item['questions']:
                if 'object_persistence.after_return' in q['mechanics'] and item['branch']==0:
                    v=q['visuals']['2']
                    if item['engine_seed']==0:self.assertNotIn('target_crop',v)
                    else:
                        frame=np.asarray(Image.open(self.root/'export'/frames[2]['generated']))
                        np.testing.assert_array_equal(np.asarray(Image.open(self.root/'export'/v['target_crop'])),frame[21:28,35:42])
        self.assertEqual(json.loads((self.root/'export'/'labels-template.json').read_text())['labels'],[])

    def test_import_round_trip_and_reject_mismatches_atomically(self):
        m=export_review(self.root/'export',self.paths,smoke_limit=2);label=self.label(m)
        doc=dict(schema_version=1,manifest_id=m['manifest_id'],labels=[label]);self.assertEqual(validate_labels(m,json.loads(json.dumps(doc))),[label])
        for field,value in [('item_id','wrong'),('evidence_hash','wrong'),('scorer_versions',['wrong']),('original_automatic',{}),('human_verdicts',{'wrong':'pass'}),('decision','wrong')]:
            broken=deepcopy(doc);broken['labels'][0][field]=value
            with self.assertRaises(ValueError,msg=field):validate_labels(m,broken)
        with self.assertRaises(ValueError):validate_labels(m,dict(doc,labels=[label,label]))

    def test_summary_empty_labels_never_claims_accuracy_or_verification(self):
        m=export_review(self.root/'export',self.paths,smoke_limit=2)
        stats=summarize_review(self.root/'export/review_manifest.json',self.root/'export/labels-template.json',self.root/'summary')
        self.assertEqual(stats['reviewed_questions'],0);self.assertEqual(stats['fully_reviewed_items'],0)
        self.assertEqual(stats['binary_verdict_denominator'],0);self.assertIsNone(stats['false_positive_rate'])

    def test_summary_deduplicates_aliases_and_separates_policy_denominators(self):
        m=export_review(self.root/'export',self.paths,smoke_limit=2);i=m['items'][0];q=i['questions'][0]
        l=self.label(m,i,q);l['observation_agreement']='agree';dump_json(self.root/'labels.json',dict(schema_version=1,manifest_id=m['manifest_id'],labels=[l]))
        s=summarize_review(self.root/'export/review_manifest.json',self.root/'labels.json',self.root/'summary')
        self.assertEqual(s['reviewed_questions'],1);self.assertEqual(s['observation_agreement_denominator'],1)
        self.assertTrue(all(len({x['question_id'] for x in p['unique_question_verdicts']})==len(p['unique_question_verdicts']) for p in s['policies']))

    def test_missing_files_are_reported_exactly(self):
        p=self.root/'raw/case0/trajectory.npz';p.unlink()
        with self.assertRaisesRegex(FileNotFoundError,str(p)):build_items(self.paths)

    def test_return_prerequisite_policy_differences_are_one_physical_question(self):
        path=self.paths[1];r=json.loads(path.read_text());raw=r['samples'][1]['measurements_by_branch'][0][1]
        raw.update(status='not_tested',facts=[],reason='prerequisite_unknown');dump_json(path,r)
        items,*_=build_items(self.paths,0,0)
        item=next(i for i in items if i['engine_seed']==1 and i['branch']==0 and i['family']=='object.crafting_table')
        questions=[q for q in item['questions'] if 'object_persistence.after_return' in q['mechanics']]
        self.assertEqual(len(questions),1)
        self.assertEqual({r['automatic']['status'] for r in questions[0]['records']},{'fail','not_tested'})

    def test_rejection_review_does_not_expose_future_return(self):
        m=export_review(self.root/'export',self.paths,smoke_limit=1)
        self.assertEqual(m['items'][0]['family'],'departure_eligibility')
        item=m['items'][0];self.assertTrue(all(f['timestep']<=7 for f in item['trajectory']['frames']))
        self.assertTrue(item['questions'][0]['blocker_visuals'])
        self.assertFalse(any('return_to_origin' in q['mechanics'] for q in item['questions']))

    def test_hud_crops_are_original_pixels_and_all_relevant_times_exist(self):
        m=export_review(self.root/'export',self.paths,fail_limit=0,pass_limit=0)
        found=0
        for i in m['items']:
            for q in i['questions']:
                if any('wood' in v['hud'] for v in q['visuals'].values()):
                    self.assertEqual(set(q['visuals']),{'0','1','2'})
                    with np.load(Path(i['trajectory']['source'])/'trajectory.npz') as a:
                        image=np.asarray(Image.open(self.root/'export'/q['visuals']['1']['hud']['wood']['path']))
                        np.testing.assert_array_equal(image,a[f'b{i["branch"]}_generated'][0][49:56,35:42])
                    found+=1
        self.assertGreater(found,0)

    def test_verdict_errors_count_physical_questions_once_and_exclude_unobservable(self):
        m=export_review(self.root/'export',self.paths,fail_limit=0,pass_limit=0)
        i=next(i for i in m['items'] if i['engine_seed']==1 and i['branch']==0 and i['family']=='object.crafting_table')
        q=next(q for q in i['questions'] if 'object_persistence.after_return' in q['mechanics'])
        l=self.label(m,i,q,'READER_ERROR');l['human_verdicts']={r['id']:'pass' for r in q['records']}
        doc=dict(schema_version=1,manifest_id=m['manifest_id'],labels=[l]);dump_json(self.root/'labels.json',doc)
        s=summarize_review(self.root/'export/review_manifest.json',self.root/'labels.json',self.root/'one')
        self.assertEqual(s['false_negative_fails'],1);self.assertEqual(s['binary_verdict_denominator'],1)
        self.assertEqual(s['rejection_decisions']['unreviewed'],12);self.assertEqual(len(s['returned_table_failures']),6)
        l['decision']='GENUINELY_UNOBSERVABLE';dump_json(self.root/'labels.json',doc)
        s=summarize_review(self.root/'export/review_manifest.json',self.root/'labels.json',self.root/'two')
        self.assertEqual(s['binary_verdict_denominator'],0)
