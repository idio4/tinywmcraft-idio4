from mechanics_eval.tests.paths import RUNS,OUTPUT
"""Media-only filtering over saved RGB and scores; no inference or rescoring."""
import copy
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.parse import unquote,urlsplit
from mechanics_eval.evidence import dump_json,digest
from mechanics_eval.gifs import export_gifs,gif_enabled

ROOT=(RUNS/'scores')


class Links(HTMLParser):
    def __init__(self):super().__init__();self.urls=[];self.cards=0;self.images=0
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='article':self.cards+=1
        if tag=='img':self.images+=1
        for key in ('src','href'):
            if attrs.get(key):self.urls.append(attrs[key])


class GalleryTests(unittest.TestCase):
    def fixture(self,root,conditional=False):
        source=root/'evidence';source.mkdir()
        if not ROOT.exists():self.skipTest('saved Level 1 evidence unavailable')
        if conditional:
            report=json.loads((ROOT/'conditional-gallery-results.json').read_text())
            names=['table_leave_return_s0_m0','table_leave_return_s1_m0','table_leave_return_s3_m1']
        else:
            report=json.loads((ROOT/'existing/results.json').read_text())
            names=['gather_wood_s0_m0','attack_zombie_s0_m0','death_s0_m0']
            controlled=json.loads((ROOT/'controlled/results.json').read_text())
            report['samples'].extend([r for r in controlled['samples'] if Path(r['evidence']).name=='table_leave_return_s6_m0'])
            names.append('table_leave_return_s6_m0')
        report['samples']=[r for r in report['samples'] if Path(r['evidence']).name in names]
        for row in report['samples']:
            src=Path(row['evidence']);shutil.copytree(src,source/src.name)
        dump_json(source/'manifest.json',dict(cases=names,config=dict(suite='saved-test')))
        report.pop('evidence_manifest_sha256',None)
        dump_json(root/'results.json',report)
        if conditional:
            for name in ('unconditional-results.json','conditional-results.json','summary.json'):shutil.copy2(ROOT/name,root/name)
        return source,root/'results.json'

    def assert_links(self,out):
        parser=Links();parser.feed((out/'index.html').read_text())
        for url in parser.urls:
            parsed=urlsplit(url)
            if not parsed.scheme and parsed.path:self.assertTrue((out/unquote(parsed.path)).exists(),url)
        return parser

    def test_all_modes_preserve_every_card_and_scientific_data(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);source,result=self.fixture(root)
            before=digest(result);scientific=None;card_count=None
            with patch('mechanics_eval.milestone.score_run',side_effect=AssertionError('export must not rescore')),patch('mechanics_eval.rollouts.ModuleAdapter',side_effect=AssertionError('export must not infer')):
                for mode in ('all','fail','unknown','nonpass','none'):
                    out=root/mode;export_gifs([source],[result],out,mode)
                    data=json.loads((out/'gallery.json').read_text())
                    self.assertEqual(data['design']['contexts'],[1])
                    self.assertEqual(data['design']['generated_steps'],[16])
                    self.assertEqual(digest(out/'suite0/results.json'),before)
                    self.assertEqual(digest(result),before)
                    summary=json.dumps(data['cohorts'],sort_keys=True)
                    self.assertEqual(summary,scientific or summary);scientific=summary
                    plain=copy.deepcopy(data['cards'])
                    for card in plain:
                        for view in card['views'].values():view.pop('gif')
                    self.assertEqual(plain,card_count or plain);card_count=plain
                    parser=self.assert_links(out);self.assertEqual(parser.cards,8)
                    for card in data['cards']:
                        for view in card['views'].values():
                            self.assertEqual(bool(view['gif']),gif_enabled(mode,view['status']))
                            if view['gif']:self.assertTrue((out/view['gif']).exists())
                    # Primary table creation PASS cannot hide a required persistence FAIL.
                    table=next(c for c in data['cards'] if c['case']=='table_leave_return_s6_m0' and c['branch']=='selected')
                    self.assertEqual(table['atomic_status'],'pass');self.assertEqual(table['views']['current']['status'],'fail')
                    if mode=='fail':self.assertTrue(table['views']['current']['gif'])
                    if mode=='none':
                        self.assertEqual(parser.images,0);self.assertFalse(list(out.rglob('*.gif')))
                        self.assertIn('GIF not exported for this view.',(out/'index.html').read_text())

    def test_conditional_views_and_branch_denominators_remain_separate(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);source,result=self.fixture(root,True)
            export_gifs([source],[result],root/'gallery','fail')
            data=json.loads((root/'gallery/gallery.json').read_text())
            self.assertEqual(len(data['cards']),6)
            card=next(c for c in data['cards'] if c['case']=='table_leave_return_s0_m0' and c['branch']=='selected')
            self.assertEqual(card['views']['current']['status'],'fail')
            self.assertEqual(card['views']['ordinary']['status'],'unknown')
            self.assertTrue(card['views']['current']['gif']);self.assertIsNone(card['views']['ordinary']['gif'])
            groups={c['id']:c for c in data['cohorts']}
            self.assertEqual(groups['first']['summary']['all']['scenarios']['n'],2)
            self.assertEqual(groups['selected']['summary']['all']['scenarios']['n'],2)
            self.assertEqual(groups['rejected']['summary']['all']['scenarios']['n'],1)
            self.assertEqual(groups['attempts']['summary']['all']['scenarios']['n'],3)
            self.assertEqual(data['conditional']['conditional']['n'],10)
            parser=self.assert_links(root/'gallery');self.assertGreater(parser.cards,0)

    def test_safe_html_and_cli_alias(self):
        import subprocess,sys
        for command in ('export-gallery','export-gifs'):
            result=subprocess.run([sys.executable,'-m','mechanics_eval.cli',command,'--help'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0);self.assertIn('--gifs',result.stdout)
        from mechanics_eval.gallery import write_page
        with TemporaryDirectory() as tmp:
            write_page(tmp,dict(cards=[],cohorts=[],source_note='</script><img src=x onerror=alert(1)>'))
            page=(Path(tmp)/'index.html').read_text()
            self.assertNotIn('</script><img src=x',page)

    def test_compact_example_gifs_are_present(self):
        """The repository ships two illustration GIFs, not a full generated report."""
        snapshot=Path(__file__).resolve().parents[1]/'example'
        self.assertTrue((snapshot/'LICENSE.txt').is_file())
        for filename in ('craft_wood_pickaxe_s0_m0.gif','craft_stone_pickaxe_s0_m0.gif'):
            image=snapshot/'media'/filename
            self.assertTrue(image.is_file(), str(image))
            self.assertIn(image.read_bytes()[:6], (b'GIF87a',b'GIF89a'))

    def test_supplement_labels_use_actual_scenarios(self):
        if not ROOT.exists():self.skipTest('saved evidence unavailable')
        for name,label in [('controlled','Controlled movement'),('table','Table placement')]:
            with TemporaryDirectory() as tmp:
                root=Path(tmp);source=root/'evidence';source.mkdir()
                report=json.loads((ROOT/name/'results.json').read_text())
                report['samples']=[r for r in report['samples'] if r['seed']==0]
                names=[]
                for row in report['samples']:
                    path=Path(row['evidence']);names.append(path.name);shutil.copytree(path,source/path.name)
                dump_json(source/'manifest.json',dict(cases=names,config=dict(suite='existing')))
                report.pop('evidence_manifest_sha256',None);dump_json(root/'results.json',report)
                export_gifs([source],[root/'results.json'],root/'gallery','none')
                data=json.loads((root/'gallery/gallery.json').read_text())
                self.assertTrue(data['cohorts'][0]['label'].startswith(label))
                self.assertIn(label+' results JSON',data['downloads'])
