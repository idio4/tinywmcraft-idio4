"""Per-mechanic reporting uses existing selected-branch verdicts without pooling."""
import unittest
from copy import deepcopy
from mechanics_eval.performance import summarize, scenario_verdict


def case(name, status, seed, *, stage="crafting.tool_creation",
         stage_status="pass", scope="static", ordinary=None):
    result={"status":status}
    entry={"name":stage, "category":"Crafting", "required":True,
           "status":stage_status, "eligible":stage_status in ("pass","fail","unknown")}
    row={"scenario":name,"category":"crafting","scope":scope,"seed":seed,"model_sample":0,
         "scenario_results_by_branch":[result,{"status":"pass"}],
         "measurements_by_branch":[[entry],[]]}
    if ordinary is not None:
        row["ordinary_scenario_results_by_branch"]=[{"status":ordinary},{"status":"pass"}]
        row["ordinary_measurements_by_branch"]=[[entry],[]]
    return row


class MechanicFrequencyTests(unittest.TestCase):
    def test_pass_rate_is_per_mechanic_and_unknown_is_not_success(self):
        rows=[case("craft_stone_pickaxe","pass",i) for i in range(9)]
        rows+=[case("craft_stone_pickaxe","unknown",9,stage_status="unknown")]
        rows+=[case("craft_wood_pickaxe","fail",i,stage_status="fail") for i in range(5)]
        rows+=[case("craft_wood_pickaxe","pass",i) for i in range(5,10)]
        result=summarize(rows)
        by_name={r["name"]:r for r in result["scenarios"]}
        self.assertEqual((by_name["craft_stone_pickaxe"]["passed"],by_name["craft_stone_pickaxe"]["unknown"],
                          by_name["craft_stone_pickaxe"]["pass_rate"]), (9,1,.9))
        self.assertEqual(by_name["craft_stone_pickaxe"]["decision_coverage"],.9)
        self.assertEqual(by_name["craft_wood_pickaxe"]["pass_rate"],.5)
        self.assertEqual(by_name["craft_wood_pickaxe"]["engine_seed_count"],10)

    def test_atomic_not_tested_does_not_fake_pass_and_reports_eligibility(self):
        rows=[case("place_table","pass",0,stage="object_persistence.table",stage_status="pass"),
              case("place_table","fail",1,stage="object_persistence.table",stage_status="fail"),
              case("place_table","fail",2,stage="object_persistence.table",stage_status="not_tested")]
        a=summarize(rows)["atomic"][0]
        self.assertEqual((a["passed"],a["failed"],a["eligible"],a["not_tested"]),(1,1,2,1))
        self.assertEqual(a["pass_rate"],.5)
        self.assertAlmostEqual(a["eligibility_coverage"],2/3)

    def test_conditional_first_and_retained_can_be_reported_independently(self):
        rows=[case("table_leave_return","fail",0,ordinary="pass"),
              case("table_leave_return","fail",1,ordinary="unknown")]
        self.assertEqual(summarize(rows)["scenarios"][0]["pass_rate"],0)
        self.assertEqual(summarize(rows,ordinary=True)["scenarios"][0]["pass_rate"],.5)
        self.assertEqual(summarize(rows,ordinary=True)["scenarios"][0]["unknown"],1)

    def test_empty_scope_and_invalid_status(self):
        self.assertEqual(summarize([]),{"scenarios":[],"atomic":[]})
        with self.assertRaises(ValueError):
            summarize([case("bad","not_tested",0)])

    def test_nine_pass_one_failure_and_single_attempt(self):
        rows=[case('craft_stone_pickaxe','pass',i) for i in range(9)]
        rows.append(case('craft_stone_pickaxe','fail',9))
        item=summarize(rows)['scenarios'][0]
        self.assertEqual((item['passed'],item['failed'],item['pass_rate'],item['decision_coverage']),(9,1,.9,1))
        single=summarize(rows[:1])['scenarios'][0]
        self.assertEqual((single['attempts'],single['pass_rate'],single['engine_seed_count']),(1,1,1))

    def test_fallbacks_do_not_eagerly_require_legacy_outcome(self):
        row={'scenario':'test','scenario_result':{'status':'fail'}}
        self.assertEqual(summarize([row])['scenarios'][0]['failed'],1)
        self.assertEqual(scenario_verdict({'branch_outcomes':[{'status':'unknown'},{'status':'pass'}]})['status'],'unknown')
        self.assertEqual(scenario_verdict({'outcome':{'status':'pass'}})['status'],'pass')
        self.assertEqual(scenario_verdict(case('test','fail',0),ordinary=True)['status'],'fail')

    def test_required_diagnostics_and_inapplicable_are_distinct(self):
        rows=[case('test','pass',0),case('test','fail',1,stage_status='not_tested'),
              case('test','unknown',2,stage_status='not_applicable')]
        for row in rows:
            diagnostic=dict(row['measurements_by_branch'][0][0],required=False,status='fail')
            row['measurements_by_branch'][0].append(diagnostic)
        atoms=summarize(rows)['atomic']
        required=next(a for a in atoms if a['required'])
        diagnostic=next(a for a in atoms if not a['required'])
        self.assertEqual((required['instances'],required['applicable'],required['eligible']),(3,2,1))
        self.assertEqual((required['not_tested'],required['not_applicable'],required['eligibility_coverage']),(1,1,.5))
        self.assertEqual((diagnostic['failed'],diagnostic['pass_rate']),(3,0))
        self.assertEqual(summarize(rows)['scenarios'][0]['passed'],1)
        for status in ('not_tested','not_applicable'):
            atom=summarize([case('test','fail',0,stage_status=status)])['atomic'][0]
            self.assertIsNone(atom['pass_rate'])
            self.assertEqual(atom['eligible'],0)

    def test_scopes_and_repeated_draws_remain_distinct(self):
        rows=[case('test','pass',0),case('test','fail',0),case('test','unknown',1)]
        rows[1]['model_sample']=1
        rows.append(case('test','fail',2,scope='dynamic'))
        result=summarize(rows)
        by_scope={r['scope']:r for r in result['scenarios']}
        static=by_scope['static']
        self.assertEqual((static['attempts'],static['engine_seed_count'],static['model_sample_count']),(3,2,2))
        self.assertEqual((static['samples_per_seed_min'],static['samples_per_seed_max']),(1,2))
        self.assertEqual(by_scope['dynamic']['attempts'],1)
        self.assertEqual({a['scope'] for a in result['atomic']},{'static','dynamic'})

    def test_gallery_summary_uses_same_fallbacks_and_keeps_input_unchanged(self):
        from mechanics_eval.gallery import summary
        rows=[case('test','fail',0,ordinary='unknown'),case('test','pass',1,scope='dynamic')]
        before=deepcopy(rows)
        result=summary(rows,ordinary=True)
        self.assertEqual(result['all']['scenarios']['unknown'],1)
        self.assertEqual(result['static']['mechanics']['scenarios'][0]['unknown'],1)
        self.assertEqual(result['dynamic']['mechanics']['scenarios'][0]['passed'],1)
        self.assertEqual(rows,before)
        legacy={'scenario':'legacy','scenario_result':{'status':'pass'}}
        self.assertEqual(summary([legacy])['all']['scenarios']['pass'],1)

    def test_shared_atomic_names_do_not_pool_different_scenarios(self):
        rows=[case('craft_wood_pickaxe','pass',0),case('craft_stone_pickaxe','fail',0,stage_status='fail')]
        atoms={r['scenario']:r for r in summarize(rows)['atomic']}
        self.assertEqual(len(atoms),2)
        self.assertEqual(atoms['craft_wood_pickaxe']['pass_rate'],1)
        self.assertEqual(atoms['craft_stone_pickaxe']['pass_rate'],0)
