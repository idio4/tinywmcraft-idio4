"""Per-mechanic reporting uses existing selected-branch verdicts without pooling."""
import unittest
from mechanics_eval.performance import summarize


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
