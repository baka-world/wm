"""Fixed nonzero-tick audit gates, no model training."""
import unittest
from experiments.world_model_integration.dynamic_value_audit import audit as a

class AuditContracts(unittest.TestCase):
    def test_categories(self):
        self.assertEqual(a.classify(False,True,True,False),"S0")
        self.assertEqual(a.classify(True,False,True,False),"S1")
        self.assertEqual(a.classify(True,True,False,False),"S2")
        self.assertEqual(a.classify(True,True,True,False),"S3")
        self.assertEqual(a.classify(True,True,False,True),"S4")
    def test_no_independent_groups(self):
        cases=a.frozen_development()
        self.assertEqual(len(cases),60)
        self.assertNotIn("test",{x for x,_ in cases})
        self.assertNotIn(771000,{s["seed"] for _,s in cases})
    def test_plan_matching_order_not_fake_choice(self):
        from lab_transport.transition_schema import Assignment
        p=(Assignment("r0","s0"),Assignment("r1","s1"))
        self.assertEqual(a.norm(p),a.norm(tuple(reversed(p))))
    def test_selection_no_true_cost(self):
        state=lambda seed,tick,category:{"seed":seed,"tick":tick,"category":category,
                "distinct_legal_plans":2,"id":str(seed)+":"+str(tick)}
        cases=[{"states":[state(1,1,"S2"),state(1,2,"S4"),state(1,3,"S3"),
                           state(2,1,"S3"),state(3,1,"S1")]}]
        selected,stats=a.select_states(cases,{"max_states_per_scenario":2,"max_selected_states":12})
        self.assertEqual([x["id"] for x in selected],["1:2","1:3","2:1"])
        self.assertEqual(stats["eligible_count"],4)
    def test_future_classification_is_single_realization(self):
        self.assertEqual(a.json.loads(a.PROTOCOL.read_text())["paired_futures"],
                         "single_original_seed_only")

if __name__=="__main__":
    unittest.main()
