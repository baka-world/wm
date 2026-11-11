"""Empirical hazards must use TRAIN labels and current effective progress only."""
import unittest
from experiments.world_model_integration.empirical_baselines import EmpiricalHazards

def row(group,elapsed,expected,y):
    return {"label":y,"event":"finished" if y else "continues",
            "query":{"robot_id":"R1","station_id":"A0"},
            "features":{"observed_facts":{
                "robots":[{"id":"R1","service_active_elapsed":{"known":True,"value":elapsed}}],
                "stations":[{"id":"A0","expected_service_ticks_public":{"known":True,"value":expected}}]}}}
class HazardTests(unittest.TestCase):
    def test_survival_hazards_do_not_use_target_label_at_inference(self):
        train=[row("train",e,4,int(e==3)) for e in (1,2,3)]*2
        hazard=EmpiricalHazards(train)
        self.assertAlmostEqual(hazard.p0,1/3)
        probe=row("holdout",3,4,0)
        for k in ("B0","B1","B2"):
            self.assertEqual(hazard.predict_one(probe,k),
                             hazard.predict_one({**probe,"label":1},k))
        self.assertGreater(hazard.predict_one(probe,"B1"),hazard.predict_one(row("",1,4,0),"B1"))
    def test_missing_history_refuses_not_a_synthetic_negative(self):
        train=[row("",1,4,0),row("",3,4,1)]
        h=EmpiricalHazards(train)
        x=row("",2,4,0);x["features"]["observed_facts"]["robots"][0]["service_active_elapsed"]={"known":False,"value":None}
        with self.assertRaisesRegex(ValueError,"unknown"):h.predict_one(x,"B2")
        with self.assertRaisesRegex(ValueError,"nonfactual"):
            EmpiricalHazards(train+[dict(row("",2,4,0),event="paused")])
if __name__=="__main__":unittest.main()
