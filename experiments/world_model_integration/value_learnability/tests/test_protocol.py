"""Frozen protocol, feature gate and grouped relative learning smoke tests."""
import json
import unittest
from pathlib import Path
from types import SimpleNamespace as N
import numpy as np
from experiments.world_model_integration.value_learnability import run as exp

class Tests(unittest.TestCase):
    def test_groups_and_zero_old_seeds(self):
        p=json.loads(exp.PROTOCOL.read_text())
        groups=exp.frozen_specs(p)
        self.assertEqual({k:len(v) for k,v in groups.items()},{'train':48,'validation':12,'test':16})
        self.assertEqual(len({g['seed'] for v in groups.values() for g in v}),76)
        self.assertTrue(all(g['seed'] not in (43011,43015,43019) for v in groups.values() for g in v))
        self.assertFalse(set(g['layout'] for g in groups['train']) & set(g['layout'] for g in groups['test']))

    def test_public_feature_does_not_consult_simulator(self):
        geo=N(distance=lambda a,b:abs(a[0]-b[0])+abs(a[1]-b[1]),unreachable=999999)
        robot=N(id='r',position=(0,0),task_id=None,service_elapsed=None)
        sample=N(id='s',pickup='a',dropoff='b',status='waiting',workflow='w',stage_index=0)
        station=lambda id,pos:N(id=id,position=pos,resource_occupied=0,expected_service_ticks=3,device_available=True,available=True)
        obs=N(tick=0,robots=(robot,),samples=(sample,),stations=(station('a',(1,0)),station('b',(4,0))))
        resource=N(id='res',capacity=1)
        rules=N(resources=(resource,),station_resources={'a':resource,'b':resource},workflows={'w':(('a','b'),)})
        plan=(N(robot_id='r',sample_id='s'),)
        f=exp.base_features(obs,plan,geo,rules)
        self.assertEqual(len(f),len(exp.BASE_KEYS))
        self.assertEqual(f[exp.BASE_KEYS.index('sum_approach')],1)
        with self.assertRaises(ValueError):exp.base_features(obs,plan+plan,geo,rules)

    def test_group_centered_fit_recover_ordering(self):
        groups=[]
        for i in range(12):
            rows=[]
            for j in range(3):
                x=[0.]*len(exp.BASE_KEYS)
                x[exp.BASE_KEYS.index('sum_approach')]=j
                rows.append({'base_features':x,'label_centered':j-1,
                    'rule_rollout':[0.]*5,'v4_rollout':[0.]*5})
            groups.append({'rows':rows})
        m=exp.fit(groups,'current',1.0)
        p=exp.model_predict(groups[0],m)
        self.assertEqual(exp.score_order(p),[0,1,2])

if __name__=='__main__':unittest.main()
