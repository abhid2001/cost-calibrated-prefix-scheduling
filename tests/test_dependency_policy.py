import pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from prefix_scheduler.core import Request
from prefix_scheduler.dependencies import immediate_unlocks
from prefix_scheduler.policy import AdmissionPolicy

def r(id,status='ready',parents=None,t='1'):
    return Request('q',id,'x','0','0',status=status,ready_at=t if status=='ready' else None,input_token_ids=[1]*10,parent_ids=parents or [])
class Predictor:
    def predict(self, length, cached): return float(cached)
class DependencyTests(unittest.TestCase):
    def test_unlock_requires_all_other_parents(self):
        a=r('a'); b=r('b','completed'); c=r('c','blocked',['a','b']); nodes=[a,b,c]
        self.assertEqual(immediate_unlocks(a,nodes),1)
        b.status='failed'; self.assertEqual(immediate_unlocks(a,nodes),0)
        self.assertEqual(immediate_unlocks(a,[a,c]),0)
    def test_bonus_changes_cache_slot_not_oldest_slot(self):
        a=r('a',t='1');b=r('b',t='2');sibling=r('sib','completed');child=r('eval','blocked',['b','sib']);nodes=[a,b,sibling,child]
        p=AdmissionPolicy('dependency-aware',2,predictor=Predictor(),unlock_bonus_ms=4)
        match=lambda x: 6 if x is a else 3
        unlock=lambda x: immediate_unlocks(x,nodes)
        d=p.select(nodes,match,unlocks=unlock);self.assertIs(d.request,a);p.admitted(d)
        d=p.select(nodes,match,unlocks=unlock);self.assertIs(d.request,b);self.assertEqual(d.reason,'dependency_score');self.assertEqual(d.diagnostics['b']['unlock_count'],1)
    def test_unavailable_predictions_fall_back_uniformly(self):
        a=r('a',t='1');b=r('b',t='2');p=AdmissionPolicy('dependency-aware',2,predictor=Predictor(),unlock_bonus_ms=4)
        d=p.select([a,b],lambda r:1,unlocks=lambda r:0);p.admitted(d)
        d=p.select([a,b],lambda r:1);self.assertEqual(d.reason,'baseline_fallback');self.assertIs(d.request,a)
    def test_reject_invalid_bonus(self):
        with self.assertRaises(ValueError):AdmissionPolicy('dependency-aware',unlock_bonus_ms=-1)
