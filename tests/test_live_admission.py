import json,pathlib,sys,tempfile,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from prefix_scheduler.live_admission import DependencySnapshot,LiveAdmission

class Req:
    def __init__(self,rid,length): self.rid,self.origin_input_ids=rid,[0]*length

class Predictor:
    def predict(self,length,cached): return None if cached==0 else float(cached*2)

class LiveAdmissionTests(unittest.TestCase):
    def test_oldest_then_cache_score_and_confirmed_cycle(self):
        policy=LiveAdmission('k-lpm',k=3)
        a,b,c=Req('a',100),Req('b',100),Req('c',100)
        policy.order([b,a,c],lambda r:{'a':10,'b':80,'c':50}[r.rid])
        self.assertEqual(policy.pending['ranked_ids'],['b','a','c'])
        # Scheduling attempts do not advance the fairness cycle.
        policy.order([a,b,c],lambda r:0)
        self.assertEqual(policy.pending['slot'],0)
        policy.admitted([a])
        self.assertEqual(policy.admissions,1)
        queue=[a,b,c]; policy.order(queue,lambda r:{'a':10,'b':80,'c':50}[r.rid])
        self.assertEqual(policy.pending['reason'],'k-lpm')
        self.assertEqual([r.rid for r in queue],['b','c','a'])

    def test_calibrated_uses_whole_queue_baseline_fallback(self):
        policy=LiveAdmission('calibrated',k=2,predictor=Predictor())
        a,b=Req('a',100),Req('b',100)
        policy.order([a,b],lambda r:{'a':20,'b':40}[r.rid]); policy.admitted([a])
        queue=[a,b]; policy.order(queue,lambda r:0 if r is a else 40)
        self.assertEqual(policy.pending['reason'],'baseline_fallback')
        self.assertEqual([r.rid for r in queue],['b','a'])

    def test_rejects_non_single_actual_admission(self):
        with self.assertRaises(RuntimeError):LiveAdmission('fcfs').admitted([Req('a',1),Req('b',1)])

    def test_dependency_score_uses_materialized_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path=pathlib.Path(directory)/'graph.json'
            path.write_text(json.dumps({'requests':[
                {'request_id':'a','status':'running','parent_ids':[]},
                {'request_id':'b','status':'running','parent_ids':[]},
                {'request_id':'a-child-1','status':'blocked','parent_ids':['a']},
                {'request_id':'a-child-2','status':'blocked','parent_ids':['a']},
            ]}))
            policy=LiveAdmission('dependency-aware',k=2,predictor=Predictor(),
                                 dependency_snapshot=DependencySnapshot(path),unlock_bonus_ms=10)
            a,b=Req('a',100),Req('b',100)
            policy.order([a,b],lambda r:5 if r is a else 9);policy.admitted([a])
            queue=[a,b];policy.order(queue,lambda r:5 if r is a else 9)
            self.assertEqual(policy.pending['reason'],'dependency_score')
            self.assertEqual(policy.pending['candidates'][0]['unlock_count'],2)
            self.assertEqual([r.rid for r in queue],['a','b'])

    def test_dependency_snapshot_unavailable_falls_back_to_baseline(self):
        policy=LiveAdmission('dependency-aware',k=2,predictor=Predictor(),
                             dependency_snapshot=DependencySnapshot('/does/not/exist'),unlock_bonus_ms=1)
        a,b=Req('a',100),Req('b',100)
        policy.order([a,b],lambda r:1 if r is a else 9);policy.admitted([a])
        queue=[a,b];policy.order(queue,lambda r:1 if r is a else 9)
        self.assertEqual(policy.pending['reason'],'baseline_fallback')
        self.assertEqual([r.rid for r in queue],['b','a'])
