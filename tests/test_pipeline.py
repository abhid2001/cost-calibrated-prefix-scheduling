import pathlib, sys, unittest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from prefix_scheduler.core import Request
from prefix_scheduler.policy import AdmissionPolicy
from prefix_scheduler.reasoning import Controller

def request(name, t, status='ready'):
    return Request('q',name,'test','0','0',status=status,ready_at=t,input_token_ids=[1]*10)

class PolicyTests(unittest.TestCase):
    def test_cycle_ties_and_rejection(self):
        p=AdmissionPolicy('k-lpm',k=2); a=request('a','1'); b=request('b','2'); blocked=request('x','0','blocked')
        d=p.select([a,b,blocked],lambda r: {'a':1,'b':9}[r.request_id]); self.assertEqual(d.request,a)
        p.rejected(d); self.assertEqual(p.admissions,0)
        d=p.select([a,b],lambda r: 0); p.admitted(d)
        d=p.select([a,b],lambda r: {'a':1,'b':9}[r.request_id]); self.assertEqual(d.request,b); p.admitted(d)
        self.assertEqual(p.select([a,b],lambda r:0).request,a)
    def test_fcfs_k1_and_empty(self):
        for p in (AdmissionPolicy('fcfs'),AdmissionPolicy('k-lpm',1)):
            self.assertIsNone(p.select([],lambda r:0))
            a,b=request('a','1'),request('b','2'); d=p.select([a,b],lambda r:9 if r==b else 0); self.assertEqual(d.request,a)
    def test_score_ties_use_ready_time_then_id(self):
        p=AdmissionPolicy('k-lpm',2); a=request('b','1'); b=request('a','1')
        d=p.select([a,b],lambda r:5); p.admitted(d)
        d=p.select([a,b],lambda r:5); self.assertEqual(d.request.request_id,'a')
    def test_ratio_differs_from_token_count(self):
        a=request('a','1'); b=request('b','2'); b.input_token_ids=[1]*100
        p=AdmissionPolicy('k-lpm',2); d=p.select([a,b],lambda r:0); p.admitted(d)
        self.assertEqual(p.select([a,b],lambda r: 5 if r==a else 20).request,a)

class ReasoningTests(unittest.TestCase):
    def test_votes_and_stable_seeds(self):
        outputs=iter(['Final answer: 2','Final answer: 3','Final answer: 3'])
        c=Controller(lambda r,m,t: next(outputs)); result=c.self_consistency('q','question')
        self.assertEqual(result['answer'],'3'); self.assertEqual(len(c.records),3)
        self.assertEqual(len({r.sampling_seed for r in c.records}),3)
        self.assertEqual(c.records[0].sampling_seed,Controller(lambda *a:'').call('q',result['finished_at'],'sample1','x',[],[],0).sampling_seed)
    def test_all_invalid(self):
        c=Controller(lambda *args:'no answer'); self.assertEqual(c.self_consistency('q','?')['status'],'failed')
    def test_tot_dependencies_and_fallback(self):
        answers=iter(['a','b','bad selection','c','d','Selected: 2','Final answer: 9'])
        c=Controller(lambda *a: next(answers)); result=c.tot('q','?')
        self.assertEqual(result['answer'],'9'); self.assertEqual(len(c.records),7); self.assertEqual(result['evaluator_failures'],1)
        self.assertEqual(c.records[2].parent_ids,[c.records[0].request_id,c.records[1].request_id])
        self.assertEqual(c.records[3].parent_ids,[c.records[0].request_id,c.records[2].request_id])
        self.assertEqual(c.records[6].parent_ids,[c.records[4].request_id,c.records[5].request_id])
    def test_candidate_failure_terminates(self):
        def fail(*args): raise TimeoutError('timeout')
        c=Controller(fail); result=c.tot('q','?'); self.assertEqual(result['status'],'failed'); self.assertEqual(len(c.records),2)
