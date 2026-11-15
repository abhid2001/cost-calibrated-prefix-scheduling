import pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from prefix_scheduler.calibration import fit,validate
from prefix_scheduler.core import Request
from prefix_scheduler.policy import AdmissionPolicy
class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.data=[{'length':100,'cached':20,'cold_prefill_ms':10,'warm_prefill_ms':8,'warm_cached_tokens':20}, {'length':200,'cached':100,'cold_prefill_ms':20,'warm_prefill_ms':12,'warm_cached_tokens':100}]
    def test_fit_validate_and_out_of_range(self):
        m=fit(self.data); self.assertEqual(m.predict(100,20),2);self.assertIsNone(m.predict(300,100)); self.assertEqual(validate(m,self.data)['mae_ms'],0)
    def test_bad_cache_rejected(self):
        x=dict(self.data[0],warm_cached_tokens=19)
        with self.assertRaises(ValueError):fit([x])
    def test_policy_uses_prediction_and_falls_back_uniformly(self):
        a=Request('q','a','x','0','0',status='ready',ready_at='1',input_token_ids=[1]*100)
        b=Request('q','b','x','0','0',status='ready',ready_at='2',input_token_ids=[1]*200)
        p=AdmissionPolicy('calibrated',2,predictor=fit(self.data));d=p.select([a,b],lambda r:20 if r==a else 100);p.admitted(d)
        d=p.select([a,b],lambda r:20 if r==a else 100);self.assertEqual(d.request,b);self.assertEqual(d.reason,'predicted_saving');p.rejected(d)
        d=p.select([a,b],lambda r: 20 if r==a else 5);self.assertEqual(d.reason,'baseline_fallback');self.assertEqual(d.request,a)
