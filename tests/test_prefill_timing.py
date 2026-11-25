import pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from prefix_scheduler.prefill_timing import PrefillTimer
class Event:
    def __init__(self,enable_timing): self.stream=None; self.synced=False
    def record(self,stream): self.stream=stream
    def synchronize(self): self.synced=True
    def elapsed_time(self,other):
        assert other.synced and self.stream is other.stream
        return 2.5
class CUDA: Event=Event
class TestPrefillTimer(unittest.TestCase):
    def test_single_stream_and_synchronization(self):
        t=PrefillTimer(CUDA,batch_size=1,chunked=False,overlap=False); stream=object()
        t.start(stream); t.stop(stream); self.assertEqual(t.milliseconds(),2.5)
    def test_disallowed_and_order(self):
        for args in [(2,False,False),(1,True,False),(1,False,True)]:
            with self.assertRaises(ValueError): PrefillTimer(CUDA,batch_size=args[0],chunked=args[1],overlap=args[2])
        t=PrefillTimer(CUDA,batch_size=1,chunked=False,overlap=False)
        with self.assertRaises(RuntimeError): t.milliseconds()
