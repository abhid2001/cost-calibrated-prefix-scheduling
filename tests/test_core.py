import pathlib, sys, unittest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from prefix_scheduler.core import Request, normalize_number, parse_answer, prompt_for

class CoreTests(unittest.TestCase):
    def test_state_machine(self):
        r=Request('q','r','direct','a','b')
        with self.assertRaises(ValueError): r.transition('running')
        r.transition('ready','c'); r.transition('running','d'); r.transition('completed','e')
        self.assertEqual((r.ready_at,r.dispatched_at,r.admitted_at,r.finished_at),('c','d',None,'e'))
        with self.assertRaises(ValueError): r.transition('ready')
    def test_failure_from_blocked(self):
        r=Request('q','r','direct','a','b'); r.transition('failed','c')
        self.assertEqual(r.finished_at,'c')
    def test_answers(self):
        self.assertEqual(parse_answer('steps\nFinal answer: -1,234.50'), '-1234.5')
        self.assertEqual(parse_answer('#### 0'), '0')
        self.assertIsNone(parse_answer('steps 42'))
        self.assertIsNone(parse_answer('Final answer: 1 or 2'))
        self.assertEqual(normalize_number('100.00'), '100')
    def test_prompt_modes(self):
        self.assertNotEqual(prompt_for('q','direct')[0],prompt_for('q','cot')[0])
        with self.assertRaises(ValueError): prompt_for('q','other')
if __name__ == '__main__': unittest.main()
