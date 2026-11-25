import collections, http.server, json, pathlib, sys, threading, unittest
from unittest import mock
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from prefix_scheduler.core import Request, chat_token_ids, generate, normalize_number, parse_answer, prompt_for

class CoreTests(unittest.TestCase):
    def test_state_machine(self):
        r=Request('q','r','direct','a','b')
        with self.assertRaises(ValueError): r.transition('running')
        r.transition('ready','c'); r.transition('running','d'); r.transition('completed','e')
        self.assertEqual((r.ready_at,r.dispatched_at,r.admitted_at,r.finished_at),('c','d','d','e'))
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

class FakeTokenizer:
    """Mimics transformers 5: dict-like BatchEncoding unless return_dict=False."""
    def __init__(self, ids): self.ids, self.kwargs = ids, None
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict=True):
        self.kwargs = dict(tokenize=tokenize, add_generation_prompt=add_generation_prompt, return_dict=return_dict)
        return collections.UserDict(input_ids=self.ids) if return_dict else self.ids

class ChatTokenIdTests(unittest.TestCase):
    def test_requests_flat_list(self):
        tok = FakeTokenizer([1, 2, 3])
        self.assertEqual(chat_token_ids(tok, []), [1, 2, 3])
        self.assertEqual(tok.kwargs, dict(tokenize=True, add_generation_prompt=True, return_dict=False))
    def test_normalizes_compatibility_outputs(self):
        tok = FakeTokenizer([1, 2, 3]); tok.apply_chat_template = lambda *a, **k: collections.UserDict(input_ids=[1])
        self.assertEqual(chat_token_ids(tok, []), [1])
        tok.apply_chat_template = lambda *a, **k: [[1, 2]]
        self.assertEqual(chat_token_ids(tok, []), [1, 2])

class StreamHandler(http.server.BaseHTTPRequestHandler):
    """Replies with whatever event lines the test put in `lines`."""
    lines = []
    def log_message(self, *args): pass
    def do_POST(self):
        StreamHandler.body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.end_headers()
        for line in StreamHandler.lines: self.wfile.write(line.encode() + b'\n\n')

def event(chunk): return 'data: ' + json.dumps(chunk)

class GenerateStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.HTTPServer(('127.0.0.1', 0), StreamHandler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.url = 'http://127.0.0.1:%d/generate' % cls.server.server_port
    @classmethod
    def tearDownClass(cls): cls.server.shutdown(); cls.server.server_close()
    def run_stream(self, lines, record=None):
        StreamHandler.lines = lines
        return generate(self.url, [1, 2], {'max_new_tokens': 4}, 5, record)

    def test_final_chunk_and_first_token(self):
        r = Request('q', 'r', 'cot', 'a', 'b'); r.transition('ready'); r.transition('running')
        text, meta = self.run_stream([event({'text': 'Fin', 'meta_info': {'completion_tokens': 1}}),
                                      event({'text': 'Final answer: 42', 'meta_info': {'completion_tokens': 2, 'cached_tokens': 7}}),
                                      'data: [DONE]'], r)
        r.transition('completed')
        self.assertEqual((text, meta['cached_tokens']), ('Final answer: 42', 7))
        self.assertTrue(StreamHandler.body['stream'])
        self.assertEqual(StreamHandler.body['input_ids'], [1, 2])
        self.assertTrue(r.admitted_at <= r.first_token_at <= r.finished_at)
    def test_first_token_not_overwritten(self):
        r = Request('q', 'r', 'cot', 'a', 'b'); r.first_token_at = 'set-earlier'
        self.run_stream([event({'text': 'x', 'meta_info': {'completion_tokens': 1}}), 'data: [DONE]'], r)
        self.assertEqual(r.first_token_at, 'set-earlier')
    def test_token_count_beats_empty_text(self):
        r = Request('q', 'r', 'cot', 'a', 'b')
        self.run_stream([event({'text': '', 'meta_info': {'completion_tokens': 1}}), 'data: [DONE]'], r)
        self.assertIsNotNone(r.first_token_at)
        r = Request('q', 'r', 'cot', 'a', 'b')
        self.run_stream([event({'text': '', 'meta_info': {'completion_tokens': 0}}), 'data: [DONE]'], r)
        self.assertIsNone(r.first_token_at)
    def test_text_fallback_without_token_count(self):
        r = Request('q', 'r', 'cot', 'a', 'b')
        self.run_stream([event({'text': '', 'meta_info': {}}), event({'text': 'hi'}), 'data: [DONE]'], r)
        self.assertIsNotNone(r.first_token_at)
    def test_metadata_from_earlier_stream_chunks_is_preserved(self):
        text, meta = self.run_stream([event({'text': 'x', 'meta_info': {'prefill_gpu_ms': 2.5, 'cached_tokens': 3}}),
                                      event({'text': 'xy', 'meta_info': {'completion_tokens': 2}}), 'data: [DONE]'])
        self.assertEqual((text, meta['prefill_gpu_ms'], meta['cached_tokens'], meta['completion_tokens']), ('xy', 2.5, 3, 2))
    def test_empty_stream_and_engine_error(self):
        with self.assertRaises(RuntimeError): self.run_stream(['data: [DONE]'])
        with self.assertRaises(RuntimeError): self.run_stream([event({'error': {'message': 'boom'}})])
    def test_malformed_stream_is_a_clear_error(self):
        with self.assertRaisesRegex(RuntimeError, 'malformed streaming event'):
            self.run_stream(['data: not-json'])
        with self.assertRaisesRegex(RuntimeError, 'expected JSON object'):
            self.run_stream(['data: []'])
    def test_transport_timeout_propagates(self):
        with mock.patch('prefix_scheduler.core.urllib.request.urlopen', side_effect=TimeoutError('timeout')):
            with self.assertRaisesRegex(TimeoutError, 'timeout'):
                generate(self.url, [1], {'max_new_tokens': 1}, 0.01)

if __name__ == '__main__': unittest.main()
