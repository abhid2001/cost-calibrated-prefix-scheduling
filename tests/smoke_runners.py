"""Exercise all CLI runners with a mock tokenizer and server, not a GPU."""
import dataclasses, http.server, json, os, pathlib, subprocess, sys, tempfile, threading
ROOT=pathlib.Path(__file__).resolve().parents[1]
class Handler(http.server.BaseHTTPRequestHandler):
    calls=0
    def log_message(self, *args): pass
    def do_POST(self):
        data=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if not data.get('input_ids'): raise AssertionError('missing input IDs')
        self.__class__.calls += 1
        call=self.__class__.calls
        answer='Selected: 1' if MODE == 'tot' and call in (3,6) else 'Final answer: 42'
        payload=json.dumps({'text':answer,'meta_info':{'cached_tokens':0}}).encode()
        self.send_response(200); self.send_header('Content-Type','application/json'); self.end_headers(); self.wfile.write(payload)
server=http.server.HTTPServer(('127.0.0.1',0),Handler)
threading.Thread(target=server.serve_forever,daemon=True).start()
env=os.environ.copy(); env.update(HF_HOME=str(ROOT/'.cache/huggingface'), XDG_CACHE_HOME=str(ROOT/'.cache'), TMPDIR=str(ROOT/'tmp'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as directory:
    fake=pathlib.Path(directory)/'transformers'; fake.mkdir()
    (fake/'__init__.py').write_text("class Tokenizer:\n def apply_chat_template(self,messages,tokenize,add_generation_prompt): return [1,2,3] if tokenize else 'mock prompt'\nclass AutoTokenizer:\n @classmethod\n def from_pretrained(cls,*args,**kwargs): return Tokenizer()\n")
    env['PYTHONPATH']=directory
    q=pathlib.Path(directory)/'questions.jsonl'; q.write_text(json.dumps({'question_id':'q1','question':'What is 40 plus 2?'})+'\n')
    for MODE in ('direct','cot','self_consistency','tot'):
        Handler.calls=0; out=pathlib.Path(directory)/MODE
        script='run_questions.py' if MODE in ('direct','cot') else 'run_reasoning.py'
        subprocess.run([sys.executable,'scripts/'+script,'--mode',MODE,'--questions',str(q),'--url',f'http://127.0.0.1:{server.server_port}','--out',str(out)],cwd=ROOT,env=env,check=True)
        results=[json.loads(s) for s in (out/'questions.jsonl').read_text().splitlines()]
        records=[json.loads(s) for s in (out/'requests.jsonl').read_text().splitlines()]
        count={'direct':1,'cot':1,'self_consistency':3,'tot':7}[MODE]
        assert len(records)==count and Handler.calls==count and results[0]['status']=='completed' and results[0]['answer']=='42', (MODE,results)
        if MODE=='tot':
            assert len(records[2]['parent_ids'])==2 and len(records[3]['parent_ids'])==2 and len(records[5]['parent_ids'])==2 and len(records[6]['parent_ids'])==2
        print(MODE, 'ok:', count, 'calls')
    for MODE in ('self_consistency','tot'):
        Handler.calls=0
        trace=pathlib.Path(directory)/('trace-'+MODE+'.jsonl')
        trace.write_text(json.dumps({'question_id':'q1','question':'What is 40 plus 2?','mode':MODE,'offset_seconds':0,'seed':17})+'\n')
        out=pathlib.Path(directory)/('load-'+MODE)
        subprocess.run([sys.executable,'scripts/load_reasoning_trace.py','--trace',str(trace),'--url',f'http://127.0.0.1:{server.server_port}','--out',str(out)],cwd=ROOT,env=env,check=True)
        result=json.loads((out/'questions.jsonl').read_text().strip())
        records=[json.loads(x) for x in (out/'requests.jsonl').read_text().splitlines()]
        assert result['status']=='completed' and len(records)==(3 if MODE=='self_consistency' else 7)
        if MODE=='tot': assert len(records[2]['parent_ids'])==2 and len(records[5]['parent_ids'])==2
        print('load trace', MODE, 'ok:', len(records), 'calls')
server.shutdown()
