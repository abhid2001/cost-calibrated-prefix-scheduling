"""Exercise all CLI runners with a mock tokenizer and server, not a GPU."""
import dataclasses, http.server, json, os, pathlib, subprocess, sys, tempfile, threading, time
ROOT=pathlib.Path(__file__).resolve().parents[1]
class Handler(http.server.BaseHTTPRequestHandler):
    calls=0; in_flight=0; max_in_flight=0; lock=threading.Lock(); admission_log=None; fail_rid=None; omit_timing=False; drop_admission=False; duplicate_admission=False
    def log_message(self, *args): pass
    def do_POST(self):
        data=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if not data.get('input_ids'): raise AssertionError('missing input IDs')
        if data.get('rid') == self.__class__.fail_rid and self.__class__.fail_rid is not None:
            self.send_error(503, 'injected failure'); return
        with self.__class__.lock:
            self.__class__.calls += 1; call=self.__class__.calls
            self.__class__.in_flight += 1
            self.__class__.max_in_flight=max(self.__class__.max_in_flight,self.__class__.in_flight)
            if self.__class__.admission_log and data.get('rid') and not self.__class__.drop_admission:
                with open(self.__class__.admission_log,'a') as log:
                    event=json.dumps({'actual_id':data['rid'],'admission':call-1,'slot':(call-1)%4,
                                          'reason':'oldest','ranked_ids':[data['rid']],
                                          'candidates':[{'id':data['rid'],'cached_tokens':0}],'ts_ns':time.time_ns()})+'\n'
                    log.write(event)
                    if self.__class__.duplicate_admission: log.write(event)
        time.sleep(.01)
        answer='Selected: 1' if MODE == 'tot' and call in (3,6) else 'Final answer: 42'
        metadata={'cached_tokens':0,'completion_tokens':1}
        if not self.__class__.omit_timing: metadata['prefill_gpu_ms']=1.25
        payload=json.dumps({'text':answer,'meta_info':metadata}).encode()
        self.send_response(200)
        if data.get('stream'):
            self.send_header('Content-Type','text/event-stream')
            self.end_headers()
            self.wfile.write(b'data: '+payload+b'\n\n')
            self.wfile.write(b'data: [DONE]\n\n')
        else:
            self.send_header('Content-Type','application/json')
            self.end_headers(); self.wfile.write(payload)
        with self.__class__.lock: self.__class__.in_flight -= 1
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
threading.Thread(target=server.serve_forever,daemon=True).start()
env=os.environ.copy(); env.update(HF_HOME=str(ROOT/'.cache/huggingface'), XDG_CACHE_HOME=str(ROOT/'.cache'), TMPDIR=str(ROOT/'tmp'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as directory:
    fake=pathlib.Path(directory)/'transformers'; fake.mkdir()
    (fake/'__init__.py').write_text("class Tokenizer:\n def apply_chat_template(self,messages,tokenize,add_generation_prompt,return_dict=False): return [1,2,3] if tokenize else 'mock prompt'\nclass AutoTokenizer:\n @classmethod\n def from_pretrained(cls,*args,**kwargs): return Tokenizer()\n")
    env['PYTHONPATH']=directory
    q=pathlib.Path(directory)/'questions.jsonl'; q.write_text(json.dumps({'question_id':'q1','question':'What is 40 plus 2?'})+'\n')
    for MODE in ('direct','cot','self_consistency','tot'):
        Handler.calls=0; Handler.admission_log=None; out=pathlib.Path(directory)/MODE
        script='run_questions.py' if MODE in ('direct','cot') else 'run_reasoning.py'
        subprocess.run([sys.executable,'scripts/'+script,'--mode',MODE,'--questions',str(q),'--url',f'http://127.0.0.1:{server.server_port}','--out',str(out)],cwd=ROOT,env=env,check=True)
        results=[json.loads(s) for s in (out/'questions.jsonl').read_text().splitlines()]
        records=[json.loads(s) for s in (out/'requests.jsonl').read_text().splitlines()]
        count={'direct':1,'cot':1,'self_consistency':3,'tot':7}[MODE]
        assert len(records)==count and Handler.calls==count and results[0]['status']=='completed' and results[0]['answer']=='42', (MODE,results)
        if MODE=='tot':
            assert len(records[2]['parent_ids'])==2 and len(records[3]['parent_ids'])==2 and len(records[5]['parent_ids'])==2 and len(records[6]['parent_ids'])==2
        print(MODE, 'ok:', count, 'calls')
    for MODE in ('direct','cot','self_consistency','tot'):
        Handler.calls=0; Handler.max_in_flight=0
        trace=pathlib.Path(directory)/('trace-'+MODE+'.jsonl')
        trace.write_text(json.dumps({'question_id':'q1','question':'What is 40 plus 2?','mode':MODE,'offset_seconds':0,'seed':17})+'\n')
        out=pathlib.Path(directory)/('load-'+MODE)
        log=pathlib.Path(directory)/('admission-'+MODE+'.jsonl'); log.write_text(''); graph=pathlib.Path(directory)/('graph-'+MODE+'.json'); Handler.admission_log=log
        subprocess.run([sys.executable,'scripts/load_reasoning_trace.py','--trace',str(trace),'--url',f'http://127.0.0.1:{server.server_port}','--out',str(out),'--admission-log',str(log),'--dependency-graph',str(graph),'--require-live-admissions','--require-stream-telemetry'],cwd=ROOT,env=env,check=True)
        result=json.loads((out/'questions.jsonl').read_text().strip())
        records=[json.loads(x) for x in (out/'requests.jsonl').read_text().splitlines()]
        events=[json.loads(x) for x in (out/'events.jsonl').read_text().splitlines()]
        count={'direct':1,'cot':1,'self_consistency':3,'tot':7}[MODE]
        assert result['status']=='completed' and len(records)==count and Handler.calls==count
        assert all(r['dispatched_at'] and r['admitted_at'] and r['first_token_at'] and r['engine_metadata']['prefill_gpu_ms'] == 1.25 for r in records)
        snapshot=json.loads(graph.read_text()); assert {r['request_id']:r['status'] for r in snapshot['requests']} == {r['request_id']:r['status'] for r in records}
        assert max(e['sequence'] for e in events if e['event']=='materialized') < min(e['sequence'] for e in events if e['event']=='question_arrived')
        if MODE=='tot': assert len(records[2]['parent_ids'])==2 and len(records[5]['parent_ids'])==2
        if MODE=='self_consistency': assert Handler.max_in_flight > 1
        print('load trace', MODE, 'ok:', len(records), 'calls')
    MODE='tot'; Handler.fail_rid='q1:r1c1'; Handler.admission_log=None
    out=pathlib.Path(directory)/'load-failure'
    subprocess.run([sys.executable,'scripts/load_reasoning_trace.py','--trace',str(trace),'--url',f'http://127.0.0.1:{server.server_port}','--out',str(out)],cwd=ROOT,env=env,check=True)
    records={r['request_id']:r for r in (json.loads(x) for x in (out/'requests.jsonl').read_text().splitlines())}
    assert records['q1:r1c1']['status']=='failed' and records['q1:r1c1']['admitted_at'] is None
    assert all(records['q1:'+node]['status']=='failed' and records['q1:'+node]['dispatched_at'] is None for node in ('r1eval','r2c1','r2c2','r2eval','final'))
    print('failed parent propagates to undispatched descendants: ok')
    Handler.fail_rid=None; Handler.omit_timing=True; MODE='direct'
    failed=subprocess.run([sys.executable,'scripts/load_reasoning_trace.py','--trace',str(pathlib.Path(directory)/'trace-direct.jsonl'),'--url',f'http://127.0.0.1:{server.server_port}','--out',str(pathlib.Path(directory)/'missing-telemetry'),'--require-stream-telemetry'],cwd=ROOT,env=env,capture_output=True,text=True)
    assert failed.returncode != 0 and 'missing or invalid prefill_gpu_ms' in failed.stderr
    print('missing streaming telemetry fails clearly: ok')
    Handler.omit_timing=False; Handler.admission_log=pathlib.Path(directory)/'missing-admission.jsonl'; Handler.admission_log.write_text(''); Handler.drop_admission=True
    failed=subprocess.run([sys.executable,'scripts/load_reasoning_trace.py','--trace',str(pathlib.Path(directory)/'trace-direct.jsonl'),'--url',f'http://127.0.0.1:{server.server_port}','--out',str(pathlib.Path(directory)/'missing-admission'),'--admission-log',str(Handler.admission_log),'--require-live-admissions'],cwd=ROOT,env=env,capture_output=True,text=True)
    assert failed.returncode != 0 and 'Missing or non-unique engine admissions' in failed.stderr
    Handler.drop_admission=False; Handler.duplicate_admission=True; Handler.admission_log=pathlib.Path(directory)/'duplicate-admission.jsonl'; Handler.admission_log.write_text('')
    failed=subprocess.run([sys.executable,'scripts/load_reasoning_trace.py','--trace',str(pathlib.Path(directory)/'trace-direct.jsonl'),'--url',f'http://127.0.0.1:{server.server_port}','--out',str(pathlib.Path(directory)/'duplicate-admission'),'--admission-log',str(Handler.admission_log),'--require-live-admissions'],cwd=ROOT,env=env,capture_output=True,text=True)
    assert failed.returncode != 0 and 'Missing or non-unique engine admissions' in failed.stderr
    print('missing and duplicate admissions fail clearly: ok')
server.shutdown()
