#!/usr/bin/env python3
"""Concurrent question arrivals, dependency-ordered reasoning calls to SGLang.

One controller thread per question; each controller submits its calls in dependency
order. The engine owns its queue and admission policy. This runner does not claim
policy-controlled scheduling or simultaneous branches within a question.
"""
import argparse, concurrent.futures, dataclasses, datetime, json, pathlib, sys, time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from prefix_scheduler.core import now, chat_token_ids, generate, parse_answer, normalize_number, prompt_for
from prefix_scheduler.reasoning import Controller
from transformers import AutoTokenizer
p=argparse.ArgumentParser();p.add_argument('--trace',required=True);p.add_argument('--out',required=True);p.add_argument('--labels');p.add_argument('--url',default='http://127.0.0.1:30000');p.add_argument('--config',default='configs/foundation.json');p.add_argument('--question-workers',type=int,default=32);a=p.parse_args()
c=json.loads(pathlib.Path(a.config).read_text());trace=[json.loads(x) for x in pathlib.Path(a.trace).read_text().splitlines() if x.strip()]
if not trace or a.question_workers<1:p.error('nonempty trace and positive question-workers required')
if any(x['offset_seconds']<0 for x in trace) or any(trace[i]['offset_seconds']>trace[i+1]['offset_seconds'] for i in range(len(trace)-1)):p.error('offsets must be nonnegative and ordered')
if any(x['mode'] not in ('direct','cot','self_consistency','tot') for x in trace):p.error('unsupported mode')
labels={} if not a.labels else {r['question_id']:normalize_number(r['answer']) for r in (json.loads(x) for x in pathlib.Path(a.labels).read_text().splitlines() if x.strip())}
tok=AutoTokenizer.from_pretrained(c['model'],revision=c['model_revision']); start=time.monotonic(); epoch=datetime.datetime.now(datetime.timezone.utc)
def prepare(r,messages):
    r.prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    r.input_token_ids=chat_token_ids(tok,messages)
def execute(r,messages,temperature):
    started=time.monotonic()
    try:
        text,r.engine_metadata=generate(a.url+'/generate',r.input_token_ids,{'temperature':temperature,'max_new_tokens':r.max_new_tokens,'sampling_seed':r.sampling_seed},c['request_timeout_seconds'],r)
        r.matched_prefix_tokens=r.engine_metadata.get('cached_tokens');return text
    finally:r.client_duration_ms=(time.monotonic()-started)*1000
def run(row):
    arrival=(epoch+datetime.timedelta(seconds=row['offset_seconds'])).isoformat();ctrl=Controller(execute,c['max_new_tokens'],prepare);mode=row['mode'];qid=row['question_id']
    if mode in ('tot','self_consistency'):
        result=getattr(ctrl,mode)(qid,row['question'],arrival)
    else:
        req=ctrl.call(qid,arrival,mode,mode,[],prompt_for(row['question'],mode),0.0)
        answer=parse_answer(req.output) if req.status=='completed' else None
        result={'question_id':qid,'mode':mode,'status':'completed' if answer is not None else 'failed','answer':answer,'error':req.error if req.error else (None if answer is not None else 'unparseable answer'),'finished_at':now()}
    result['arrival']=arrival; result['trace_seed']=row['seed']
    result['correct']=(result['answer']==labels[qid]) if qid in labels and labels[qid] is not None else None
    return result,[dataclasses.asdict(r) for r in ctrl.records]
futures=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=a.question_workers) as pool:
    for row in trace:
        time.sleep(max(0,start+row['offset_seconds']-time.monotonic())); futures.append(pool.submit(run,row))
    results=[f.result() for f in futures]
out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
with (out/'questions.jsonl').open('w') as q, (out/'requests.jsonl').open('w') as r:
    for question,records in results:
        q.write(json.dumps(question)+'\n')
        for record in records:r.write(json.dumps(record)+'\n')
print('Saved',len(results),'question results to',out)
