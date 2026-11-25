#!/usr/bin/env python3
"""Open-loop HTTP load generator for single-call modes. No scheduling claims."""
import argparse,concurrent.futures,datetime,json,pathlib,sys,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from prefix_scheduler.core import chat_token_ids, now,parse_answer,post_json,prompt_for
from transformers import AutoTokenizer
p=argparse.ArgumentParser();p.add_argument('--trace',required=True);p.add_argument('--out',required=True);p.add_argument('--url',default='http://127.0.0.1:30000');p.add_argument('--config',default='configs/foundation.json');p.add_argument('--workers',type=int,default=32);a=p.parse_args()
c=json.loads(pathlib.Path(a.config).read_text()); trace=[json.loads(x) for x in pathlib.Path(a.trace).read_text().splitlines() if x.strip()]
if not trace or a.workers<1: p.error('nonempty trace and positive workers required')
if any(x['offset_seconds']<0 for x in trace) or any(trace[i]['offset_seconds']>trace[i+1]['offset_seconds'] for i in range(len(trace)-1)): p.error('trace offsets must be nonnegative and ordered')
tok=AutoTokenizer.from_pretrained(c['model'],revision=c['model_revision']); inputs=[chat_token_ids(tok,prompt_for(x['question'],x['mode'])) for x in trace]
start=time.monotonic(); epoch=datetime.datetime.now(datetime.timezone.utc)
def send(row, ids):
    dispatch=now(); t=time.monotonic()
    try:
        response=post_json(a.url+'/generate',{'input_ids':ids,'sampling_params':{'temperature':0,'max_new_tokens':c['max_new_tokens'],'sampling_seed':row['seed']}},c['request_timeout_seconds'])
        text=response['text'];answer=parse_answer(text);status='completed' if answer is not None else 'failed';error=None if answer else 'unparseable answer';meta=response.get('meta_info',{})
    except Exception as exc:
        text=None;answer=None;status='failed';error=type(exc).__name__+': '+str(exc);meta={}
    return {'question_id':row['question_id'],'mode':row['mode'],'arrival':(epoch+datetime.timedelta(seconds=row['offset_seconds'])).isoformat(),'dispatch':dispatch,'finished_at':now(),'client_duration_ms':(time.monotonic()-t)*1000,'status':status,'answer':answer,'error':error,'output':text,'engine_metadata':meta,'input_tokens':len(ids),'seed':row['seed']}
futures=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
    for row,ids in zip(trace,inputs):
        time.sleep(max(0,start+row['offset_seconds']-time.monotonic()))
        futures.append(pool.submit(send,row,ids))
    rows=[f.result() for f in futures]
out=pathlib.Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(''.join(json.dumps(r)+'\n' for r in rows));print('Saved',len(rows),'results to',out)
