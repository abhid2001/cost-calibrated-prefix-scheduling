#!/usr/bin/env python3
"""Summarize question and request logs without inventing unavailable metrics."""
import argparse,datetime,json,math,pathlib,statistics
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args()
d=pathlib.Path(a.run)
def rows(name):return [json.loads(s) for s in (d/name).read_text().splitlines() if s.strip()]
qs=rows('questions.jsonl');rs=rows('requests.jsonl')
def sec(a,b):return (datetime.datetime.fromisoformat(b)-datetime.datetime.fromisoformat(a)).total_seconds()
def quantile(xs,q):return sorted(xs)[max(0,math.ceil(len(xs)*q)-1)] if xs else None
lat=[sec(q['arrival'],q['finished_at']) for q in qs if q.get('arrival') and q.get('finished_at')]
wait=[sec(r['ready_at'],r['admitted_at']) for r in rs if r.get('ready_at') and r.get('admitted_at')]
cached=[r['matched_prefix_tokens'] for r in rs if r.get('matched_prefix_tokens') is not None]
correct=[q['correct'] for q in qs if q.get('correct') is not None]
dur=sec(min(q['arrival'] for q in qs),max(q['finished_at'] for q in qs)) if qs and all(q.get('arrival') and q.get('finished_at') for q in qs) else 0
result={'question_count':len(qs),'question_failures':sum(q['status']!='completed' for q in qs),
        'accuracy_over_scored_questions':sum(correct)/len(correct) if correct else None,'scored_questions':len(correct),
        'latency_p50_seconds':statistics.median(lat) if lat else None,'latency_p95_seconds':quantile(lat,.95),
        'completed_questions_per_minute':sum(q['status']=='completed' for q in qs)*60/dur if dur>0 else None,
        'request_count':len(rs),'request_failures':sum(r['status']=='failed' for r in rs),
        'reported_cache_tokens_total':sum(cached),'cache_telemetry_requests':len(cached),
        'max_client_ready_to_dispatch_seconds':max(wait) if wait else None,
        'evaluator_failures':sum(q.get('evaluator_failures',0) for q in qs),
        'gpu_prefill_prediction_error':None,
        'notes':['Failures included in question latency and accuracy when labels exist.', 'Ready-to-dispatch is client-side, not engine queue time.', 'GPU prefill prediction error unavailable without paired instrumentation.']}
out=pathlib.Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
