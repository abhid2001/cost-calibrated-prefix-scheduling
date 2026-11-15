#!/usr/bin/env python3
import argparse,datetime,json,math,pathlib
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',required=True);a=p.parse_args()
rows=[json.loads(x) for x in pathlib.Path(a.input).read_text().splitlines() if x.strip()]
def seconds(a,b): return (datetime.datetime.fromisoformat(b)-datetime.datetime.fromisoformat(a)).total_seconds()
def pct(xs,q):
    if not xs:return None
    xs=sorted(xs); return xs[max(0,math.ceil(q*len(xs))-1)]
lat=[seconds(x['arrival'],x['finished_at']) for x in rows]; waits=[seconds(x['arrival'],x['dispatch']) for x in rows if x.get('dispatch')]
completed=sum(r['status']=='completed' for r in rows)
duration=seconds(min(x['arrival'] for x in rows),max(x['finished_at'] for x in rows)) if rows else 0
result={'total':len(rows),'completed':completed,'failed':len(rows)-completed,'latency_median_seconds':pct(lat,.5),'latency_p95_seconds':pct(lat,.95),'max_arrival_to_dispatch_seconds':max(waits) if waits else None,'completed_per_minute':60*completed/duration if duration>0 else None,'note':'when available, arrival-to-dispatch includes local executor waiting; not engine queue time; failures included in latency'}
pathlib.Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
