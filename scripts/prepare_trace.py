#!/usr/bin/env python3
"""Create reproducible open-loop request arrivals for a frozen question set."""
import argparse,json,pathlib,random
p=argparse.ArgumentParser(); p.add_argument('--questions',required=True);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=20261114);p.add_argument('--rate',type=float,required=True,help='mean arrivals per second');p.add_argument('--mode',choices=['direct','cot','self_consistency','tot'],default='cot');p.add_argument('--modes',help='comma-separated mode cycle, for example direct,cot,self_consistency,tot');p.add_argument('--limit',type=int,help='optional number of questions');a=p.parse_args()
if a.rate<=0: p.error('rate must be positive')
if a.limit is not None and a.limit<1: p.error('--limit must be positive')
if a.modes:
    modes=a.modes.split(',')
    if not modes or any(mode not in ('direct','cot','self_consistency','tot') for mode in modes): p.error('--modes must contain supported comma-separated modes')
else: modes=[a.mode]
questions=[json.loads(s) for s in pathlib.Path(a.questions).read_text().splitlines() if s.strip()]
if a.limit is not None: questions=questions[:a.limit]
if not questions: p.error('empty questions')
rng=random.Random(a.seed); offset=0.0; seen=set(); result=[]
for i,q in enumerate(questions):
    if q['question_id'] in seen: p.error('duplicate question id')
    seen.add(q['question_id'])
    if i: offset+=rng.expovariate(a.rate)
    result.append({'question_id':q['question_id'],'question':q['question'],'mode':modes[i % len(modes)],'offset_seconds':offset,'seed':rng.randrange(2**31)})
out=pathlib.Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(''.join(json.dumps(x)+'\n' for x in result));print('Saved',len(result),'arrivals; seed',a.seed,'rate',a.rate)
