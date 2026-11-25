#!/usr/bin/env python3
"""Exercise and independently verify SGLang's opt-in live admission hook."""
import argparse, concurrent.futures, json, pathlib, sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from prefix_scheduler.core import post_json
from transformers import AutoTokenizer
p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--log',required=True);a=p.parse_args()
t=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct',revision='a09a35458c702b33eeacc393d103063234e8bc28')
shared=t.encode('Shared cache prefix arithmetic example. '*160,add_special_tokens=False)[:300]
def send(i):
 # Prime contains 300 common tokens.  The contenders have deliberately
 # different live-cache matches, while retaining the predictor's calibrated
 # cache-fraction range (roughly 0.25--0.75).
 keep=300 if i==99 else (112,160,208,256)[i % 4]
 suffix=t.encode(' Item '+str(i)+' unique arithmetic branch. ',add_special_tokens=False)
 ids=(shared[:keep]+suffix*256)[:384]
 return post_json(a.url+'/generate',{'rid':'live-test-'+str(i),'input_ids':ids,'sampling_params':{'temperature':0,'max_new_tokens':32}},300)
# Prime a real prefix; the subsequent calls contend while one is active.
send(99)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 results=dict(zip(('live-test-'+str(i) for i in range(8)),pool.map(send,range(8))))
events=[json.loads(x) for x in pathlib.Path(a.log).read_text().splitlines() if x.strip()]
events=[x for x in events if str(x.get('actual_id','')).startswith('live-test-')]
if len(events)!=9:raise RuntimeError('expected nine live admissions: '+str(len(events)))
if not any(len(e['candidates'])>=2 for e in events):raise RuntimeError('no queued competitors observed')
if not any(any(c['cached_tokens']>0 for c in e['candidates']) for e in events):raise RuntimeError('no live cache matches')
for e in events:
 if e['slot'] != e['admission'] % 4:raise RuntimeError('admission cycle counter is inconsistent')
 if e['reason']=='calibrated' and any(c['predicted_ms'] is None for c in e['candidates']):raise RuntimeError('calibrated choice lacks predictions')
 candidates={c['id']:c for c in e['candidates']}
 if e['reason']=='oldest': score=lambda c:0
 elif e['reason'] in ('k-lpm','baseline_fallback'): score=lambda c:c['baseline']
 elif e['reason']=='calibrated': score=lambda c:c['predicted_ms']
 else:raise RuntimeError('unknown policy reason: '+str(e['reason']))
 expected=[c['id'] for c in sorted(e['candidates'],key=lambda c:(-score(c),c['age'],str(c['id'])))]
 if e['ranked_ids'] != expected:raise RuntimeError('logged ranking disagrees with policy rule')
 if e['actual_id'] != expected[0]:raise RuntimeError('engine admitted a request other than the policy choice')
 actual=candidates[e['actual_id']]
 if e['actual_id'] != 'live-test-99':
  telemetry=results[e['actual_id']].get('meta_info',{}).get('cached_tokens')
  if telemetry is None:raise RuntimeError('engine response lacks cached-token telemetry')
  if int(telemetry)!=actual['cached_tokens']:raise RuntimeError('logged cache match disagrees with engine telemetry')
scored=[e for e in events if e['reason'] in ('k-lpm','calibrated') and len(e['candidates'])>=2]
if scored and not any(len({(c['baseline'] if e['reason']=='k-lpm' else c['predicted_ms']) for c in e['candidates']})>1 for e in scored):raise RuntimeError('no scored live-cache decision had distinct candidate values')
pathlib.Path(a.log).with_name('responses.json').write_text(json.dumps(results,indent=2,sort_keys=True)+'\n')
print('Validated',len(events),'actual admissions, ranked cycle, and engine cache telemetry')
