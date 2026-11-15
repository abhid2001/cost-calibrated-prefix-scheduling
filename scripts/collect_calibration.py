#!/usr/bin/env python3
"""Collect paired samples from an explicitly GPU-instrumented SGLang server."""
import argparse,json,pathlib,sys,time,urllib.request
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from prefix_scheduler.core import post_json
from transformers import AutoTokenizer
p=argparse.ArgumentParser();p.add_argument('--grid',required=True,help='JSON list of {length,cached} pairs');p.add_argument('--out',required=True);p.add_argument('--url',default='http://127.0.0.1:30000');p.add_argument('--config',default='configs/foundation.json');p.add_argument('--repetitions',type=int,default=3);a=p.parse_args()
c=json.loads(pathlib.Path(a.config).read_text());grid=json.loads(pathlib.Path(a.grid).read_text())
if a.repetitions<1:p.error('repetitions must be positive')
tok=AutoTokenizer.from_pretrained(c['model'],revision=c['model_revision']); token=tok.encode(' apple',add_special_tokens=False)
if not token: raise SystemExit('tokenizer produced no tokens')
def flush():
    with urllib.request.urlopen(a.url+'/flush_cache',timeout=30) as r:
        if r.status != 200: raise RuntimeError('cache flush failed')
def generate(ids):
    return post_json(a.url+'/generate',{'input_ids':ids,'sampling_params':{'temperature':0,'max_new_tokens':1}},300)
def timing(resp):
    value=resp.get('meta_info',{}).get('prefill_gpu_ms')
    if not isinstance(value,(int,float)) or value<0: raise RuntimeError('missing prefill_gpu_ms engine telemetry; instrument GPU prefill before collecting data')
    return value
# Warm kernels before samples; exclude warmup from collected timings.
generate((token*128)[:128]);out=pathlib.Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w') as f:
    for pair in grid:
        length,cached=int(pair['length']),int(pair['cached'])
        if length<=0 or cached<=0 or cached>=length: raise ValueError('expected 0 < cached < length')
        ids=(token*((length+len(token)-1)//len(token)))[:length]
        for rep in range(a.repetitions):
            flush();cold=generate(ids);cold_ms=timing(cold)
            if cold.get('meta_info',{}).get('cached_tokens') != 0: raise RuntimeError('cold cache not empty')
            flush();generate(ids[:cached+1]);warm=generate(ids); warm_ms=timing(warm)
            observed=warm.get('meta_info',{}).get('cached_tokens')
            if observed!=cached: raise RuntimeError('actual matched tokens %r differs from target %d' % (observed,cached))
            row={'length':length,'cached':cached,'cold_prefill_ms':cold_ms,'warm_prefill_ms':warm_ms,'warm_cached_tokens':observed,'repeat':rep,'model_revision':c['model_revision'],'input_ids':ids}
            f.write(json.dumps(row)+'\n');f.flush()
