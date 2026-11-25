#!/usr/bin/env python3
"""Collect paired samples from an explicitly GPU-instrumented SGLang server."""
import argparse,json,pathlib,sys,time,urllib.request
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from prefix_scheduler.core import post_json
from transformers import AutoTokenizer
p=argparse.ArgumentParser();p.add_argument('--grid',help='JSON list of synthetic {length,cached} pairs');p.add_argument('--prompts',help='JSONL real tokenized prompts');p.add_argument('--fractions',default='0.25,0.6',help='cache fractions for --prompts');p.add_argument('--out',required=True);p.add_argument('--url',default='http://127.0.0.1:30000');p.add_argument('--config',default='configs/foundation.json');p.add_argument('--repetitions',type=int,default=3);a=p.parse_args()
if bool(a.grid)==bool(a.prompts): p.error('provide exactly one of --grid or --prompts')
c=json.loads(pathlib.Path(a.config).read_text());grid=json.loads(pathlib.Path(a.grid).read_text()) if a.grid else None
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
if a.prompts:
    fractions=[float(value) for value in a.fractions.split(',')]
    if not fractions or any(not 0 < value < 1 for value in fractions): p.error('fractions must lie strictly between zero and one')
    samples=[]
    for line in pathlib.Path(a.prompts).read_text().splitlines():
        if not line.strip(): continue
        prompt=json.loads(line);ids=prompt.get('input_ids',[])
        if not isinstance(ids,list) or len(ids)<2 or not all(isinstance(value,int) for value in ids): raise ValueError('invalid prompt input_ids')
        for fraction in fractions:
            cached=max(1,min(len(ids)-1,round(len(ids)*fraction)))
            samples.append(dict(prompt,length=len(ids),cached=cached,cache_fraction=fraction))
else:
    samples=[]
    for pair in grid:
        length,cached=int(pair['length']),int(pair['cached'])
        ids=(token*((length+len(token)-1)//len(token)))[:length]
        samples.append({'length':length,'cached':cached,'input_ids':ids,'synthetic':True})
with out.open('w') as f:
    for pair in samples:
        length,cached=int(pair['length']),int(pair['cached'])
        if length<=0 or cached<=0 or cached>=length: raise ValueError('expected 0 < cached < length')
        ids=pair['input_ids']
        for rep in range(a.repetitions):
            flush();cold=generate(ids);cold_ms=timing(cold)
            if cold.get('meta_info',{}).get('cached_tokens') != 0: raise RuntimeError('cold cache not empty')
            flush();generate(ids[:cached]);warm=generate(ids); warm_ms=timing(warm)
            observed=warm.get('meta_info',{}).get('cached_tokens')
            if observed!=cached: raise RuntimeError('actual matched tokens %r differs from target %d' % (observed,cached))
            row={'length':length,'cached':cached,'cold_prefill_ms':cold_ms,'warm_prefill_ms':warm_ms,'warm_cached_tokens':observed,'repeat':rep,'model_revision':c['model_revision'],'input_ids':ids}
            row.update({key:value for key,value in pair.items() if key not in ('length','cached','input_ids')})
            f.write(json.dumps(row)+'\n');f.flush()
