#!/usr/bin/env python3
import argparse,json,math,pathlib,sys,urllib.request
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from prefix_scheduler.core import post_json
from transformers import AutoTokenizer
p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--out',required=True);a=p.parse_args()
tok=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct',revision='a09a35458c702b33eeacc393d103063234e8bc28')
ids=tok.encode('A shared prefix for GPU timing. '*100,add_special_tokens=False)
with urllib.request.urlopen(a.url+'/flush_cache',timeout=30) as resp:
    if resp.status!=200: raise RuntimeError('cache flush failed')
payload={'input_ids':ids,'sampling_params':{'temperature':0,'max_new_tokens':1}}
cold=post_json(a.url+'/generate',payload);warm=post_json(a.url+'/generate',payload)
result={'input_ids':ids,'cold':cold,'warm':warm}
path=pathlib.Path(a.out);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n')
for label,res in [('cold',cold),('warm',warm)]:
    value=res.get('meta_info',{}).get('prefill_gpu_ms')
    if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value) or value<=0: raise RuntimeError('%s missing valid GPU prefill timing: %r'%(label,value))
if cold['meta_info'].get('cached_tokens') != 0: raise RuntimeError('cold prefix was cached')
if warm['meta_info'].get('cached_tokens',0) <= 0: raise RuntimeError('warm prefix was not cached')
print('Validated cold GPU prefill %.3f ms and warm %.3f ms, cached tokens %d'%(cold['meta_info']['prefill_gpu_ms'],warm['meta_info']['prefill_gpu_ms'],warm['meta_info']['cached_tokens']))
