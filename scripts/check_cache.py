#!/usr/bin/env python3
"""Prove reuse with SGLang's per-request cached_tokens telemetry."""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from prefix_scheduler.core import post_json
from transformers import AutoTokenizer
p = argparse.ArgumentParser(); p.add_argument('--url', default='http://127.0.0.1:30000'); p.add_argument('--config', default='configs/foundation.json'); p.add_argument('--out', required=True)
a=p.parse_args(); c=json.loads(pathlib.Path(a.config).read_text())
tok=AutoTokenizer.from_pretrained(c['model'], revision=c['model_revision'])
text=('A shared example of arithmetic reasoning: 19 plus 23 equals 42.\n' * 80) + 'What is 7 plus 8? Final answer:'
ids=tok.encode(text, add_special_tokens=False)
payload={'input_ids':ids,'sampling_params':{'temperature':0,'max_new_tokens':8}}
first=post_json(a.url+'/generate',payload); second=post_json(a.url+'/generate',payload)
result={'input_token_ids':ids,'first':first,'second':second,'model_revision':c['model_revision']}
path=pathlib.Path(a.out); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(result,indent=2)+'\n')
cached=second.get('meta_info',{}).get('cached_tokens')
if not isinstance(cached,int) or cached <= 0: raise SystemExit('Cache verification failed: second cached_tokens=%r; see %s' % (cached,path))
print('Verified second request reused %d cached tokens; evidence: %s' % (cached,path))
