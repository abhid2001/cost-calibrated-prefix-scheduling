#!/usr/bin/env python3
"""Run self consistency or bounded ToT against the native SGLang API serially."""
import argparse, dataclasses, json, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from prefix_scheduler.core import chat_token_ids, generate, normalize_number, now
from prefix_scheduler.reasoning import Controller
from transformers import AutoTokenizer

p=argparse.ArgumentParser()
p.add_argument('--mode', choices=['self_consistency','tot'], required=True)
p.add_argument('--questions', default='data/development_questions.jsonl')
p.add_argument('--labels')
p.add_argument('--config', default='configs/foundation.json')
p.add_argument('--url', default='http://127.0.0.1:30000')
p.add_argument('--out', required=True)
p.add_argument('--limit', type=int, default=10)
a=p.parse_args(); config=json.loads(pathlib.Path(a.config).read_text())
tokenizer=AutoTokenizer.from_pretrained(config['model'], revision=config['model_revision'])
questions=[json.loads(x) for x in pathlib.Path(a.questions).read_text().splitlines() if x.strip()][:a.limit]
labels={} if not a.labels else {x['question_id']:x['answer'] for x in (json.loads(s) for s in pathlib.Path(a.labels).read_text().splitlines() if s.strip())}
out=pathlib.Path(a.out); out.mkdir(parents=True,exist_ok=True)

def prepare(r, messages):
    r.prompt=tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    r.input_token_ids=chat_token_ids(tokenizer, messages)

def execute(r, messages, temperature):
    start=time.monotonic()
    try:
        text, r.engine_metadata=generate(a.url+'/generate', r.input_token_ids, {
            'temperature':temperature, 'max_new_tokens':r.max_new_tokens, 'sampling_seed':r.sampling_seed}, config['request_timeout_seconds'], r)
        r.matched_prefix_tokens=r.engine_metadata.get('cached_tokens')
        return text
    finally:
        r.client_duration_ms=(time.monotonic()-start)*1000

with (out/'requests.jsonl').open('w') as req, (out/'questions.jsonl').open('w') as summary:
    for q in questions:
        c=Controller(execute,config['max_new_tokens'],prepare); arrival=now()
        result=getattr(c,a.mode)(q['question_id'],q['question'],arrival)
        result['arrival']=arrival
        label=normalize_number(labels[q['question_id']]) if q['question_id'] in labels else None
        result['correct']=(result['answer']==label) if label is not None else None
        for record in c.records:
            req.write(json.dumps(dataclasses.asdict(record))+'\n')
        req.flush(); summary.write(json.dumps(result)+'\n'); summary.flush()
