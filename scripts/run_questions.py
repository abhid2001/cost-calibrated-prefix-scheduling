#!/usr/bin/env python3
"""Serial direct and CoT runners; timestamps are client-side observations."""
import argparse, dataclasses, hashlib, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from prefix_scheduler.core import Request, now, parse_answer, post_json, prompt_for
from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument('--questions', default='data/development_questions.jsonl')
p.add_argument('--labels', default=None, help='Optional separate labels used only after model generation')
p.add_argument('--mode', choices=['direct','cot'], required=True)
p.add_argument('--url', default='http://127.0.0.1:30000')
p.add_argument('--config', default='configs/foundation.json')
p.add_argument('--out', required=True)
p.add_argument('--limit', type=int, default=10)
a = p.parse_args()
c = json.loads(pathlib.Path(a.config).read_text())
tokenizer = AutoTokenizer.from_pretrained(c['model'], revision=c['model_revision'])
questions = [json.loads(s) for s in pathlib.Path(a.questions).read_text().splitlines() if s.strip()][:a.limit]
labels = {} if not a.labels else {r['question_id']:r['answer'] for r in (json.loads(s) for s in pathlib.Path(a.labels).read_text().splitlines() if s.strip())}
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
from prefix_scheduler.core import normalize_number
with (out/'requests.jsonl').open('w') as requests, (out/'questions.jsonl').open('w') as summaries:
    for q in questions:
        arrival = now(); qid = q['question_id']; rid = qid + ':' + a.mode
        seed = int.from_bytes(hashlib.sha256(rid.encode()).digest()[:4], 'big') % (2**31)
        record = Request(qid, rid, a.mode, arrival, now(), sampling_seed=seed, max_new_tokens=c['max_new_tokens'])
        answer = None
        try:
            messages = prompt_for(q['question'], a.mode)
            record.prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            record.input_token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            record.transition('ready'); record.transition('running')
            result = post_json(a.url + '/generate', {'input_ids': record.input_token_ids,
                'sampling_params': {'temperature': c['temperature'], 'max_new_tokens': record.max_new_tokens, 'sampling_seed': seed}}, c['request_timeout_seconds'])
            record.output = result['text']; record.engine_metadata = result.get('meta_info', {})
            record.matched_prefix_tokens = record.engine_metadata.get('cached_tokens')
            answer = parse_answer(record.output)
            if answer is None: raise ValueError('missing or ambiguous final numeric answer')
            record.transition('completed')
        except Exception as exc:
            record.error = '%s: %s' % (type(exc).__name__, exc)
            record.transition('failed')
        requests.write(json.dumps(dataclasses.asdict(record))+'\n'); requests.flush()
        label = normalize_number(labels[qid]) if qid in labels else None
        summaries.write(json.dumps({'question_id': qid, 'mode': a.mode, 'arrival': arrival, 'finished_at': record.finished_at,
            'status': record.status, 'answer': answer, 'correct': answer == label if label is not None else None,
            'error': record.error})+'\n'); summaries.flush()
