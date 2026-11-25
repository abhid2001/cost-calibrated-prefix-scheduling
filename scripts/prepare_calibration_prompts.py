#!/usr/bin/env python3
"""Create varied, real GSM8K chat prompts for prefill calibration."""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from prefix_scheduler.core import chat_token_ids, prompt_for
from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument('--questions', required=True)
p.add_argument('--demonstrations', default='data/demonstrations_questions.jsonl')
p.add_argument('--out', required=True)
p.add_argument('--config', default='configs/foundation.json')
p.add_argument('--start', type=int, default=0)
p.add_argument('--limit', type=int, required=True)
p.add_argument('--demo-counts', default='0,2,4')
p.add_argument('--modes', default='direct,cot')
a = p.parse_args()
if a.start < 0 or a.limit < 1:
    p.error('--start must be nonnegative and --limit positive')
counts = [int(value) for value in a.demo_counts.split(',')]
modes = a.modes.split(',')
if not counts or any(value < 0 for value in counts) or any(mode not in ('direct', 'cot') for mode in modes):
    p.error('invalid demo counts or modes')
questions = [json.loads(line) for line in pathlib.Path(a.questions).read_text().splitlines() if line.strip()]
selected = questions[a.start:a.start + a.limit]
if len(selected) != a.limit:
    p.error('requested prompt range exceeds questions')
demos = [json.loads(line) for line in pathlib.Path(a.demonstrations).read_text().splitlines() if line.strip()]
if max(counts, default=0) > len(demos):
    p.error('not enough demonstration questions')
config = json.loads(pathlib.Path(a.config).read_text())
tokenizer = AutoTokenizer.from_pretrained(config['model'], revision=config['model_revision'])
out = pathlib.Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
with out.open('w') as handle:
    for question in selected:
        for mode in modes:
            for count in counts:
                messages = prompt_for(question['question'], mode)
                if count:
                    examples = '\n\n'.join('Example question %d: %s' % (index + 1, demo['question'])
                                           for index, demo in enumerate(demos[:count]))
                    messages[0] = dict(messages[0])
                    messages[0]['content'] += '\nUse the following real GSM8K questions as related examples:\n' + examples
                ids = chat_token_ids(tokenizer, messages)
                handle.write(json.dumps({'prompt_id': '%s:%s:demos%d' % (question['question_id'], mode, count),
                                         'question_id': question['question_id'], 'mode': mode,
                                         'demo_count': count, 'input_ids': ids}) + '\n')
