#!/usr/bin/env python3
"""Freeze disjoint dataset row IDs and write prompts separately from labels."""
import argparse, json, pathlib, random
from datasets import load_dataset

p = argparse.ArgumentParser()
p.add_argument('--config', default='configs/foundation.json')
p.add_argument('--output', default='data')
a = p.parse_args()
c = json.loads(pathlib.Path(a.config).read_text())
rng = random.Random(c['manifest_seed'])
train = load_dataset(c['dataset'], c['dataset_config'], split='train', revision=c['dataset_revision'])
test = load_dataset(c['dataset'], c['dataset_config'], split='test', revision=c['dataset_revision'])
train_ids = rng.sample(range(len(train)), c['development_count'] + c['demonstration_count'])
test_ids = rng.sample(range(len(test)), c['evaluation_count'])
sets = {'development': ('train', train_ids[:c['development_count']]),
        'demonstrations': ('train', train_ids[c['development_count']:]), 'evaluation': ('test', test_ids)}
out = pathlib.Path(a.output); out.mkdir(parents=True, exist_ok=True)
manifest = {'dataset': c['dataset'], 'revision': c['dataset_revision'], 'config': c['dataset_config'],
            'seed': c['manifest_seed'], 'sets': {k: {'split': v[0], 'indices': v[1]} for k,v in sets.items()}}
(out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
for name, (split, ids) in sets.items():
    ds = train if split == 'train' else test
    questions, labels = [], []
    for i in ids:
        row = ds[i]; qid = '%s:%d' % (split, i)
        questions.append({'question_id': qid, 'question': row['question']})
        labels.append({'question_id': qid, 'answer': row['answer'].split('####')[-1].strip()})
    for suffix, records in [('questions', questions), ('labels', labels)]:
        (out / (name + '_' + suffix + '.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in records))
print('Saved manifest and separated questions/labels to', out)
