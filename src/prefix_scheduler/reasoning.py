"""Dependency-aware, single-active-call self consistency and bounded ToT."""
from collections import Counter
import hashlib
import re
from .core import Request, now, parse_answer


def seed_for(question_id, node):
    return int.from_bytes(hashlib.sha256((question_id + ':' + node).encode()).digest()[:4], 'big') % (2**31)

class Controller:
    """execute(record, messages, temperature) returns output text or raises.

    The callback owns engine interaction; all timestamps here are controller-side.
    """
    def __init__(self, execute, max_new_tokens=256, prepare=None):
        self.execute = execute
        self.prepare = prepare
        self.max_new_tokens = max_new_tokens
        self.records = []
        self.evaluator_failures = 0

    def call(self, qid, arrival, node, stage, parents, messages, temperature):
        r = Request(qid, qid + ':' + node, stage, arrival, now(), parent_ids=[p.request_id for p in parents],
                    sampling_seed=seed_for(qid, node), max_new_tokens=self.max_new_tokens)
        self.records.append(r)
        if any(p.status != 'completed' for p in parents):
            r.error = 'parent request failed'; r.transition('failed'); return r
        try:
            if self.prepare is not None: self.prepare(r, messages)
            r.transition('ready'); r.transition('running')
            r.output = self.execute(r, messages, temperature)
            if not isinstance(r.output, str) or not r.output.strip(): raise ValueError('empty model output')
            r.transition('completed')
        except Exception as exc:
            r.error = '%s: %s' % (type(exc).__name__, exc)
            r.transition('failed')
        return r

    def self_consistency(self, qid, question, arrival=None):
        arrival = arrival or now()
        messages = [{'role':'system','content':'Solve step by step. End with Final answer: <number>.'},
                    {'role':'user','content':question}]
        branches = [self.call(qid, arrival, 'sample%d' % i, 'self_consistency_sampling', [], messages, 0.7) for i in range(1,4)]
        answers = [parse_answer(r.output) if r.status == 'completed' else None for r in branches]
        valid = [a for a in answers if a is not None]
        counts = Counter(valid)
        answer = next((a for a in answers if a is not None and counts[a] == max(counts.values())), None) if counts else None
        return {'question_id':qid, 'mode':'self_consistency', 'status':'completed' if answer is not None else 'failed',
                'answer':answer, 'branch_answers':answers, 'disagreement':len(set(valid)) > 1,
                'parse_failures':sum(a is None for a in answers), 'finished_at':now()}

    def tot(self, qid, question, arrival=None):
        arrival = arrival or now(); selected = None; prior_eval = None; selections = []
        for round_no in (1, 2):
            parents = [] if prior_eval is None else [selected, prior_eval]
            context = '' if selected is None else '\nSelected partial solution:\n' + selected.output
            prompt = 'Question: ' + question + context + '\nProduce a distinct partial solution, without completing the final answer.'
            candidates = [self.call(qid, arrival, 'r%dc%d' % (round_no,i), 'candidate_generation', parents,
                                    [{'role':'user','content':prompt}], 0.7) for i in (1,2)]
            if any(r.status != 'completed' for r in candidates):
                return {'question_id':qid,'mode':'tot','status':'failed','answer':None,'error':'candidate request failed', 'finished_at':now(), 'evaluator_failures':self.evaluator_failures}
            eval_prompt = ('Question: '+question+'\nCandidate 1:\n'+candidates[0].output+'\nCandidate 2:\n'+candidates[1].output+
                           '\nEvaluate mathematical consistency. End with Selected: 1 or Selected: 2.')
            ev = self.call(qid, arrival, 'r%deval' % round_no, 'candidate_evaluation', candidates,
                           [{'role':'user','content':eval_prompt}], 0.0)
            match = re.search(r'(?im)^\s*selected\s*:\s*([12])\s*$', ev.output or '') if ev.status == 'completed' else None
            if not match:
                self.evaluator_failures += 1
                # A failed evaluator still leads to an explicit failure; malformed completed output uses candidate 1.
                if ev.status != 'completed':
                    return {'question_id':qid,'mode':'tot','status':'failed','answer':None,'error':'evaluator request failed', 'finished_at':now(), 'evaluator_failures':self.evaluator_failures}
            selected = candidates[int(match.group(1))-1 if match else 0]
            selections.append({'round':round_no, 'selected':selected.request_id, 'fallback':match is None})
            prior_eval = ev
        final = self.call(qid, arrival, 'final', 'final_synthesis', [selected,prior_eval],
                          [{'role':'user','content':'Question: '+question+'\nSelected partial solution:\n'+selected.output+
                            '\nComplete the solution. End with Final answer: <number>.'}], 0.0)
        answer = parse_answer(final.output) if final.status == 'completed' else None
        return {'question_id':qid,'mode':'tot','status':'completed' if answer is not None else 'failed',
                'answer':answer,'error':None if answer is not None else (final.error or 'missing final answer'),
                'evaluator_failures':self.evaluator_failures,'selections':selections,'finished_at':now()}
