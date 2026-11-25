"""Materialized reasoning DAGs for the concurrent live SGLang workload driver.

The graph is deliberately separate from transport.  Every node exists before
the question arrives; only ready nodes are sent to SGLang.  ToT's selected
candidate is dynamic, so its next-round nodes depend on the evaluator (which
already depends on both candidates) and select the evaluator's result while
their prompt is built.
"""
from collections import Counter
import re

from .core import Request, now, parse_answer, prompt_for
from .reasoning import seed_for

SELECTED = re.compile(r"(?im)^\s*selected\s*:\s*([12])\s*$")


class WorkloadGraph:
    def __init__(self, question_id, question, mode, arrival, max_new_tokens):
        self.question_id = question_id
        self.question = question
        self.mode = mode
        self.arrival = arrival
        self.max_new_tokens = max_new_tokens
        self.records = []
        self.by_id = {}

    def add(self, node, stage, parents=()):
        request_id = self.question_id + ':' + node
        request = Request(
            self.question_id, request_id, stage, self.arrival, now(),
            parent_ids=[p.request_id for p in parents],
            sampling_seed=seed_for(self.question_id, node),
            max_new_tokens=self.max_new_tokens,
        )
        self.records.append(request)
        self.by_id[request_id] = request
        return request

    def node(self, suffix):
        return self.by_id[self.question_id + ':' + suffix]

    def messages_for(self, request):
        """Build a request only after all of its logical parents completed."""
        if self.mode in ('direct', 'cot'):
            return prompt_for(self.question, self.mode), 0.0
        if self.mode == 'self_consistency':
            return ([
                {'role': 'system', 'content': 'Solve step by step. End with Final answer: <number>.'},
                {'role': 'user', 'content': self.question},
            ], 0.7)
        if request.stage == 'candidate_generation':
            round_no = request.request_id.rsplit(':', 1)[1][1]
            selected = ''
            if round_no == '2':
                selected_request, _ = self._selected(1)
                selected = '\nSelected partial solution:\n' + selected_request.output
            return ([{'role': 'user', 'content':
                     'Question: ' + self.question + selected +
                     '\nProduce a distinct partial solution, without completing the final answer.'}], 0.7)
        if request.stage == 'candidate_evaluation':
            round_no = request.request_id.rsplit(':', 1)[1][1]
            left = self.node('r%sc1' % round_no)
            right = self.node('r%sc2' % round_no)
            return ([{'role': 'user', 'content':
                     'Question: ' + self.question + '\nCandidate 1:\n' + left.output +
                     '\nCandidate 2:\n' + right.output +
                     '\nEvaluate mathematical consistency. End with Selected: 1 or Selected: 2.'}], 0.0)
        if request.stage == 'final_synthesis':
            selected, _ = self._selected(2)
            return ([{'role': 'user', 'content':
                     'Question: ' + self.question + '\nSelected partial solution:\n' + selected.output +
                     '\nComplete the solution. End with Final answer: <number>.'}], 0.0)
        raise ValueError('unknown graph stage: ' + request.stage)

    def _selected(self, round_no):
        evaluator = self.node('r%seval' % round_no)
        match = SELECTED.search(evaluator.output or '')
        return self.node('r%sc%s' % (round_no, match.group(1) if match else '1')), match is None


def materialize(question_id, question, mode, arrival, max_new_tokens):
    """Create every DAG node before dispatch; all records begin blocked."""
    graph = WorkloadGraph(question_id, question, mode, arrival, max_new_tokens)
    if mode in ('direct', 'cot'):
        graph.add(mode, mode)
    elif mode == 'self_consistency':
        for index in range(1, 4):
            graph.add('sample%d' % index, 'self_consistency_sampling')
    elif mode == 'tot':
        c11 = graph.add('r1c1', 'candidate_generation')
        c12 = graph.add('r1c2', 'candidate_generation')
        e1 = graph.add('r1eval', 'candidate_evaluation', [c11, c12])
        c21 = graph.add('r2c1', 'candidate_generation', [e1])
        c22 = graph.add('r2c2', 'candidate_generation', [e1])
        e2 = graph.add('r2eval', 'candidate_evaluation', [c21, c22])
        graph.add('final', 'final_synthesis', [e2])
    else:
        raise ValueError('unsupported mode: ' + mode)
    return graph


def release(graph):
    """Advance blocked nodes after a state change; return ready and failed nodes.

    A descendant of an HTTP failure is never sent to the engine.  The loop
    continues so that failure propagates through the complete materialized DAG.
    """
    ready, failed = [], []
    changed = True
    while changed:
        changed = False
        for request in graph.records:
            if request.status != 'blocked':
                continue
            parents = [graph.by_id[parent_id] for parent_id in request.parent_ids]
            if any(parent.status == 'failed' for parent in parents):
                request.error = 'parent request failed'
                request.transition('failed')
                failed.append(request)
                changed = True
            elif all(parent.status == 'completed' for parent in parents):
                request.transition('ready')
                ready.append(request)
                changed = True
    return ready, failed


def question_result(graph):
    """Compute the existing four-mode result from the final graph snapshot."""
    finished_at = max((request.finished_at for request in graph.records if request.finished_at), default=now())
    if graph.mode in ('direct', 'cot'):
        request = graph.records[0]
        answer = parse_answer(request.output) if request.status == 'completed' else None
        error = request.error if request.status == 'failed' else (None if answer is not None else 'unparseable answer')
        return {'question_id': graph.question_id, 'mode': graph.mode,
                'status': 'completed' if answer is not None else 'failed', 'answer': answer,
                'error': error, 'finished_at': finished_at}
    if graph.mode == 'self_consistency':
        answers = [parse_answer(request.output) if request.status == 'completed' else None for request in graph.records]
        valid = [answer for answer in answers if answer is not None]
        counts = Counter(valid)
        answer = next((item for item in answers if item is not None and counts[item] == max(counts.values())), None) if counts else None
        return {'question_id': graph.question_id, 'mode': graph.mode,
                'status': 'completed' if answer is not None else 'failed', 'answer': answer,
                'branch_answers': answers, 'disagreement': len(set(valid)) > 1,
                'parse_failures': sum(item is None for item in answers), 'finished_at': finished_at}
    final = graph.node('final')
    answer = parse_answer(final.output) if final.status == 'completed' else None
    selections = []
    evaluator_failures = 0
    for round_no in (1, 2):
        evaluator = graph.node('r%seval' % round_no)
        if evaluator.status == 'completed':
            selected, fallback = graph._selected(round_no)
            selections.append({'round': round_no, 'selected': selected.request_id, 'fallback': fallback})
            evaluator_failures += int(fallback)
    return {'question_id': graph.question_id, 'mode': 'tot',
            'status': 'completed' if answer is not None else 'failed', 'answer': answer,
            'error': None if answer is not None else (final.error or 'missing final answer'),
            'evaluator_failures': evaluator_failures, 'selections': selections,
            'finished_at': finished_at}
