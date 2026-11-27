"""Experimental single-request SGLang queue ordering adapter."""
import json
import math
import os
import time

from .calibration import LookupPredictor


class DependencySnapshot:
    # Reads the client's JSON copy of the DAG ({request_id, parent_ids, status} per request) written by load_reasoning_trace.py
    """Read the workload driver's atomically replaced materialized-DAG snapshot."""
    def __init__(self, path):
        self.path = path

    def unlocks(self, request_id):
        #to summarize, it checks for duplicate request_ids, unknown requests, skips child nodes that aren't blocked and counts the number of parents in completed state for a blocked child
        # 'q042:r1c2' => 1 if r1eval is blocked and its other parent r1c1 is completed; any read/parse problem => None
        try:
            with open(self.path) as handle:
                payload = json.load(handle)
            records = payload['requests']
            by_id = {record['request_id']: record for record in records}
            # duplicate request_ids (dict shrinks) or unknown request => refuse to guess
            if len(by_id) != len(records) or request_id not in by_id:
                return None
            total = 0
            for child in records:
                parents = child.get('parent_ids', [])
                if child.get('status') != 'blocked' or request_id not in parents:
                    continue
                others = [parent for parent in parents if parent != request_id]
                if all(parent in by_id and by_id[parent].get('status') == 'completed' for parent in others):
                    total += 1
            return total
        except (OSError, ValueError, KeyError, TypeError):
            return None


class LiveAdmission:
    # The policy SGLang actually calls: order() sorts its waiting queue, admitted() logs the pick
    #it sorts in descending order of assigned scores
    def __init__(self, mode, k=4, predictor=None, log_path=None, dependency_snapshot=None, unlock_bonus_ms=0.0):
        if mode not in ('fcfs', 'k-lpm', 'calibrated', 'dependency-aware') or k < 1:
            raise ValueError('invalid live policy')
        if not math.isfinite(unlock_bonus_ms) or unlock_bonus_ms < 0:
            raise ValueError('invalid unlock bonus')
        self.mode, self.k, self.predictor, self.log_path = mode, k, predictor, log_path
        self.dependency_snapshot, self.unlock_bonus_ms = dependency_snapshot, unlock_bonus_ms
        # k = fairness cycle length: every k-th admission (slot 0) goes to the oldest request; k=1 behaves like FCFS
        self.admissions = 0  # requests admitted so far; slot = admissions % k
        self.age = {}  # rid => monotonic time first seen in the queue (tie-breaker: older first)
        self.pending = None  # latest order() decision, logged by admitted()

    def order(self, queue, match):
        # Scores every waiting request and sorts SGLang's queue in place (best first); admits nothing itself.
        # slot 0 or fcfs => oldest | k-lpm => cached/length | calibrated => predicted_ms | dependency-aware => predicted_ms + bonus*unlocks
        #if the slot>0, then we prefer predicred prefill time as a metric over the baseline (klpm,i.e, cached/length).
        #If the predicted prefill of any request is unknown, we fallback and resort to an order on the basis of our baseline (klpm,i.e, cached/length)
        for req in queue:
            if req.rid not in self.age:
                self.age[req.rid] = time.monotonic_ns()
        data = []
        for req in queue:
            length = len(req.origin_input_ids)
            cached = match(req)
            base = cached / length if length else 0
            predicted = self.predictor.predict(length, cached) if self.predictor is not None else None
            unlock_count = self.dependency_snapshot.unlocks(req.rid) if self.mode == 'dependency-aware' else None
            dependency_score = None if predicted is None or unlock_count is None else predicted + self.unlock_bonus_ms * unlock_count
            data.append({'id': req.rid, 'cached_tokens': cached, 'length': length, 'baseline': base,
                         'predicted_ms': predicted, 'unlock_count': unlock_count,
                         'dependency_score': dependency_score, 'age': self.age[req.rid]})
        # one missing prediction => whole decision uses the k-LPM baseline (compare like units)
        fallback = ((self.mode == 'calibrated' and any(item['predicted_ms'] is None for item in data)) or
                    (self.mode == 'dependency-aware' and any(item['dependency_score'] is None for item in data)))
        slot = self.admissions % self.k

        def score(item):
            if self.mode == 'fcfs' or slot == 0:
                return 0
            if self.mode == 'k-lpm' or fallback:
                return item['baseline']
            if self.mode == 'calibrated':
                return item['predicted_ms']
            return item['dependency_score']

        for item in data:
            item['score'] = score(item)
        by_id = {item['id']: item for item in data}
        # highest score first, then oldest, then rid
        queue.sort(key=lambda req: (-score(by_id[req.rid]), self.age[req.rid], str(req.rid)))
        if self.mode == 'fcfs' or slot == 0:
            reason = 'oldest'
        elif fallback:
            reason = 'baseline_fallback'
        elif self.mode == 'dependency-aware':
            reason = 'dependency_score'
        else:
            reason = self.mode
        self.pending = {'slot': slot, 'reason': reason, 'candidates': data,
                        'ranked_ids': [req.rid for req in queue]}

    def admitted(self, admitted):
        # After SGLang admits exactly one request: log {pending decision, actual_id, admission, ts_ns}, advance the slot
        # ts_ns is the engine-side time later joined in as Request.admitted_at
        if not admitted:
            return
        if len(admitted) != 1:
            raise RuntimeError('live policy requires single admission')
        event = dict(self.pending or {}, actual_id=admitted[0].rid, admission=self.admissions, ts_ns=time.time_ns())
        self.admissions += 1
        self.age.pop(admitted[0].rid, None)
        self.pending = None
        if self.log_path:
            with open(self.log_path, 'a') as handle:
                handle.write(json.dumps(event) + '\n')


def from_environment():
    # Builds LiveAdmission from env vars set by the Slurm scripts: PREFIX_ADMISSION_POLICY (required), PREFIX_PREDICTOR_PATH,
    # PREFIX_DEPENDENCY_GRAPH_PATH, PREFIX_DEPENDENCY_UNLOCK_BONUS_MS (default 0), PREFIX_ADMISSION_K (default 4), PREFIX_ADMISSION_LOG
    mode = os.environ['PREFIX_ADMISSION_POLICY']
    path = os.environ.get('PREFIX_PREDICTOR_PATH')
    if path:
        with open(path) as handle:
            predictor = LookupPredictor.from_dict(json.load(handle))
    else:
        predictor = None
    if mode in ('calibrated', 'dependency-aware') and predictor is None:
        raise RuntimeError('%s policy requires PREFIX_PREDICTOR_PATH' % mode)
    snapshot = None
    bonus = 0.0
    if mode == 'dependency-aware':
        graph_path = os.environ.get('PREFIX_DEPENDENCY_GRAPH_PATH')
        if not graph_path:
            raise RuntimeError('dependency-aware policy requires PREFIX_DEPENDENCY_GRAPH_PATH')
        snapshot = DependencySnapshot(graph_path)
        bonus = float(os.environ.get('PREFIX_DEPENDENCY_UNLOCK_BONUS_MS', '0'))
    return LiveAdmission(mode, int(os.environ.get('PREFIX_ADMISSION_K', '4')), predictor,
                         os.environ.get('PREFIX_ADMISSION_LOG'), snapshot, bonus)
