"""Admission ordering. The engine integration must provide fresh live matches."""
from dataclasses import dataclass
from typing import Callable, Optional
from .core import Request

@dataclass(frozen=True)
class Decision:
    request: Request
    cycle_slot: int
    reason: str
    scores: dict

class AdmissionPolicy:
    def __init__(self, policy='fcfs', k=4, score='ratio', predictor=None):
        if policy not in ('fcfs', 'k-lpm', 'calibrated'): raise ValueError('unknown policy')
        if k < 1: raise ValueError('k must be positive')
        if score not in ('ratio', 'tokens'): raise ValueError('unknown score')
        self.policy, self.k, self.score, self.admissions = policy, k, score, 0
        self.predictor = predictor
        self._pending = None

    def select(self, requests, match: Callable[[Request], Optional[int]], eligible=lambda r: True):
        """Refresh scores on each call. Only ready and resource-eligible calls compete.

        Selection does not advance the cycle; call admitted only after successful admission.
        None from match means unavailable, not a cold-cache zero.
        """
        if self._pending is not None: raise RuntimeError('resolve prior selection first')
        ready = [r for r in requests if r.status == 'ready' and eligible(r)]
        if not ready: return None
        scores = {}
        fallback_scores = {}
        for r in ready:
            c = match(r)
            length = len(r.input_token_ids or [])
            if c is not None and (c < 0 or c > length): raise ValueError('invalid cache match')
            base = None if c is None or not length else (c / length if self.score == 'ratio' else c)
            fallback_scores[r.request_id] = base
            scores[r.request_id] = (self.predictor.predict(length,c) if self.policy == 'calibrated' and self.predictor is not None and c is not None else None) if self.policy == 'calibrated' else base
        oldest = lambda r: (r.ready_at, r.request_id)
        slot = self.admissions % self.k
        if self.policy == 'fcfs' or slot == 0:
            chosen = min(ready, key=oldest); reason = 'oldest'
        else:
            # Compare only like units: if any prediction is unavailable, use baseline for all candidates.
            fallback = self.policy == 'calibrated' and any(v is None for v in scores.values())
            if fallback: scores = fallback_scores
            known = [r for r in ready if scores[r.request_id] is not None]
            chosen = min(known, key=lambda r: (-scores[r.request_id], *oldest(r))) if known else min(ready, key=oldest)
            reason = ('baseline_fallback' if fallback else ('predicted_saving' if self.policy == 'calibrated' else 'cache_score')) if known else 'score_unavailable_oldest'
        self._pending = Decision(chosen, slot, reason, scores)
        return self._pending

    def admitted(self, decision):
        if decision is not self._pending: raise ValueError('not the pending selection')
        self.admissions += 1
        self._pending = None

    def rejected(self, decision):
        if decision is not self._pending: raise ValueError('not the pending selection')
        self._pending = None
