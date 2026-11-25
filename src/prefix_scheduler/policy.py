"""Admission ordering. The engine integration must provide fresh live matches."""
from dataclasses import dataclass, field
import math
from typing import Callable, Optional
from .core import Request

@dataclass(frozen=True)
class Decision:
    request: Request
    cycle_slot: int
    reason: str
    scores: dict
    diagnostics: dict = field(default_factory=dict)

class AdmissionPolicy:
    def __init__(self, policy='fcfs', k=4, score='ratio', predictor=None, unlock_bonus_ms=0.0):
        if policy not in ('fcfs', 'k-lpm', 'calibrated', 'dependency-aware'): raise ValueError('unknown policy')
        if k < 1: raise ValueError('k must be positive')
        if score not in ('ratio', 'tokens'): raise ValueError('unknown score')
        self.policy, self.k, self.score, self.admissions = policy, k, score, 0
        if not math.isfinite(unlock_bonus_ms) or unlock_bonus_ms < 0: raise ValueError('unlock bonus must be finite and nonnegative')
        self.predictor = predictor
        self.unlock_bonus_ms = unlock_bonus_ms
        self._pending = None

    def select(self, requests, match: Callable[[Request], Optional[int]], eligible=lambda r: True, unlocks=None):
        """Refresh scores on each call. Only ready and resource-eligible calls compete.

        Selection does not advance the cycle; call admitted only after successful admission.
        None from match means unavailable, not a cold-cache zero.
        """
        if self._pending is not None: raise RuntimeError('resolve prior selection first')
        ready = [r for r in requests if r.status == 'ready' and eligible(r)]
        if not ready: return None
        scores = {}
        fallback_scores = {}
        diagnostics = {}
        for r in ready:
            c = match(r)
            length = len(r.input_token_ids or [])
            if c is not None and (c < 0 or c > length): raise ValueError('invalid cache match')
            base = None if c is None or not length else (c / length if self.score == 'ratio' else c)
            fallback_scores[r.request_id] = base
            predicted = (self.predictor.predict(length, c) if self.predictor is not None and c is not None else None)
            if predicted is not None and (not math.isfinite(predicted) or predicted < 0): raise ValueError('invalid predicted saving')
            count = unlocks(r) if self.policy == 'dependency-aware' and unlocks is not None else None
            if count is not None and (not isinstance(count, int) or isinstance(count, bool) or count < 0): raise ValueError('invalid unlock count')
            diagnostics[r.request_id] = {'matched_tokens': c, 'predicted_saving_ms': predicted, 'unlock_count': count}
            if self.policy == 'calibrated': scores[r.request_id] = predicted
            elif self.policy == 'dependency-aware': scores[r.request_id] = None if predicted is None or count is None else predicted + self.unlock_bonus_ms * count
            else: scores[r.request_id] = base
        oldest = lambda r: (r.ready_at, r.request_id)
        slot = self.admissions % self.k
        if self.policy == 'fcfs' or slot == 0:
            chosen = min(ready, key=oldest); reason = 'oldest'
        else:
            # Compare only like units: if any prediction is unavailable, use baseline for all candidates.
            fallback = self.policy in ('calibrated', 'dependency-aware') and any(v is None for v in scores.values())
            if fallback: scores = fallback_scores
            known = [r for r in ready if scores[r.request_id] is not None]
            chosen = min(known, key=lambda r: (-scores[r.request_id], *oldest(r))) if known else min(ready, key=oldest)
            reason = ('baseline_fallback' if fallback else ('dependency_score' if self.policy == 'dependency-aware' else ('predicted_saving' if self.policy == 'calibrated' else 'cache_score'))) if known else 'score_unavailable_oldest'
        self._pending = Decision(chosen, slot, reason, scores, diagnostics)
        return self._pending

    def admitted(self, decision):
        if decision is not self._pending: raise ValueError('not the pending selection')
        self.admissions += 1
        self._pending = None

    def rejected(self, decision):
        if decision is not self._pending: raise ValueError('not the pending selection')
        self._pending = None
