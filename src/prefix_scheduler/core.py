"""Shared state, answer parsing, prompt construction, and HTTP transport."""
import dataclasses
import datetime
import decimal
import json
import re
import urllib.request
from typing import Optional


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


@dataclasses.dataclass
class Request:
    question_id: str
    request_id: str
    stage: str
    question_arrival: str
    created_at: str
    parent_ids: list = dataclasses.field(default_factory=list)
    status: str = 'blocked'
    ready_at: Optional[str] = None
    admitted_at: Optional[str] = None
    first_token_at: Optional[str] = None
    finished_at: Optional[str] = None
    input_token_ids: Optional[list] = None
    prompt: Optional[str] = None
    matched_prefix_tokens: Optional[int] = None
    predicted_saving_ms: Optional[float] = None
    sampling_seed: Optional[int] = None
    max_new_tokens: int = 256
    output: Optional[str] = None
    error: Optional[str] = None
    engine_metadata: Optional[dict] = None
    client_duration_ms: Optional[float] = None

    def transition(self, state, timestamp=None):
        allowed = {'blocked': {'ready', 'failed'}, 'ready': {'running', 'failed'},
                   'running': {'completed', 'failed'}, 'completed': set(), 'failed': set()}
        if state not in allowed[self.status]:
            raise ValueError('invalid transition: %s -> %s' % (self.status, state))
        stamp = timestamp or now()
        self.status = state
        if state == 'ready': self.ready_at = stamp
        if state == 'running': self.admitted_at = stamp
        if state in ('completed', 'failed'): self.finished_at = stamp


NUMBER = re.compile(r'(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w])')
FINAL = re.compile(r'(?:final\s+answer\s*[:=]|####)\s*([^\n]+)', re.I)

def normalize_number(text):
    """Return a canonical decimal string only for one whole numeric value."""
    value = text.strip().replace(',', '').replace('$', '')
    if not re.fullmatch(r'[-+]?(?:\d+(?:\.\d+)?|\.\d+)', value): return None
    try:
        n = decimal.Decimal(value)
        if not n.is_finite(): return None
        return format(n.normalize(), 'f') if n else '0'
    except decimal.InvalidOperation:
        return None

def parse_answer(text):
    matches = FINAL.findall(text or '')
    if not matches: return None
    numbers = NUMBER.findall(matches[-1])
    return normalize_number(numbers[0]) if len(numbers) == 1 else None

def prompt_for(question, mode):
    if mode == 'direct':
        instruction = 'Answer the math question directly without explaining your work.'
    elif mode == 'cot':
        instruction = 'Solve the math question step by step.'
    else: raise ValueError('unsupported reasoning mode: ' + mode)
    return [{'role': 'system', 'content': instruction + ' End with exactly Final answer: <number>.'},
            {'role': 'user', 'content': question}]

def post_json(url, data, timeout=300):
    request = urllib.request.Request(url, json.dumps(data).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)
