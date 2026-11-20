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

def chat_token_ids(tokenizer, messages):
    """Chat-template token IDs as a flat list of ints.

    transformers 5 returns a BatchEncoding (input_ids and attention_mask) by default, which is
    not JSON-serializable and has len() == 2. Fail loudly if the type ever changes again.
    """
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
    if not isinstance(ids, list) or not all(isinstance(t, int) for t in ids):
        raise TypeError('expected a flat list of token IDs, got %s' % type(ids).__name__)
    return ids

def post_json(url, data, timeout=300):
    request = urllib.request.Request(url, json.dumps(data).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)

def generate(url, input_ids, sampling_params, timeout=300, record=None):
    """Stream SGLang's native /generate endpoint and return (text, meta_info).

    The server sends `data: {json}` lines and ends with `data: [DONE]`. In the default mode each
    chunk carries the cumulative text, so the final text and meta_info come from the last chunk.
    If a record is given, record.first_token_at is set when the first chunk reporting a generated
    token arrives. That timestamp is a client-side observation (it includes HTTP overhead): valid
    for TTFT, never a GPU prefill measurement.
    """
    body = json.dumps({'input_ids': input_ids, 'sampling_params': sampling_params, 'stream': True}).encode()
    request = urllib.request.Request(url, body, {'Content-Type': 'application/json'})
    last = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode('utf-8').strip()
            if not line.startswith('data:'): continue
            data = line[len('data:'):].strip()
            if data == '[DONE]': break
            chunk = json.loads(data)
            if 'error' in chunk: raise RuntimeError('engine error: %s' % (chunk['error'],))
            meta = chunk.get('meta_info') or {}
            # Prefer the engine's token count: the first token can decode to empty text.
            generated = meta['completion_tokens'] >= 1 if 'completion_tokens' in meta else bool(chunk.get('text'))
            if record is not None and record.first_token_at is None and generated:
                record.first_token_at = now()
            last = chunk
    if last is None: raise RuntimeError('stream ended without any output chunk')
    return last.get('text', ''), last.get('meta_info') or {}
