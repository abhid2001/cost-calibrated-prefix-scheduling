"""CUDA event timing primitive for single-request, unchunked engine prefill.

This module does not hook SGLang by itself. Record both events on the stream
that executes the prefill forward pass; synchronize the end event before
reading elapsed time. It must not be used to attribute batched work to a
single request.
"""
import math

class PrefillTimer:
    def __init__(self, cuda, *, batch_size, chunked, overlap):
        if batch_size != 1 or chunked or overlap:
            raise ValueError('timing requires one request, no chunking and no overlap')
        self.cuda = cuda
        self.start_event = cuda.Event(enable_timing=True)
        self.end_event = cuda.Event(enable_timing=True)
        self.state = 'new'

    def start(self, stream):
        if self.state != 'new': raise RuntimeError('timer already started')
        self.stream = stream
        self.start_event.record(stream)
        self.state = 'running'

    def stop(self, stream):
        if self.state != 'running': raise RuntimeError('timer is not running')
        if stream != self.stream: raise ValueError('events must use the same stream')
        self.end_event.record(stream)
        self.state = 'stopped'

    def milliseconds(self):
        if self.state != 'stopped': raise RuntimeError('timer has not stopped')
        self.end_event.synchronize()
        value = float(self.start_event.elapsed_time(self.end_event))
        if not math.isfinite(value) or value < 0: raise RuntimeError('invalid CUDA event duration')
        return value
