"""Experimental bounded q8 encoder work on a producer-owned MLX GPU stream.

The main thread alone uses the decoder and tokenizer. The producer exclusively
advances/closes the waveform source and calls the read-only audio encoder. Only
fully evaluated feature arrays cross threads. No CLI/default installs this path.
"""
from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event, Thread

import mlx.core as mx
from mlx_qwen3_asr.audio import N_FFT, _hann_window, compute_features


@dataclass(frozen=True)
class EncodedChunk:
    index: int
    offset: float
    samples: int
    features: mx.array


class EncoderPrefetch:
    """At most depth queued features plus one producer chunk and consumer data.

The caller must not invoke/mutate this encoder while the producer is live. Model
parameters, positional buffers and the upstream cached Hann window are evaluated
on the owner thread before starting. A new GPU stream is created and used only
inside the producer. Always close in finally if consumption fails or stops early.
"""

    def __init__(self, source, encoder, depth=2):
        if type(depth) is not int or not 1 <= depth <= 2:
            raise ValueError("Encoder prefetch depth must be 1 or 2")
        mx.eval(encoder.parameters())
        position = getattr(getattr(encoder, "embed_positions", None), "_pe", None)
        mx.eval(position, _hann_window(N_FFT))
        self.source, self.encoder = iter(source), encoder
        self.queue = Queue(maxsize=depth)
        self.stop = Event()
        self.closed = False
        self.thread = Thread(target=self._produce, name="qwen-encoder-prefetch", daemon=True)
        self.thread.start()

    def _put(self, kind, value):
        while not self.stop.is_set():
            try:
                self.queue.put((kind, value), timeout=0.05)
                return
            except Full:
                continue

    def _produce(self):
        failure = None
        try:
            stream = mx.new_stream(mx.gpu)
            with mx.stream(stream):
                index = 0
                while not self.stop.is_set():
                    try:
                        wave, offset = next(self.source)
                    except StopIteration:
                        break
                    if not 0 < len(wave) <= 30 * 16000:
                        raise ValueError("Encoder prefetch requires 0 < audio <= 30 seconds")
                    mel, lengths = compute_features(wave)
                    features, output_lengths = self.encoder(mel.astype(mx.float16), lengths)
                    mx.eval(features, output_lengths)
                    item = EncodedChunk(index, offset, len(wave), features)
                    self._put("item", item)
                    index += 1
                    del item, wave, mel, lengths, features, output_lengths
                mx.synchronize(stream)
        except BaseException as error:  # noqa: BLE001 -- Preserve producer failures for the consumer.
            failure = error
        finally:
            try:
                close = getattr(self.source, "close", None)
                if close:
                    close()
            except BaseException as error:  # noqa: BLE001 -- Preserve source cleanup errors too.
                if failure is None:
                    failure = error
            self._put("error" if failure is not None else "end", failure)

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed:
            raise StopIteration
        while True:
            try:
                kind, value = self.queue.get(timeout=0.05)
            except Empty:
                if self.thread.is_alive():
                    continue
                try:
                    kind, value = self.queue.get_nowait()
                except Empty:
                    raise RuntimeError("Encoder producer ended without completion") from None
            if kind == "item":
                return value
            self.close()
            if kind == "error":
                raise value
            raise StopIteration

    def close(self):
        if self.closed:
            return
        self.stop.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError("Encoder producer did not stop within 5s")
        self.closed = True
        while True:
            try:
                self.queue.get_nowait()
            except Empty:
                break
