"""Bounded CPU-only audio preparation ahead of the main MLX inference thread."""
from __future__ import annotations

from queue import Empty, Full, Queue
from threading import Event, Thread


class PrefetchChunks:
    """Own a waveform iterator and prepare at most `depth + 1` chunks ahead.

    The producer alone advances/closes the source. No MLX operation belongs in the
    producer: it may decode FFmpeg PCM and run CPU VAD. Consume in a try/finally
    that calls close, including when the consumer fails or stops early.
    """

    def __init__(self, source, depth=2):
        if not 1 <= depth <= 8:
            raise ValueError("Prefetch depth must be1..8")
        self.source = iter(source)
        self.queue = Queue(maxsize=depth)
        self.stop = Event()
        self.closed = False
        self.thread = Thread(target=self._produce, name="qwen-audio-prefetch", daemon=True)
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
            while not self.stop.is_set():
                try:
                    value = next(self.source)
                except StopIteration:
                    break
                self._put("item", value)
        except BaseException as error:  # noqa: BLE001 -- Forward every producer failure to the consumer.
            failure = error
        finally:
            try:
                close = getattr(self.source, "close", None)
                if close:
                    close()
            except BaseException as error:  # noqa: BLE001 -- Preserve cleanup failures for the consumer.
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
                # The producer may enqueue completion between our timeout and
                # this liveness check. Recheck the queue before reporting failure.
                try:
                    kind, value = self.queue.get_nowait()
                except Empty:
                    raise RuntimeError("Audio producer ended without completion") from None
            if kind == "item":
                return value
            self.close()
            if kind == "error":
                raise value
            raise StopIteration

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError("Audio preparation did not stop within5s")
        # Release queued waveform buffers promptly on early consumer exit.
        while True:
            try:
                self.queue.get_nowait()
            except Empty:
                break
