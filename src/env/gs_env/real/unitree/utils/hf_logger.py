import queue
import threading
from collections import deque
from collections.abc import Sequence

import numpy as np


class AsyncHFLogger:
    """
    Async logger with a bounded ring buffer of size max_seconds * rate_hz.
    Producer (RT thread) just queue.put(); consumer drains into deques.
    Stores q, dq, tau, and "relative dof positions" that you compute upstream.
    """

    def __init__(self, nj: int, max_seconds: float = 10.0, rate_hz: int = 1000) -> None:
        self.nj = int(nj)
        self.running = False

        self.maxlen = int(max_seconds * rate_hz)

        self.t_buf = deque(maxlen=self.maxlen)  # int64 ns
        self.q_buf = deque(maxlen=self.maxlen)  # (nj,)
        self.dq_buf = deque(maxlen=self.maxlen)  # (nj,)
        self.tau_buf = deque(maxlen=self.maxlen)  # (nj,)

        self.queue = queue.SimpleQueue()
        self.thread: threading.Thread | None = None

        # numpy views filled on stop()
        self.t_s = np.zeros((0,), dtype=np.int64)
        self.q_arr = np.zeros((0, self.nj), dtype=np.float32)
        self.dq_arr = np.zeros((0, self.nj), dtype=np.float32)
        self.tau_arr = np.zeros((0, self.nj), dtype=np.float32)

    def start(self) -> None:
        self.running = True
        self.thread = threading.Thread(target=self._consumer, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        self.queue.put(None)  # sentinel
        if self.thread is not None:
            self.thread.join()

        # convert to numpy once
        self.t_s = np.asarray(self.t_buf, dtype=np.int64)
        self.q_arr = np.asarray(self.q_buf, dtype=np.float32)
        self.dq_arr = np.asarray(self.dq_buf, dtype=np.float32)
        self.tau_arr = np.asarray(self.tau_buf, dtype=np.float32)

    def push(
        self,
        t_ns: int,
        q: Sequence[float],
        dq: Sequence[float],
        tau: Sequence[float],
    ) -> None:
        """Producer-side push (very lightweight)."""
        if self.running:
            self.queue.put((int(t_ns), tuple(q), tuple(dq), tuple(tau)))

    def _consumer(self) -> None:
        while True:
            item = self.queue.get()
            if item is None:
                break
            t_ns, q, dq, tau = item
            self.t_buf.append(t_ns)
            self.q_buf.append(q)
            self.dq_buf.append(dq)
            self.tau_buf.append(tau)

    def get(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns (t_ns, q_arr, dq_arr, tau_arr)."""
        return self.t_s, self.q_arr, self.dq_arr, self.tau_arr
