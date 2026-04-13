from __future__ import annotations

import time


class FPSCounter:
    def __init__(self, averaging: int = 30) -> None:
        self.averaging = max(1, averaging)
        self.prev_time = time.perf_counter()
        self.values = []

    def update(self) -> float:
        now = time.perf_counter()
        dt = max(now - self.prev_time, 1e-6)
        self.prev_time = now

        fps = 1.0 / dt
        self.values.append(fps)
        if len(self.values) > self.averaging:
            self.values.pop(0)

        return sum(self.values) / len(self.values)
