from __future__ import annotations
import time
from typing import Callable, Any

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_time_seconds: int = 30):
        self.failure_threshold = failure_threshold
        self.recovery_time_seconds = recovery_time_seconds
        self.failures = 0
        self.open_until = 0.0

    def allow(self) -> bool:
        if self.open_until and time.time() < self.open_until:
            return False
        return True

    def record_success(self):
        self.failures = 0
        self.open_until = 0.0

    def record_failure(self):
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.open_until = time.time() + self.recovery_time_seconds

