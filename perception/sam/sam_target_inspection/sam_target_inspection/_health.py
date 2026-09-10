#!/usr/bin/env python3
"""The health line every node in this package publishes, and the rules it obeys.

FORMAT: `STATE|rate_hz|age_s|n_in|n_out|detail` — the obstacle detector's, so an operator reads
one format across the perception stack and a bag can be diffed against another run.

THREE RULES THIS FILE EXISTS TO ENFORCE, each one paid for once:

1. **NO_INPUT NAMES THE SIDE.** "Nothing has arrived" is two different faults: nothing is
   producing, or something is producing and the subscription is not receiving it (QoS,
   discovery, a stale registration after a Unity stop/play — SETTLED §3f0r). The publisher count
   tells them apart and the detail says which. Never guess between them.
2. **A STATE CHANGE IS LOGGED FROM A SEPARATE CALL SITE PER SEVERITY.** rclpy caches logging
   state per call site and raises "Logger severity cannot be changed between calls" if one line
   logs at two severities — which killed the obstacle detector's health line the first time it
   ran on the rig (2026-08-12).
3. **A REFUSAL NAMES ITS REASON** and a number that was never measured is never printed.
"""
import math
from typing import Callable, Optional


class HealthLine:
    """State machine + formatter. Pure: the node hands it counts and it hands back a string."""

    def __init__(self, *, stale_timeout_s: float = 2.0,
                 publisher_count: Optional[Callable[[], int]] = None):
        self.stale_timeout_s = float(stale_timeout_s)
        self.publisher_count = publisher_count or (lambda: -1)
        self.last_in_at = None
        self.n_in = 0
        self.n_out = 0
        self.rate_hz = 0.0
        self._rate_t0 = None
        self._rate_n0 = 0
        self.state = None
        self.detail = ""
        self.blind_streak = 0
        self.last_reason = ""

    def note_input(self, now: float) -> None:
        self.last_in_at = now
        self.n_in += 1

    def note_output(self, n: int = 1) -> None:
        self.n_out += n

    def _rate(self, now: float) -> None:
        if self._rate_t0 is None:
            self._rate_t0, self._rate_n0 = now, self.n_in
            return
        dt = now - self._rate_t0
        if dt >= 2.0:
            self.rate_hz = (self.n_in - self._rate_n0) / dt
            self._rate_t0, self._rate_n0 = now, self.n_in

    def compute(self, now: float, *, blind_after: int = 5, ok_detail: str = "") -> str:
        self._rate(now)
        age = float("inf") if self.last_in_at is None else now - self.last_in_at
        if self.last_in_at is None:
            n = self.publisher_count()
            if n == 0:
                detail = "nothing since start; 0 publishers matched — nothing is producing"
            elif n > 0:
                detail = (f"nothing since start; {n} publisher(s) matched but nothing delivered "
                          f"— subscriber-side (QoS / discovery / stale registration)")
            else:
                detail = "nothing since start; publisher count unavailable"
            state = "NO_INPUT"
        elif age > self.stale_timeout_s:
            state, detail = "STALE", f"nothing for {age:.0f} s"
        elif self.blind_streak >= blind_after:
            state = "BLIND"
            detail = f"{self.blind_streak} consecutive: {self.last_reason}"
        else:
            state, detail = "OK", ok_detail
        self.state, self.detail = state, detail
        a = age if math.isfinite(age) else -1.0
        return f"{state}|{self.rate_hz:.1f}|{a:.1f}|{self.n_in}|{self.n_out}|{detail}"
