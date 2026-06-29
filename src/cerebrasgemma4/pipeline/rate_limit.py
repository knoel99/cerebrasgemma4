"""Sliding-window rate limiter for Cerebras hackathon RPM / TPM quotas."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

# Hackathon elevated limits (docs/Gemma 4 Hackathon.md)
HACKATHON_RPM = 100
HACKATHON_TPM = 100_000


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


@dataclass
class CerebrasRateLimiter:
    """Pace API calls to stay under Cerebras RPM and TPM limits."""

    rpm_limit: int = field(default_factory=lambda: _env_int("CEREBRAS_RPM_LIMIT", HACKATHON_RPM))
    tpm_limit: int = field(default_factory=lambda: _env_int("CEREBRAS_TPM_LIMIT", HACKATHON_TPM))
    window_sec: float = 60.0
    rpm_headroom: float = 0.92
    tpm_headroom: float = 0.92
    default_token_estimate: int = 8_000

    _request_times: list[float] = field(default_factory=list, init=False, repr=False)
    _token_events: list[tuple[float, int]] = field(default_factory=list, init=False, repr=False)
    total_pause_sec: float = field(default=0.0, init=False)
    pause_events: int = field(default=0, init=False)

    @property
    def rpm_cap(self) -> int:
        return max(1, int(self.rpm_limit * self.rpm_headroom))

    @property
    def tpm_cap(self) -> int:
        return max(1, int(self.tpm_limit * self.tpm_headroom))

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_sec
        self._request_times = [t for t in self._request_times if t > cutoff]
        self._token_events = [(t, n) for t, n in self._token_events if t > cutoff]

    def _tokens_in_window(self) -> int:
        return sum(n for _, n in self._token_events)

    def _rpm_wait(self, now: float) -> float:
        if len(self._request_times) < self.rpm_cap:
            return 0.0
        oldest = self._request_times[0]
        return max(0.0, (oldest + self.window_sec) - now + 0.05)

    def _tpm_wait(self, now: float, token_estimate: int) -> float:
        used = self._tokens_in_window()
        if used + token_estimate <= self.tpm_cap:
            return 0.0
        overflow = used + token_estimate - self.tpm_cap
        released = 0
        for ts, count in self._token_events:
            released += count
            if released >= overflow:
                return max(0.0, (ts + self.window_sec) - now + 0.05)
        if self._token_events:
            ts, _ = self._token_events[0]
            return max(0.0, (ts + self.window_sec) - now + 0.05)
        return 0.0

    def wait_for_slot(self, *, token_estimate: int | None = None) -> float:
        """Block until the next request fits RPM/TPM budgets. Returns seconds slept."""
        estimate = token_estimate or self.default_token_estimate
        slept = 0.0

        while True:
            now = time.monotonic()
            self._prune(now)
            wait = max(self._rpm_wait(now), self._tpm_wait(now, estimate))
            if wait <= 0:
                break
            time.sleep(wait)
            slept += wait

        if slept > 0:
            self.total_pause_sec += slept
            self.pause_events += 1
        return slept

    def record(self, usage: dict | None) -> None:
        now = time.monotonic()
        self._request_times.append(now)

        usage = usage or {}
        total = int(usage.get("total_tokens") or 0)
        if total <= 0:
            total = int(usage.get("prompt_tokens") or 0) + int(
                usage.get("completion_tokens") or 0
            )
        if total > 0:
            self._token_events.append((now, total))
            self.default_token_estimate = max(2_000, int(total * 1.1))

    def snapshot(self) -> dict:
        now = time.monotonic()
        self._prune(now)
        return {
            "rpm_limit": self.rpm_limit,
            "tpm_limit": self.tpm_limit,
            "rpm_cap": self.rpm_cap,
            "tpm_cap": self.tpm_cap,
            "requests_in_window": len(self._request_times),
            "tokens_in_window": self._tokens_in_window(),
            "total_pause_sec": round(self.total_pause_sec, 2),
            "pause_events": self.pause_events,
        }


def estimate_pipeline_minutes(
    total_calls: int,
    *,
    rpm_limit: int = HACKATHON_RPM,
    tpm_limit: int = HACKATHON_TPM,
    tokens_per_call: int = 2_200,
    rpm_headroom: float = 0.92,
    tpm_headroom: float = 0.92,
    avg_call_sec: float = 0.7,
) -> float:
    """Legacy helper: rough wall-clock minutes from total call count."""
    if total_calls <= 0:
        return 0.0
    rpm_cap = max(1, int(rpm_limit * rpm_headroom))
    tpm_cap = max(1, int(tpm_limit * tpm_headroom))
    rpm_minutes = total_calls / rpm_cap
    tpm_minutes = (total_calls * tokens_per_call) / tpm_cap
    compute_minutes = (total_calls * avg_call_sec) / 60.0
    return max(rpm_minutes, tpm_minutes, compute_minutes)