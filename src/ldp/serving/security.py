"""API-key authentication and a per-client token-bucket rate limit."""

from __future__ import annotations

import hmac
import threading
import time
from dataclasses import dataclass, field


class KeyStore:
    def __init__(self, keys: dict[str, str]) -> None:
        self._keys = keys  # key -> client name

    def client_for(self, presented: str | None) -> str | None:
        if not presented:
            return None
        found = None
        # Compare against every key in constant time so timing does not reveal which keys exist.
        for key, client in self._keys.items():
            if hmac.compare_digest(presented.encode(), key.encode()):
                found = client
        return found


@dataclass
class _Bucket:
    tokens: float
    updated: float


@dataclass
class RateLimiter:
    per_minute: int
    _buckets: dict[str, _Bucket] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def allow(self, client: str, cost: float = 1.0, now: float | None = None) -> tuple[bool, float]:
        """Return (allowed, seconds until a token is available)."""
        now = time.monotonic() if now is None else now
        rate = self.per_minute / 60.0
        with self._lock:
            bucket = self._buckets.setdefault(client, _Bucket(tokens=float(self.per_minute), updated=now))
            bucket.tokens = min(float(self.per_minute), bucket.tokens + (now - bucket.updated) * rate)
            bucket.updated = now
            if bucket.tokens >= cost:
                bucket.tokens -= cost
                return True, 0.0
            return False, (cost - bucket.tokens) / rate
