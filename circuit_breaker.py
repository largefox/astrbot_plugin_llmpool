"""Fine-grained circuit breaker with error classification and time-window tracking."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
import datetime
import enum
import json
from pathlib import Path
import time
from typing import Any


def get_seconds_until_next_daily_reset(reset_hour: int = 2) -> float:
    """Calculate seconds until next occurrences of reset_hour (e.g. 02:00)."""
    now = datetime.datetime.now()
    target = now.replace(hour=reset_hour, minute=0, second=0, microsecond=0)
    if now >= target:
        target += datetime.timedelta(days=1)
    diff = (target - now).total_seconds()
    return max(60.0, diff)


class ErrorCategory(str, enum.Enum):
    INSUFFICIENT_QUOTA = "InsufficientQuotaError"
    RATE_LIMIT = "RateLimitError"
    TIMEOUT = "TimeoutError"
    BAD_REQUEST = "BadRequestError"
    UPSTREAM_DEAD = "UpstreamDeadError"
    VISION_UNSUPPORTED = "VisionUnsupportedError"
    OTHER = "OtherError"


def classify_error(exc: Exception | str) -> ErrorCategory:
    """Classify exception or error message into standardized categories."""
    msg = str(exc).lower()
    # Check if upstream specifically rejects image/vision input
    if any(
        k in msg
        for k in (
            "not support image input",
            "not support image",
            "model is not a vlm",
            "not a vision model",
            "vision-capable model",
            "remove the image content",
            "does not support image",
            "does not support vision",
        )
    ):
        return ErrorCategory.VISION_UNSUPPORTED

    # Upstream distributor / channel failures (502/503/无可用渠道)
    if any(
        k in msg
        for k in (
            "503",
            "502",
            "无可用渠道",
            "upstream access forbidden",
            "distributor",
            "model_not_found",
            "upstream_error",
        )
    ):
        return ErrorCategory.UPSTREAM_DEAD

    # Upstream quota exhausted / arrears / insufficient balance (欠费 / 额度耗尽 / 余额不足)
    if any(
        k in msg
        for k in (
            "insufficient_quota",
            "insufficient quota",
            "insufficient balance",
            "exceeded your current quota",
            "quota_exceeded",
            "quota exceeded",
            "out of credits",
            "run out of credits",
            "balance not enough",
            "balance is not enough",
            "account_deactivated",
            "billing_not_active",
            "arrears",
            "欠费",
            "余额不足",
            "额度不足",
            "额度已用尽",
            "额度已耗尽",
            "无可用额度",
            "账户已欠费",
        )
    ):
        return ErrorCategory.INSUFFICIENT_QUOTA

    if any(
        k in msg
        for k in (
            "429",
            "rate limit",
            "ratelimit",
            "rate_limit",
            "too many requests",
            "requests per minute",
            "tokens per minute",
            "tpm",
            "rpm",
            "concurrency limit",
            "quota",
            "insufficient",
            "exceeded",
        )
    ):
        return ErrorCategory.RATE_LIMIT
    if any(k in msg for k in ("timeout", "timed out", "deadline", "asyncio.timeouterror")):
        return ErrorCategory.TIMEOUT
    if any(k in msg for k in ("400", "bad request", "invalid_request_error")):
        return ErrorCategory.BAD_REQUEST
    return ErrorCategory.OTHER


@dataclass
class NodeCircuitState:
    node_id: str
    is_open: bool = False
    cooldown_until: float = 0.0
    half_open: bool = False
    is_deep_sleep: bool = False
    deep_sleep_epoch: float = 0.0
    consecutive_failures: int = 0
    is_canary: bool = False
    # Store tuples of (timestamp, ErrorCategory) in sliding window (e.g. 60s)
    failures: deque[tuple[float, ErrorCategory]] = field(default_factory=deque)
    total_failures: int = 0
    total_successes: int = 0
    active_requests: int = 0
    # EWMA metrics for QoS
    ewma_ttft_ms: float = 1200.0  # default baseline 1.2s
    ewma_latency_ms: float = 2500.0  # default baseline 2.5s
    recent_call_times: deque[tuple[float, bool]] = field(default_factory=deque)  # (time, success)
    # Failure diagnostics & trip attribution
    last_error_msg: str = ""
    last_error_category: str = ""
    last_error_time: float = 0.0
    trip_reason: str = ""
    trip_error_msg: str = ""
    trip_time: float = 0.0

    def clean_old_failures(self, window_seconds: float = 60.0) -> None:
        now = time.monotonic()
        cutoff = now - window_seconds
        while self.failures and self.failures[0][0] < cutoff:
            self.failures.popleft()

    def update_qos_metrics(self, *, success: bool, latency_ms: float, ttft_ms: float | None = None) -> None:
        now = time.monotonic()
        self.recent_call_times.append((now, success))
        # Keep last 15 minutes of calls for SLA window
        cutoff = now - 900.0
        while self.recent_call_times and self.recent_call_times[0][0] < cutoff:
            self.recent_call_times.popleft()

        # Outlier clamping: clamp extreme latency spikes (e.g. 60s timeouts) so single spike doesn't ruin node
        clamped_latency = min(max(latency_ms, 100.0), 12000.0)

        # Cold-start adaptive alpha: fast initial calibration on first 5 samples, then smooth steady tracking
        sample_size = len(self.recent_call_times)
        alpha = 0.5 if sample_size <= 5 else 0.25

        self.ewma_latency_ms = alpha * clamped_latency + (1.0 - alpha) * self.ewma_latency_ms
        if ttft_ms is not None and ttft_ms > 0:
            clamped_ttft = min(max(ttft_ms, 50.0), 8000.0)
            self.ewma_ttft_ms = alpha * clamped_ttft + (1.0 - alpha) * self.ewma_ttft_ms

    def get_recent_availability(self, min_samples: int = 4) -> float:
        """Bayesian Laplace/m-estimate smoothed availability to prevent small-sample distortion.

        When sample size is small (< min_samples), blends toward prior assumption (95% healthy),
        preventing 1 failure out of 1 call from dropping availability to 0%.
        """
        if not self.recent_call_times:
            return 0.95  # Optimistic neutral prior for cold start

        actual_success = sum(1 for _, s in self.recent_call_times if s)
        actual_total = len(self.recent_call_times)

        # Prior: equivalent to 3 successes out of 3 trials (100%) or (2.85 / 3 = 95%)
        prior_success = 3.8
        prior_total = 4.0

        smoothed = (actual_success + prior_success) / (actual_total + prior_total)
        return min(max(smoothed, 0.0), 1.0)

    def reset_metrics(self) -> None:
        """Reset QoS metering profile to clean baseline state."""
        self.ewma_latency_ms = 2500.0
        self.ewma_ttft_ms = 1200.0
        self.recent_call_times.clear()
        self.failures.clear()
        self.consecutive_failures = 0
        self.last_error_msg = ""
        self.last_error_category = ""
        self.last_error_time = 0.0
        self.trip_reason = ""
        self.trip_error_msg = ""
        self.trip_time = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize state for persistent caching across restarts."""
        now_mono = time.monotonic()
        now_wall = time.time()
        # Convert monotonic cooldown_until to absolute epoch timestamp
        remaining_cd = max(0.0, self.cooldown_until - now_mono) if self.is_open else 0.0
        cooldown_epoch = (now_wall + remaining_cd) if remaining_cd > 0 else 0.0

        return {
            "node_id": self.node_id,
            "is_open": bool(self.is_open and remaining_cd > 0),
            "cooldown_epoch": cooldown_epoch,
            "is_deep_sleep": bool(self.is_deep_sleep and self.deep_sleep_epoch > now_wall),
            "deep_sleep_epoch": self.deep_sleep_epoch,
            "consecutive_failures": self.consecutive_failures,
            "is_canary": self.is_canary,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "ewma_ttft_ms": round(self.ewma_ttft_ms, 1),
            "ewma_latency_ms": round(self.ewma_latency_ms, 1),
            "last_error_msg": self.last_error_msg,
            "last_error_category": self.last_error_category,
            "last_error_time": self.last_error_time,
            "trip_reason": self.trip_reason,
            "trip_error_msg": self.trip_error_msg,
            "trip_time": self.trip_time,
        }

    def restore_from_dict(self, data: dict[str, Any]) -> None:
        """Restore warm state, preserving cooldown and EWMA timing profiles across restart."""
        now_mono = time.monotonic()
        now_wall = time.time()

        self.total_failures = int(data.get("total_failures", 0))
        self.total_successes = int(data.get("total_successes", 0))
        self.ewma_ttft_ms = float(data.get("ewma_ttft_ms", 1200.0))
        self.ewma_latency_ms = float(data.get("ewma_latency_ms", 2500.0))
        self.consecutive_failures = int(data.get("consecutive_failures", 0))
        self.is_canary = bool(data.get("is_canary", False))
        self.last_error_msg = str(data.get("last_error_msg") or "")
        self.last_error_category = str(data.get("last_error_category") or "")
        self.last_error_time = float(data.get("last_error_time", 0.0))
        self.trip_reason = str(data.get("trip_reason") or "")
        self.trip_error_msg = str(data.get("trip_error_msg") or "")
        self.trip_time = float(data.get("trip_time", 0.0))

        self.is_deep_sleep = bool(data.get("is_deep_sleep", False))
        self.deep_sleep_epoch = float(data.get("deep_sleep_epoch", 0.0))
        cooldown_epoch = float(data.get("cooldown_epoch", 0.0))

        if self.is_deep_sleep and self.deep_sleep_epoch > now_wall:
            remaining = self.deep_sleep_epoch - now_wall
            self.is_open = True
            self.half_open = False
            self.cooldown_until = now_mono + remaining
        elif data.get("is_open") and cooldown_epoch > now_wall:
            remaining = cooldown_epoch - now_wall
            self.is_open = True
            self.half_open = False
            self.cooldown_until = now_mono + remaining
        else:
            self.is_open = False
            self.half_open = False
            self.cooldown_until = 0.0
            self.is_deep_sleep = False


class CircuitBreaker:
    """Manages circuit breaking, error classification and cooldown across nodes."""

    def __init__(
        self,
        *,
        cooldown_time: float = 1800.0,
        allowed_fails: int = 1,
        aggressive_allowed_fails: int = 3,
        window_seconds: float = 60.0,
        allowed_fails_policy: dict[str, int] | None = None,
        persistence_path: Path | str | None = None,
        daily_reset_hour: int = 2,
        ratelimit_cooldown_seconds: float = 90.0,
    ) -> None:
        self.cooldown_time = cooldown_time
        self.allowed_fails = allowed_fails
        self.aggressive_allowed_fails = max(1, aggressive_allowed_fails)
        self.window_seconds = window_seconds
        self.daily_reset_hour = daily_reset_hour
        self.ratelimit_cooldown_seconds = ratelimit_cooldown_seconds
        self.allowed_fails_policy = allowed_fails_policy or {
            ErrorCategory.UPSTREAM_DEAD.value: 0,
            ErrorCategory.INSUFFICIENT_QUOTA.value: 0,
            ErrorCategory.RATE_LIMIT.value: 0,
            ErrorCategory.VISION_UNSUPPORTED.value: 9999,  # Do not circuit-break node for pure text requests!
            ErrorCategory.TIMEOUT.value: 2,
            ErrorCategory.BAD_REQUEST.value: 2,
        }
        self.persistence_path = Path(persistence_path) if persistence_path else None
        self._states: dict[str, NodeCircuitState] = {}
        self._lock = asyncio.Lock()
        self._load_persisted_state()

    def _load_persisted_state(self) -> None:
        """Load state on startup to retain warm EWMA timings and ongoing cooldowns."""
        if not self.persistence_path or not self.persistence_path.exists():
            return
        try:
            with open(self.persistence_path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for nid, node_data in data.items():
                    if isinstance(node_data, dict):
                        st = self.get_state(nid)
                        st.restore_from_dict(node_data)
        except Exception:
            pass

    def _persist_state(self) -> None:
        """Save current circuit and EWMA state to disk asynchronously or on change."""
        if not self.persistence_path:
            return
        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            export_data = {nid: st.to_dict() for nid, st in self._states.items()}
            with open(self.persistence_path, "w", encoding="utf-8-sig") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def get_state(self, node_id: str) -> NodeCircuitState:
        if node_id not in self._states:
            self._states[node_id] = NodeCircuitState(node_id=node_id)
        return self._states[node_id]

    async def is_available(self, node_id: str) -> bool:
        async with self._lock:
            state = self.get_state(node_id)
            now = time.monotonic()
            now_wall = time.time()

            # Check if deep sleep has reached daily reset time (e.g. 02:00)
            if state.is_deep_sleep and now_wall >= state.deep_sleep_epoch:
                state.is_deep_sleep = False
                state.is_open = False
                state.half_open = True
                state.is_canary = True
                state.cooldown_until = 0.0
                state.failures.clear()
                self._persist_state()
                return True

            if not state.is_open:
                return True
            # If cooldown has elapsed, allow one test trial in half-open state
            if now >= state.cooldown_until:
                state.half_open = True
                if state.is_deep_sleep:
                    state.is_deep_sleep = False
                    state.is_canary = True
                return True
            return False

    async def mark_success(
        self,
        node_id: str,
        latency_ms: float = 2000.0,
        ttft_ms: float | None = None,
    ) -> None:
        async with self._lock:
            state = self.get_state(node_id)
            state.total_successes += 1
            state.consecutive_failures = 0
            state.update_qos_metrics(success=True, latency_ms=latency_ms, ttft_ms=ttft_ms)
            if state.half_open or state.is_open or state.is_deep_sleep:
                # Successfully recovered
                state.is_open = False
                state.half_open = False
                state.is_deep_sleep = False
                state.is_canary = False
                state.cooldown_until = 0.0
                state.failures.clear()
                state.trip_reason = ""
                state.trip_error_msg = ""
                state.trip_time = 0.0
            self._persist_state()

    async def mark_failure(
        self,
        node_id: str,
        exc: Exception | str,
        custom_cooldown: float | None = None,
        latency_ms: float = 2000.0,
        is_aggressive: bool = False,
    ) -> tuple[bool, ErrorCategory, float]:
        """Record failure and check if circuit should trip.

        Returns (tripped, category, cooldown_until).
        """
        category = classify_error(exc)
        err_text = str(exc)[:500]
        now = time.monotonic()
        now_wall = time.time()

        async with self._lock:
            state = self.get_state(node_id)
            state.total_failures += 1
            state.consecutive_failures += 1
            state.last_error_msg = err_text
            state.last_error_category = category.value
            state.last_error_time = now_wall
            state.update_qos_metrics(success=False, latency_ms=latency_ms)
            state.clean_old_failures(self.window_seconds)
            state.failures.append((now, category))

            # Relaxed cooldown logic:
            # If custom cooldown is specified (e.g. aggressive mode 45s or node override),
            # apply mild exponential backoff on repeated failures (capped at 180s), NEVER locking into deep sleep!
            if custom_cooldown is not None:
                backoff_factor = min(4.0, 1.0 + (max(0, state.consecutive_failures - 1) * 0.5))
                cooldown = custom_cooldown * backoff_factor
                allow_deep_sleep = False
            elif category == ErrorCategory.RATE_LIMIT:
                cooldown = self.ratelimit_cooldown_seconds
                allow_deep_sleep = False
            elif category == ErrorCategory.INSUFFICIENT_QUOTA:
                cooldown = self.cooldown_time
                allow_deep_sleep = not is_aggressive
            else:
                cooldown = self.cooldown_time
                allow_deep_sleep = not is_aggressive

            # Relaxed deep-sleep qualification: only for persistent dead channel (502/UPSTREAM_DEAD)
            # or insufficient quota, or long consecutive failures (>= 4) with zero availability on standard nodes.
            should_deep_sleep = allow_deep_sleep and (
                category in {ErrorCategory.UPSTREAM_DEAD, ErrorCategory.INSUFFICIENT_QUOTA}
                or (
                    state.consecutive_failures >= 4
                    and state.get_recent_availability() < 0.15
                )
            )

            if category == ErrorCategory.INSUFFICIENT_QUOTA:
                trip_reason = "上游账户欠费/额度耗尽 (Insufficient Quota)"
            elif category == ErrorCategory.RATE_LIMIT:
                trip_reason = f"触发 429 限频保护 (Rate Limit，冷却 {int(cooldown)}s)"
            elif category == ErrorCategory.UPSTREAM_DEAD:
                trip_reason = "上游渠道彻底故障/不可用 (502/503/无可用渠道)"
            elif category == ErrorCategory.TIMEOUT:
                trip_reason = f"连续超时次数超限 ({state.consecutive_failures} 次)"
            elif category == ErrorCategory.BAD_REQUEST:
                trip_reason = f"连续 Bad Request (400) 错误超限 ({state.consecutive_failures} 次)"
            else:
                trip_reason = f"连续异常超限 ({category.value})"

            # If in half-open state, failed trial trips back to open with mild backoff
            if state.half_open:
                state.is_open = True
                state.half_open = False
                state.trip_reason = f"探活试探失败: {trip_reason}"
                state.trip_error_msg = err_text
                state.trip_time = now_wall
                if should_deep_sleep:
                    deep_sleep_secs = get_seconds_until_next_daily_reset(self.daily_reset_hour)
                    state.is_deep_sleep = True
                    state.deep_sleep_epoch = time.time() + deep_sleep_secs
                    state.cooldown_until = now + deep_sleep_secs
                else:
                    state.cooldown_until = now + cooldown
                self._persist_state()
                return True, category, state.cooldown_until

            # Count occurrences by category in recent window
            category_fails = sum(1 for _, cat in state.failures if cat == category)
            total_window_fails = len(state.failures)

            tripped = False
            has_category_policy = category.value in self.allowed_fails_policy
            if has_category_policy:
                threshold = self.allowed_fails_policy[category.value]
                if is_aggressive:
                    if category in {ErrorCategory.INSUFFICIENT_QUOTA, ErrorCategory.UPSTREAM_DEAD}:
                        effective_threshold = threshold
                    elif category == ErrorCategory.RATE_LIMIT:
                        effective_threshold = max(threshold, 1)
                    else:
                        effective_threshold = max(threshold, self.aggressive_allowed_fails)
                else:
                    effective_threshold = threshold

                # If threshold is 0 (like RateLimit / Quota), trip immediately.
                # If threshold > 0 (like Timeout=2), require category_fails to strictly exceed threshold
                if category_fails > effective_threshold:
                    tripped = True
            elif category in {ErrorCategory.UPSTREAM_DEAD, ErrorCategory.INSUFFICIENT_QUOTA}:
                tripped = True
            else:
                base_threshold = self.aggressive_allowed_fails if is_aggressive else self.allowed_fails
                trip_threshold = max(1, base_threshold)
                if total_window_fails > trip_threshold:
                    tripped = True

            if tripped:
                state.is_open = True
                state.half_open = False
                state.trip_reason = trip_reason
                state.trip_error_msg = err_text
                state.trip_time = now_wall
                if should_deep_sleep:
                    deep_sleep_secs = get_seconds_until_next_daily_reset(self.daily_reset_hour)
                    state.is_deep_sleep = True
                    state.deep_sleep_epoch = time.time() + deep_sleep_secs
                    state.cooldown_until = now + deep_sleep_secs
                else:
                    state.cooldown_until = now + cooldown
                self._persist_state()
                return True, category, state.cooldown_until

            self._persist_state()
            return False, category, 0.0

    async def reset(self, node_id: str | None = None, reset_metrics: bool = True) -> None:
        """Reset circuit breaker state and optionally QoS EWMA metering for a specific node or all nodes."""
        async with self._lock:
            target_id = str(node_id).strip() if node_id else None
            if target_id:
                st = self.get_state(target_id)
                st.is_open = False
                st.half_open = False
                st.is_deep_sleep = False
                st.is_canary = False
                st.consecutive_failures = 0
                st.cooldown_until = 0.0
                st.failures.clear()
                st.trip_reason = ""
                st.trip_error_msg = ""
                st.trip_time = 0.0
                st.last_error_msg = ""
                st.last_error_category = ""
                st.last_error_time = 0.0
                if reset_metrics:
                    st.reset_metrics()
            else:
                for st in self._states.values():
                    st.is_open = False
                    st.half_open = False
                    st.is_deep_sleep = False
                    st.is_canary = False
                    st.consecutive_failures = 0
                    st.cooldown_until = 0.0
                    st.failures.clear()
                    st.trip_reason = ""
                    st.trip_error_msg = ""
                    st.trip_time = 0.0
                    st.last_error_msg = ""
                    st.last_error_category = ""
                    st.last_error_time = 0.0
                    if reset_metrics:
                        st.reset_metrics()
            self._persist_state()

    async def soft_reset_healthy_qos_metrics(self) -> None:
        """Reset QoS EWMA metrics for HEALTHY nodes only, strictly keeping cooling/dead nodes locked."""
        async with self._lock:
            now = time.monotonic()
            for st in self._states.values():
                is_cooling = (st.is_open and st.cooldown_until > now) or (st.is_deep_sleep and st.cooldown_until > now)
                if not is_cooling:
                    st.reset_metrics()
            self._persist_state()

    async def get_all_status(self) -> dict[str, dict[str, Any]]:
        async with self._lock:
            now = time.monotonic()
            results = {}
            for nid, st in self._states.items():
                st.clean_old_failures(self.window_seconds)
                remaining_cd = max(0.0, st.cooldown_until - now) if st.is_open else 0.0
                if st.is_deep_sleep and remaining_cd > 0:
                    status_str = "DEEP-SLEEP"
                elif st.is_open and remaining_cd > 0:
                    status_str = "OPEN"
                elif st.half_open:
                    status_str = "HALF-OPEN"
                else:
                    status_str = "CLOSED"

                results[nid] = {
                    "node_id": nid,
                    "status": status_str,
                    "is_open": st.is_open and remaining_cd > 0,
                    "is_deep_sleep": st.is_deep_sleep and remaining_cd > 0,
                    "cooldown_remaining_seconds": round(remaining_cd, 1),
                    "active_requests": st.active_requests,
                    "consecutive_failures": st.consecutive_failures,
                    "total_successes": st.total_successes,
                    "total_failures": st.total_failures,
                    "recent_failures_in_window": len(st.failures),
                    "ewma_ttft_ms": round(st.ewma_ttft_ms, 1),
                    "ewma_latency_ms": round(st.ewma_latency_ms, 1),
                    "recent_availability_pct": round(st.get_recent_availability() * 100.0, 1),
                    "last_error_msg": st.last_error_msg,
                    "last_error_category": st.last_error_category,
                    "last_error_time": st.last_error_time,
                    "trip_reason": st.trip_reason,
                    "trip_error_msg": st.trip_error_msg,
                    "trip_time": st.trip_time,
                }
            return results
