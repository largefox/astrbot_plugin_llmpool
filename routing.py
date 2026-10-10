"""Routing logic: support usage-based-routing-v2, weighted random, round-robin and fallback chains."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import random
import time
from typing import Any, Iterable, Sequence

try:
    from .circuit_breaker import CircuitBreaker
except ImportError:
    from circuit_breaker import CircuitBreaker


@dataclass(frozen=True, slots=True)
class PoolMember:
    provider_id: str
    weight: int = 1
    max_parallel_requests: int = 6
    rpm: int = 90
    supports_vision: bool = True
    model_name: str = ""
    group_name: str = ""
    aggressive_mode: bool = False
    timeout_seconds: float = 0.0
    cooldown_seconds: float = 0.0

    @property
    def group_names(self) -> tuple[str, ...]:
        """Return all assigned group names, supporting comma/semicolon/slash separation."""
        raw = self.group_name or ""
        import re
        parts = [p.strip() for p in re.split(r"[,;，；/|]+", raw) if p.strip()]
        return tuple(parts)

    def matches_group(self, target_group: str) -> bool:
        """Check if this member belongs to the specified group."""
        target = str(target_group or "").strip()
        groups = self.group_names or ("default",)
        return target in groups or self.group_name == target

    def get_effective_timeout(self, fallback_timeout: float = 22.0) -> float:
        """Calculate effective execution timeout for this member."""
        if self.timeout_seconds > 0.0:
            return min(max(self.timeout_seconds, 2.0), 300.0)
        if self.aggressive_mode:
            return 24.0  # Aggressive preset defaults to 24.0s timeout
        return fallback_timeout

    def get_effective_cooldown(self, default_cooldown: float = 1800.0) -> float:
        """Calculate relaxed/custom cooldown time for this member."""
        if self.cooldown_seconds > 0.0:
            return max(15.0, self.cooldown_seconds)
        if self.aggressive_mode:
            return 45.0  # Aggressive mode defaults to relaxed 45s quick recovery
        return default_cooldown


def normalize_members(raw_members: Any) -> list[PoolMember]:
    """Normalize plugin config entries into a stable, deduplicated member list.

    Supports comma/semicolon-separated groups on a single member, or merging multiple
    entries configured for the same provider_id into multiple group assignments.
    """
    if not isinstance(raw_members, list):
        return []

    result: list[PoolMember] = []
    seen: dict[str, PoolMember] = {}
    for raw in raw_members:
        if not isinstance(raw, dict):
            continue
        if raw.get("enabled", True) is False:
            continue

        provider_id = str(raw.get("provider_id") or "").strip()
        if not provider_id:
            continue

        try:
            weight = int(raw.get("weight", 1))
        except (TypeError, ValueError):
            weight = 1

        try:
            max_parallel = int(raw.get("max_parallel_requests", 6))
        except (TypeError, ValueError):
            max_parallel = 6

        try:
            rpm = int(raw.get("rpm", 90))
        except (TypeError, ValueError):
            rpm = 90

        supports_vision = bool(raw.get("supports_vision", True))
        model_name = str(raw.get("model_name") or "")
        raw_grp = raw.get("group_name", "")
        if isinstance(raw_grp, list):
            group_name = ", ".join(str(x).strip() for x in raw_grp if str(x).strip())
        else:
            group_name = str(raw_grp or "")

        aggressive_mode = bool(raw.get("aggressive_mode", False))
        try:
            timeout_sec = float(raw.get("timeout_seconds", 0.0) or 0.0)
        except (TypeError, ValueError):
            timeout_sec = 0.0
        try:
            cooldown_sec = float(raw.get("cooldown_seconds", 0.0) or 0.0)
        except (TypeError, ValueError):
            cooldown_sec = 0.0

        new_member = PoolMember(
            provider_id=provider_id,
            weight=max(1, weight),
            max_parallel_requests=max(1, max_parallel),
            rpm=max(1, rpm),
            supports_vision=supports_vision,
            model_name=model_name,
            group_name=group_name,
            aggressive_mode=aggressive_mode,
            timeout_seconds=max(0.0, timeout_sec),
            cooldown_seconds=max(0.0, cooldown_sec),
        )

        if provider_id in seen:
            old_member = seen[provider_id]
            merged_groups = list(dict.fromkeys([*old_member.group_names, *new_member.group_names]))
            merged_group_str = ", ".join(merged_groups) if merged_groups else (old_member.group_name or new_member.group_name)
            merged_member = PoolMember(
                provider_id=old_member.provider_id,
                weight=old_member.weight,
                max_parallel_requests=old_member.max_parallel_requests,
                rpm=old_member.rpm,
                supports_vision=old_member.supports_vision,
                model_name=old_member.model_name or new_member.model_name,
                group_name=merged_group_str,
                aggressive_mode=old_member.aggressive_mode,
                timeout_seconds=old_member.timeout_seconds,
                cooldown_seconds=old_member.cooldown_seconds,
            )
            seen[provider_id] = merged_member
            for idx, item in enumerate(result):
                if item.provider_id == provider_id:
                    result[idx] = merged_member
                    break
        else:
            seen[provider_id] = new_member
            result.append(new_member)

    return result


def rotate_members(
    members: Sequence[PoolMember],
    start: int,
) -> list[PoolMember]:
    if not members:
        return []
    index = start % len(members)
    return [*members[index:], *members[:index]]


def weighted_random_order(
    members: Sequence[PoolMember],
    *,
    rng: random.Random | None = None,
) -> list[PoolMember]:
    """Return a weighted order without replacement."""
    generator = rng or random
    remaining = list(members)
    ordered: list[PoolMember] = []

    while remaining:
        total_weight = sum(member.weight for member in remaining)
        needle = generator.random() * total_weight
        upto = 0.0
        chosen_index = len(remaining) - 1
        for index, member in enumerate(remaining):
            upto += member.weight
            if needle < upto:
                chosen_index = index
                break
        ordered.append(remaining.pop(chosen_index))

    return ordered


def unique_provider_ids(members: Iterable[PoolMember]) -> list[str]:
    return [member.provider_id for member in members]


class UsageBasedRouter:
    """Intelligent router matching LiteLLM's usage-based-routing-v2, with fallback chains."""

    def __init__(
        self,
        circuit_breaker: CircuitBreaker,
        *,
        model_group_alias: dict[str, str] | None = None,
        fallbacks: dict[str, list[str]] | None = None,
        model_request_interval_seconds: float = 0.0,
    ) -> None:
        self.circuit_breaker = circuit_breaker
        self.model_group_alias = model_group_alias or {
            "deepseek-chat": "gemini-3-flash",
            "fox-heavy-vmodel": "gemini-3-flash",
            "ds-official": "deepseek-official",
        }
        self.fallbacks = fallbacks or {
            "gemini-3-flash": ["gemini-3-flash-fallback"],
            "gemini-3-flash-fallback": ["agnes", "deepseek-official"],
            "agnes": ["deepseek-official"],
        }
        self.model_request_interval_seconds = float(model_request_interval_seconds or 0.0)
        self.manual_active_group: str | None = None
        self._congestion_suppressed_until: float = 0.0
        self._round_robin_cursor: dict[str, int] = {}
        self._congestion_request_counter: int = 0
        self._last_dispatched_at: dict[str, float] = {}
        self._last_model_dispatched_at: dict[str, float] = {}
        self._lock = asyncio.Lock()

    def mark_dispatched(self, member: PoolMember | str, model_name: str | None = None) -> None:
        """Record dispatch timestamp for a node/model to enforce anti-rate-limit spacing."""
        now = time.monotonic()
        if isinstance(member, PoolMember):
            pid = member.provider_id
            m_name = (member.model_name or "").strip()
        else:
            pid = str(member).strip()
            m_name = str(model_name or "").strip()
        if pid:
            self._last_dispatched_at[pid] = now
        if m_name:
            self._last_model_dispatched_at[m_name] = now

    def get_last_dispatched_at(self, provider_id: str) -> float:
        return self._last_dispatched_at.get(provider_id, 0.0)

    def get_last_model_dispatched_at(self, model_name: str) -> float:
        return self._last_model_dispatched_at.get(model_name, 0.0)

    def reset_dispatch_history(self, node_id: str | None = None) -> None:
        """Reset recent dispatch timestamps for a specific node or all nodes."""
        if node_id:
            self._last_dispatched_at.pop(node_id, None)
        else:
            self._last_dispatched_at.clear()
            self._last_model_dispatched_at.clear()

    async def set_manual_active_group(self, group_name: str | None) -> None:
        async with self._lock:
            self.manual_active_group = str(group_name).strip() if group_name else None

    async def reset_active_group(self, suppress_seconds: float = 600.0) -> None:
        """Reset active group to automatic mode and reset model group congestion promotion state."""
        async with self._lock:
            self.manual_active_group = None
            self._congestion_request_counter = 0
            self._congestion_suppressed_until = time.monotonic() + suppress_seconds

    async def notify_group_failure(self, group_name: str) -> None:
        """Prematurely clear congestion suppression when a real failure occurs on the primary group."""
        async with self._lock:
            self._congestion_suppressed_until = 0.0

    def get_default_group(self, group_to_members: dict[str, list[PoolMember]]) -> str:
        """Determine default starting group strictly based on alias, fallback rule sequence, or member presence."""
        resolved = self.resolve_target_group("default")
        if resolved != "default" and resolved in group_to_members:
            return resolved

        # First source_group defined in fallbacks rules that exists in pool
        for src_grp in self.fallbacks.keys():
            resolved_src = self.resolve_target_group(src_grp)
            if resolved_src in group_to_members:
                return resolved_src

        if "gemini-3-flash" in group_to_members:
            return "gemini-3-flash"
        if group_to_members:
            return next(iter(group_to_members.keys()))
        return "default"

    def build_fallback_chain(
        self,
        start_group: str,
        group_to_members: dict[str, list[PoolMember]],
    ) -> list[str]:
        """Build sequential fallback chain following user-defined fallback rules.

        Groups that are upstream in the fallback rules will NOT be appended circularly.
        Only unconfigured orphan groups (if any) are appended at the very end.
        """
        chain = self.get_fallback_chain(start_group)

        # Gather all groups mentioned in fallback rules
        configured_rule_groups: set[str] = set()
        for src, tgts in self.fallbacks.items():
            configured_rule_groups.add(self.resolve_target_group(src))
            for tgt in tgts:
                configured_rule_groups.add(self.resolve_target_group(tgt))

        start_is_dedicated = any(k in start_group.lower() for k in ("light", "dedicated"))

        # Only append unconfigured orphan groups (not in chain AND not in any fallback rules)
        for grp in group_to_members.keys():
            resolved = self.resolve_target_group(grp)
            if resolved in chain or resolved in configured_rule_groups:
                continue
            grp_is_dedicated = any(k in resolved.lower() for k in ("light", "dedicated"))
            # Never cross dedicated light groups with main fallback chains
            if not start_is_dedicated and grp_is_dedicated:
                continue
            if start_is_dedicated and not grp_is_dedicated:
                continue
            chain.append(resolved)

        return chain

    async def get_routing_status(
        self,
        members: list[PoolMember],
        enable_congestion_promotion: bool = True,
        congestion_latency_threshold_ms: float = 15000.0,
    ) -> dict[str, Any]:
        """Return dynamic routing status including configured default, manual override, active group, and promotion state."""
        # Group members by their normalized group_names
        group_to_members: dict[str, list[PoolMember]] = {}
        for m in members:
            for g in (m.group_names or ("default",)):
                grp = self.resolve_target_group(g or "default")
                group_to_members.setdefault(grp, []).append(m)

        available_groups = sorted(list(group_to_members.keys()))

        default_group = self.get_default_group(group_to_members)
        configured_chain = self.build_fallback_chain(default_group, group_to_members)

        start_group = self.manual_active_group or default_group
        fallback_chain = self.build_fallback_chain(start_group, group_to_members)

        is_manual_override = bool(self.manual_active_group)
        is_congested = False
        is_promoted = False
        active_group = start_group
        effective_chain = list(fallback_chain)

        # Check congestion promotion: sequentially advance along user's fallback rules to find first healthy group
        is_suppressed = (time.monotonic() < self._congestion_suppressed_until)
        if not is_manual_override and not is_suppressed and enable_congestion_promotion and len(fallback_chain) > 1:
            promoted_idx = 0
            while promoted_idx < len(fallback_chain) - 1:
                curr_grp = fallback_chain[promoted_idx]
                curr_members = group_to_members.get(curr_grp, [])
                if await self.is_group_congested(curr_members, latency_threshold_ms=congestion_latency_threshold_ms):
                    promoted_idx += 1
                else:
                    break

            if promoted_idx > 0:
                is_congested = True
                is_promoted = True
                active_group = fallback_chain[promoted_idx]
                effective_chain = fallback_chain[promoted_idx:]

        return {
            "default_group": default_group,
            "configured_chain": configured_chain,
            "manual_active_group": self.manual_active_group,
            "active_group": active_group,
            "is_manual_override": is_manual_override,
            "is_congested": is_congested,
            "is_promoted": is_promoted,
            "active_chain": effective_chain,
            "available_groups": available_groups,
        }

    def resolve_target_group(self, requested_group: str) -> str:
        """Resolve group alias (e.g. deepseek-chat -> gemini-3-flash)."""
        return self.model_group_alias.get(requested_group, requested_group)

    def get_fallback_chain(self, initial_group: str) -> list[str]:
        """Produce the ordered chain of groups to attempt, avoiding cycles."""
        target = self.resolve_target_group(initial_group)
        chain = [target]
        curr = target
        visited = {target}
        while True:
            next_groups = self.fallbacks.get(curr, [])
            found_next = False
            for ng in next_groups:
                resolved_ng = self.resolve_target_group(ng)
                if resolved_ng not in visited:
                    chain.append(resolved_ng)
                    visited.add(resolved_ng)
                    curr = resolved_ng
                    found_next = True
                    break
            if not found_next:
                break
        return chain

    async def order_group_candidates(
        self,
        members: list[PoolMember],
        strategy: str = "usage-based-routing-v2",
        require_vision: bool = False,
        request_interval_seconds: float | None = None,
    ) -> list[PoolMember]:
        """Order members in a group considering circuit state, in-flight load, weight, and anti-rate-limit interval."""
        if require_vision:
            members = [m for m in members if m.supports_vision]

        if not members:
            return []

        # Filter strictly by circuit breaker: only truly available nodes can be dispatched
        available: list[PoolMember] = []
        for m in members:
            if await self.circuit_breaker.is_available(m.provider_id):
                available.append(m)

        # If no nodes in this group are available, return empty to immediately trigger
        # fallback to the next group in fallback_chain without retrying cooling/dead nodes!
        if not available:
            return []

        selection_pool = available
        interval = (
            self.model_request_interval_seconds
            if request_interval_seconds is None
            else float(request_interval_seconds or 0.0)
        )

        now = time.monotonic()

        def _member_tier_and_elapsed(m: PoolMember) -> tuple[int, float]:
            prov_time = self._last_dispatched_at.get(m.provider_id, 0.0)
            prov_elapsed = now - prov_time if prov_time > 0.0 else float("inf")
            prov_cooling = prov_elapsed < interval

            m_name = (m.model_name or "").strip()
            model_time = self._last_model_dispatched_at.get(m_name, 0.0) if m_name else 0.0
            model_elapsed = now - model_time if (m_name and model_time > 0.0) else float("inf")
            model_cooling = (model_elapsed < interval) if m_name else False

            if not prov_cooling and not model_cooling:
                tier = 0
            elif not prov_cooling:
                tier = 1
            else:
                tier = 2
            return tier, prov_elapsed

        if strategy == "usage-based-routing-v2":
            scored = []
            for m in selection_pool:
                st = self.circuit_breaker.get_state(m.provider_id)
                load_ratio = st.active_requests / max(1, m.max_parallel_requests)

                # QoS factor: TTFT penalty and EWMA total latency penalty
                # TTFT penalty covers streaming start speed; latency penalty covers responsiveness.
                # Lower score is better.
                ttft_penalty = min(st.ewma_ttft_ms / 3000.0, 3.0)
                lat = max(st.ewma_latency_ms, 300.0)
                if lat <= 16000.0:
                    latency_penalty = lat / 5000.0
                else:
                    # Steep penalty for nodes exceeding normal completion window (e.g. 20s-50s timeouts)
                    latency_penalty = 3.2 + ((lat - 16000.0) / 3000.0)
                qos_latency_penalty = (ttft_penalty * 0.10) + (latency_penalty * 0.35)

                # Availability penalty / bonus:
                # Strictly penalize broken/failing nodes so they never steal requests from healthy nodes
                # even if broken nodes currently have 0 active_requests.
                sample_count = len(st.recent_call_times)
                if sample_count >= 2:
                    actual_success = sum(1 for _, s in st.recent_call_times if s)
                    raw_avail = actual_success / sample_count
                    if raw_avail < 0.20:
                        # Broken / dead node (e.g. 0% ~ 1.5% availability): heavy penalty pushes it behind healthy nodes
                        avail_penalty = 15.0
                    elif raw_avail < 0.60:
                        avail_penalty = 3.0 + (0.60 - raw_avail) * 7.5
                    else:
                        avail_penalty = -(raw_avail - 0.60) * 1.5
                else:
                    avail_penalty = 0.0

                # Canary trial penalty: trial nodes from deep-sleep/cooldown placed at tail of group
                canary_penalty = 5.0 if (st.half_open or getattr(st, "is_canary", False)) else 0.0

                # Aggressive node priority boost: healthy aggressive nodes get a priority boost to test/respond first
                aggressive_bonus = 0.35 if (m.aggressive_mode and (sample_count < 2 or raw_avail >= 0.60)) else 0.0

                weight_bonus = (m.weight - 1) * 0.05
                load_penalty = load_ratio * 0.40
                jitter = random.uniform(0.0, 0.02)

                score = (
                    load_penalty
                    + qos_latency_penalty
                    + avail_penalty
                    + canary_penalty
                    - weight_bonus
                    - aggressive_bonus
                    + jitter
                )

                if interval > 0.0 and len(selection_pool) > 1:
                    tier, prov_elapsed = _member_tier_and_elapsed(m)
                    scored.append(((tier, -prov_elapsed if tier == 2 else 0.0, score), m))
                else:
                    scored.append(((0, 0.0, score), m))

            scored.sort(key=lambda x: x[0])
            ordered = [m for _, m in scored]
        elif strategy == "weighted_random":
            if interval > 0.0 and len(selection_pool) > 1:
                t0, t1, t2 = [], [], []
                for m in selection_pool:
                    tier, _ = _member_tier_and_elapsed(m)
                    if tier == 0:
                        t0.append(m)
                    elif tier == 1:
                        t1.append(m)
                    else:
                        t2.append(m)
                t2.sort(key=lambda m: _member_tier_and_elapsed(m)[1], reverse=True)
                ordered = [
                    *weighted_random_order(t0),
                    *weighted_random_order(t1),
                    *t2,
                ]
            else:
                ordered = weighted_random_order(selection_pool)
        elif strategy == "ordered_failover":
            if interval > 0.0 and len(selection_pool) > 1:
                t0, t1, t2 = [], [], []
                for m in selection_pool:
                    tier, _ = _member_tier_and_elapsed(m)
                    if tier == 0:
                        t0.append(m)
                    elif tier == 1:
                        t1.append(m)
                    else:
                        t2.append(m)
                t2.sort(key=lambda m: _member_tier_and_elapsed(m)[1], reverse=True)
                ordered = [*t0, *t1, *t2]
            else:
                ordered = list(selection_pool)
        elif strategy == "least_busy":
            # Pure least-busy: pick nodes with lowest active_requests first
            if interval > 0.0 and len(selection_pool) > 1:
                t0, t1, t2 = [], [], []
                for m in selection_pool:
                    tier, _ = _member_tier_and_elapsed(m)
                    if tier == 0:
                        t0.append(m)
                    elif tier == 1:
                        t1.append(m)
                    else:
                        t2.append(m)
                t0.sort(key=lambda m: self.circuit_breaker.get_state(m.provider_id).active_requests)
                t1.sort(key=lambda m: self.circuit_breaker.get_state(m.provider_id).active_requests)
                t2.sort(key=lambda m: _member_tier_and_elapsed(m)[1], reverse=True)
                ordered = [*t0, *t1, *t2]
            else:
                ordered = sorted(
                    selection_pool,
                    key=lambda m: self.circuit_breaker.get_state(m.provider_id).active_requests,
                )
        elif strategy == "latency_based":
            # Pure latency-based: pick nodes with lowest EWMA latency first
            if interval > 0.0 and len(selection_pool) > 1:
                t0, t1, t2 = [], [], []
                for m in selection_pool:
                    tier, _ = _member_tier_and_elapsed(m)
                    if tier == 0:
                        t0.append(m)
                    elif tier == 1:
                        t1.append(m)
                    else:
                        t2.append(m)
                t0.sort(key=lambda m: self.circuit_breaker.get_state(m.provider_id).ewma_latency_ms)
                t1.sort(key=lambda m: self.circuit_breaker.get_state(m.provider_id).ewma_latency_ms)
                t2.sort(key=lambda m: _member_tier_and_elapsed(m)[1], reverse=True)
                ordered = [*t0, *t1, *t2]
            else:
                ordered = sorted(
                    selection_pool,
                    key=lambda m: self.circuit_breaker.get_state(m.provider_id).ewma_latency_ms,
                )
        else:  # round_robin
            if interval > 0.0 and len(selection_pool) > 1:
                t0, t1, t2 = [], [], []
                for m in selection_pool:
                    tier, _ = _member_tier_and_elapsed(m)
                    if tier == 0:
                        t0.append(m)
                    elif tier == 1:
                        t1.append(m)
                    else:
                        t2.append(m)
                t2.sort(key=lambda m: _member_tier_and_elapsed(m)[1], reverse=True)
                async with self._lock:
                    cursor = self._round_robin_cursor.get("default", 0)
                    self._round_robin_cursor["default"] = (cursor + 1) % max(1, len(selection_pool))
                ordered = [*rotate_members(t0, cursor), *rotate_members(t1, cursor), *t2]
            else:
                async with self._lock:
                    cursor = self._round_robin_cursor.get("default", 0)
                    self._round_robin_cursor["default"] = (cursor + 1) % max(1, len(selection_pool))
                ordered = rotate_members(selection_pool, cursor)

        if ordered:
            self.mark_dispatched(ordered[0])

        return ordered

    async def is_group_congested(
        self,
        group_members: list[PoolMember],
        latency_threshold_ms: float = 18000.0,
    ) -> bool:
        """Evaluate whether a model group is severely congested, degraded, or in outage.

        Uses the top-performing healthy available nodes (best 3) rather than a raw arithmetic mean
        of all nodes, preventing broken/dead outlier nodes from falsely dragging down the entire group.
        """
        if not group_members:
            return False

        available_members: list[PoolMember] = []
        total_count = len(group_members)

        for m in group_members:
            if await self.circuit_breaker.is_available(m.provider_id):
                available_members.append(m)

        available_count = len(available_members)

        # 1. Total outage: all nodes are tripped / cooling
        if available_count == 0:
            return True

        # 2. Overwhelming failure rate: > 70% of nodes in group are in circuit cooldown
        if total_count >= 3 and (available_count / total_count) < 0.30:
            return True

        # 3. Severe latency saturation on healthy top nodes
        healthy_latencies: list[float] = []
        for m in available_members:
            st = self.circuit_breaker.get_state(m.provider_id)
            if st.recent_call_times:
                sample_count = len(st.recent_call_times)
                actual_success = sum(1 for _, s in st.recent_call_times if s)
                raw_avail = actual_success / sample_count if sample_count >= 2 else 1.0
                # Ignore broken/dead nodes (< 20% avail) from latency health assessment
                if raw_avail >= 0.20:
                    healthy_latencies.append(st.ewma_latency_ms)

        if healthy_latencies:
            healthy_latencies.sort()
            k = max(1, min(3, len(healthy_latencies)))
            top_k_latency = sum(healthy_latencies[:k]) / k
            if top_k_latency >= latency_threshold_ms:
                return True

        return False

    async def order_candidates_with_fallback(
        self,
        members: list[PoolMember],
        strategy: str = "usage-based-routing-v2",
        require_vision: bool = False,
        primary_group: str | None = None,
        max_attempts_per_group: int = 3,
        enable_congestion_promotion: bool = True,
        congestion_latency_threshold_ms: float = 15000.0,
        strict_group: bool = False,
        request_interval_seconds: float | None = None,
    ) -> list[PoolMember]:
        """Order members through the fallback chain, taking up to max_attempts_per_group from each group.

        When enable_congestion_promotion is True and the primary group is evaluated as congested,
        standby groups are sequentially promoted as primary for 90% of traffic (e.g. promoting agnes
        first, and falling back to ds-official if agnes fails), retaining 10% canary probe traffic
        to auto-detect recovery.
        """
        if require_vision:
            members = [m for m in members if m.supports_vision]

        if not members:
            return []

        # Group members by their normalized group_names
        group_to_members: dict[str, list[PoolMember]] = {}
        for m in members:
            for g in (m.group_names or ("default",)):
                grp = self.resolve_target_group(g or "default")
                group_to_members.setdefault(grp, []).append(m)

        # Determine starting group
        first_group = primary_group or self.manual_active_group
        if not first_group:
            first_group = self.get_default_group(group_to_members)

        if strict_group:
            fallback_chain = [first_group]
        else:
            fallback_chain = self.build_fallback_chain(first_group, group_to_members)

        # Congestion Promotion: if primary group is congested, sequentially advance along user's fallback rules!
        is_suppressed = (time.monotonic() < self._congestion_suppressed_until)
        if (
            not strict_group
            and not self.manual_active_group
            and not is_suppressed
            and enable_congestion_promotion
            and len(fallback_chain) > 1
        ):
            promoted_idx = 0
            while promoted_idx < len(fallback_chain) - 1:
                curr_grp = fallback_chain[promoted_idx]
                curr_members = group_to_members.get(curr_grp, [])
                if await self.is_group_congested(curr_members, latency_threshold_ms=congestion_latency_threshold_ms):
                    promoted_idx += 1
                else:
                    break

            if promoted_idx > 0:
                async with self._lock:
                    self._congestion_request_counter += 1
                    is_canary = (self._congestion_request_counter % 10 == 0)

                if not is_canary:
                    # 90% of traffic directly routes downstream from the healthy promoted group
                    fallback_chain = fallback_chain[promoted_idx:]

        final_candidates: list[PoolMember] = []
        seen_pids: set[str] = set()

        for grp in fallback_chain:
            grp_members = group_to_members.get(grp, [])
            if not grp_members:
                continue
            ordered = await self.order_group_candidates(
                grp_members,
                strategy=strategy,
                require_vision=False,  # already filtered
                request_interval_seconds=request_interval_seconds,
            )

            # Cap attempts per group if max_attempts_per_group > 0, matching LiteLLM's num_retries
            capped_grp_members = ordered[:max_attempts_per_group] if max_attempts_per_group > 0 else ordered
            for m in capped_grp_members:
                if m.provider_id not in seen_pids:
                    # If member has multiple groups, specialize group_name to the active group slot
                    if len(m.group_names) > 1:
                        target_g = next(
                            (g for g in m.group_names if self.resolve_target_group(g) == grp),
                            grp,
                        )
                        m_dispatched = replace(m, group_name=target_g)
                    else:
                        m_dispatched = m
                    final_candidates.append(m_dispatched)
                    seen_pids.add(m.provider_id)

        return final_candidates

