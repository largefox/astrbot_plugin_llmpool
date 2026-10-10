"""AstrBot meta-Provider for group-context image caption routing."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncGenerator
from typing import Any, Literal

from astrbot import logger
from astrbot.core.agent.message import ContentPart, Message
from astrbot.core.agent.tool import ToolSet
from astrbot.core.provider.entities import (
    LLMResponse,
    ProviderType,
    ToolCallsResult,
)
from astrbot.core.provider.provider import Provider
from astrbot.core.provider.register import (
    provider_cls_map,
    provider_registry,
    register_provider_adapter,
)
from astrbot.core.utils.error_redaction import safe_error

try:
    from .constants import (
        POOL_DISPLAY_NAME,
        POOL_MODEL_NAME,
        POOL_PROVIDER_ID,
        POOL_PROVIDER_TYPE,
    )
    from .routing import PoolMember, normalize_members, rotate_members, weighted_random_order
    from .runtime import get_components, get_runtime
    from .vision_utils import (
        is_context_format_error,
        is_vision_unsupported_error,
        request_contains_image,
        sanitize_contexts,
    )
except ImportError:
    from constants import (
        POOL_DISPLAY_NAME,
        POOL_MODEL_NAME,
        POOL_PROVIDER_ID,
        POOL_PROVIDER_TYPE,
    )
    from routing import PoolMember, normalize_members, rotate_members, weighted_random_order
    from runtime import get_components, get_runtime
    from vision_utils import (
        is_context_format_error,
        is_vision_unsupported_error,
        request_contains_image,
        sanitize_contexts,
    )


class LLMPoolError(RuntimeError):
    """A user-actionable pool configuration or execution failure."""


# Backwards compatibility alias
ImageCaptionPoolError = LLMPoolError


def _resolve_dedicated_pools(plugin_config: Any) -> list[dict[str, Any]]:
    """Thin wrapper around lifecycle.resolve_dedicated_pools with import fallback."""
    try:
        from .lifecycle import resolve_dedicated_pools
    except ImportError:
        from lifecycle import resolve_dedicated_pools
    return resolve_dedicated_pools(plugin_config)


def _all_dedicated_pool_ids(plugin_config: Any) -> set[str]:
    """Return the set of all configured dedicated pool provider IDs."""
    return {p["id"] for p in _resolve_dedicated_pools(plugin_config) if p.get("id")}


def _register_owned_provider():
    """Register without a Dashboard source template.

    The Provider is created and maintained by the plugin itself. Omitting a
    source template prevents users from accidentally creating a second
    ``llm-pool/<model>`` card in AstrBot 4.27.x.
    """
    existing = provider_cls_map.get(POOL_PROVIDER_TYPE)
    if existing is not None:
        existing_cls = getattr(existing, "cls_type", None)
        owned = bool(getattr(existing_cls, "_llm_pool_owned", False)) or bool(
            getattr(existing_cls, "_image_caption_pool_owned", False)
        )
        if not owned:
            module = str(getattr(existing_cls, "__module__", ""))
            raise ValueError(
                f"Provider type {POOL_PROVIDER_TYPE!r} is already owned by "
                f"{module or 'another adapter'}"
            )
        provider_registry[:] = [
            item for item in provider_registry if item is not existing
        ]
        provider_cls_map.pop(POOL_PROVIDER_TYPE, None)

    return register_provider_adapter(
        POOL_PROVIDER_TYPE,
        "AstrBot intelligent LLM routing Provider pool",
        provider_type=ProviderType.CHAT_COMPLETION,
        default_config_tmpl=None,
        provider_display_name=POOL_DISPLAY_NAME,
    )


@_register_owned_provider()
class ProviderLLMPool(Provider):
    """Delegate requests to a configured pool of Providers with fallback & circuit breaking."""

    _llm_pool_owned = True
    _image_caption_pool_owned = True

    def __init__(self, provider_config: dict, provider_settings: dict) -> None:
        super().__init__(provider_config, provider_settings)
        self_id = str(self.provider_config.get("id") or "").strip()
        if self_id == POOL_PROVIDER_ID or not self_id:
            configured_model = POOL_MODEL_NAME
        else:
            configured_model = (
                str(self.provider_config.get("model") or "").strip()
                or f"{self_id}-router"
            )
        self.provider_config["model"] = configured_model
        self.set_model(configured_model)
        self._selection_lock = asyncio.Lock()
        self._round_robin_cursor = 0
        self._cooldown_until: dict[str, float] = {}
        # Per-pool override cache (set by _resolve_candidates, read by text_chat/stream)
        self._pool_request_timeout: float = 0.0
        self._pool_stream_timeout: float = 0.0

    def get_current_key(self) -> str:
        return ""

    def set_key(self, key: str) -> None:
        # This meta-Provider owns no upstream credentials.
        return None

    async def get_models(self) -> list[str]:
        models = [self.get_model() or POOL_MODEL_NAME]
        try:
            _, plugin_config = get_runtime()
        except Exception:
            plugin_config = None

        if plugin_config is not None:
            self_id = str(self.provider_config.get("id") or "").strip()
            if self_id == POOL_PROVIDER_ID:
                # Primary pool advertises all dedicated pool model/group names
                for p in _resolve_dedicated_pools(plugin_config):
                    for m in (p["model_name"], p["group"]):
                        if m and m not in models:
                            models.append(m)
            else:
                # Dedicated pool advertises its own group
                pool_cfg = self._get_pool_config(plugin_config)
                if pool_cfg:
                    grp = pool_cfg["group"]
                    if grp and grp not in models:
                        models.append(grp)
        return models

    def _get_pool_config(self, plugin_config: Any) -> dict[str, Any] | None:
        """Return the dedicated pool config matching this instance's ID.

        Returns ``None`` if this instance is the primary pool or no matching
        dedicated pool is configured.
        """
        self_id = str(self.provider_config.get("id") or "").strip()
        if not self_id or self_id == POOL_PROVIDER_ID:
            return None
        for p in _resolve_dedicated_pools(plugin_config):
            if p["id"] == self_id:
                return p
        # Legacy fallback: _force_secondary flag
        if getattr(self, "_force_secondary", False):
            pools = _resolve_dedicated_pools(plugin_config)
            if pools:
                return pools[0]
        return None

    def _is_secondary_pool(self, plugin_config: Any) -> bool:
        """Backward-compatible alias — True if this instance is a dedicated pool."""
        return self._get_pool_config(plugin_config) is not None

    async def test(self, timeout: float = 45.0) -> None:
        # Validate the pool wiring without consuming an upstream API call.
        candidates, plugin_config = await self._resolve_candidates()
        if not candidates:
            if self._is_secondary_pool(plugin_config):
                raise ImageCaptionPoolError("轻量判定模型池没有可用成员。")
            raise ImageCaptionPoolError("图片理解模型池没有可用成员。")

    async def _member_order(
        self,
        members: list[PoolMember],
        strategy: str,
    ) -> list[PoolMember]:
        now = time.monotonic()
        ready = [
            member
            for member in members
            if self._cooldown_until.get(member.provider_id, 0.0) <= now
        ]
        if not ready:
            return []

        selection_pool = ready

        if strategy == "ordered_failover":
            ordered = list(selection_pool)
        elif strategy == "weighted_random":
            ordered = weighted_random_order(selection_pool, rng=random)
        else:
            async with self._selection_lock:
                start = self._round_robin_cursor
                self._round_robin_cursor = (self._round_robin_cursor + 1) % max(
                    1, len(selection_pool)
                )
            ordered = rotate_members(selection_pool, start)

        return ordered

    def _configured_members(self, plugin_config: Any) -> list[PoolMember]:
        members = normalize_members(plugin_config.get("members", []))
        self_id = str(self.provider_config.get("id") or "").strip()
        dedicated_ids = _all_dedicated_pool_ids(plugin_config)
        excluded_ids = {self_id, POOL_PROVIDER_ID, *dedicated_ids} - {""}
        return [
            member
            for member in members
            if member.provider_id not in excluded_ids
            and not any(member.provider_id.startswith(f"{x}/") for x in excluded_ids)
        ]

    async def _resolve_candidates(
        self,
        require_vision: bool = False,
        requested_model: str | None = None,
    ) -> tuple[list[tuple[PoolMember, Provider]], Any]:
        context, plugin_config = get_runtime()
        components = get_components()
        router = components[2] if len(components) >= 3 else None
        members = self._configured_members(plugin_config)
        if not members:
            raise ImageCaptionPoolError(
                "模型池为空。请在插件配置的“模型池成员”中添加至少一个真实 Provider。"
            )

        strategy = str(plugin_config.get("strategy", "usage-based-routing-v2") or "").strip()
        try:
            num_retries_per_group = int(plugin_config.get("max_attempts", 3))
        except (TypeError, ValueError):
            num_retries_per_group = 3

        enable_promotion = bool(plugin_config.get("enable_congestion_promotion", True))
        try:
            congestion_latency_ms = float(plugin_config.get("congestion_latency_threshold", 18.0)) * 1000.0
        except (TypeError, ValueError):
            congestion_latency_ms = 18000.0

        is_sec = self._is_secondary_pool(plugin_config)
        pool_cfg = self._get_pool_config(plugin_config)

        # Resolve per-pool overrides (fall back to global config)
        sec_group = pool_cfg["group"] if pool_cfg else ""
        sec_strict = bool(pool_cfg["strict_group"]) if pool_cfg else False
        pool_strategy = (
            str(pool_cfg["strategy"] or "").strip() if pool_cfg else ""
        ) or strategy
        pool_max_attempts = (
            int(pool_cfg["max_attempts"] or 0) if pool_cfg else 0
        ) or num_retries_per_group
        pool_request_timeout = (
            float(pool_cfg["request_timeout_seconds"] or 0) if pool_cfg else 0
        )
        pool_stream_timeout = (
            float(pool_cfg["stream_timeout"] or 0) if pool_cfg else 0
        )
        # Cache for text_chat / text_chat_stream
        self._pool_request_timeout = pool_request_timeout
        self._pool_stream_timeout = pool_stream_timeout

        # Build a set of all dedicated pool model/group names for routing matching
        all_dedicated = _resolve_dedicated_pools(plugin_config)
        dedicated_model_map: dict[str, dict[str, Any]] = {}
        for dp in all_dedicated:
            for key in (dp["model_name"], dp["group"]):
                if key:
                    dedicated_model_map[key] = dp

        target_group: str | None = None
        strict_group: bool = False

        if is_sec and pool_cfg:
            target_group = sec_group
            strict_group = sec_strict
        elif requested_model:
            cleaned_model = str(requested_model).strip()
            matched_pool = dedicated_model_map.get(cleaned_model)
            if matched_pool:
                target_group = matched_pool["group"]
                strict_group = bool(matched_pool["strict_group"])
                pool_strategy = str(matched_pool.get("strategy") or "").strip() or strategy
                pool_max_attempts = int(matched_pool.get("max_attempts") or 0) or num_retries_per_group
                self._pool_request_timeout = float(matched_pool.get("request_timeout_seconds") or 0)
                self._pool_stream_timeout = float(matched_pool.get("stream_timeout") or 0)
            elif router is not None:
                resolved = router.resolve_target_group(cleaned_model)
                if resolved != cleaned_model:
                    target_group = resolved

        if is_sec or (target_group and target_group in dedicated_model_map):
            matching_members = [
                m
                for m in members
                if any(
                    (
                        router.resolve_target_group(g or "default")
                        if router
                        else (g or "default")
                    )
                    == target_group
                    for g in (m.group_names or ("default",))
                )
            ]
            if not matching_members:
                raise ImageCaptionPoolError(
                    f"专用节点池分组 '{target_group}' 中没有配置可用成员。"
                    "请在插件配置的“模型池成员”中添加属于该分组的真实 Provider。"
                )

        if router is not None:
            try:
                model_request_interval = float(
                    plugin_config.get("model_request_interval_seconds", 3.0) or 0.0
                )
            except (TypeError, ValueError):
                model_request_interval = 3.0

            ordered_members = await router.order_candidates_with_fallback(
                members,
                strategy=pool_strategy,
                require_vision=require_vision,
                primary_group=target_group,
                max_attempts_per_group=pool_max_attempts,
                enable_congestion_promotion=enable_promotion,
                congestion_latency_threshold_ms=congestion_latency_ms,
                strict_group=strict_group,
                request_interval_seconds=model_request_interval,
            )
        else:
            if target_group:
                members_to_order = [
                    m
                    for m in members
                    if any(
                        (g or "default") == target_group
                        for g in (m.group_names or ("default",))
                    )
                ] or members
            else:
                members_to_order = members
            ordered_members = await self._member_order(members_to_order, strategy)

        candidates: list[tuple[PoolMember, Provider]] = []
        missing: list[str] = []
        ignored_pool_members: list[str] = []
        for member in ordered_members:
            provider = context.get_provider_by_id(member.provider_id)
            if provider is None:
                missing.append(member.provider_id)
                continue
            if provider is self:
                ignored_pool_members.append(member.provider_id)
                continue
            if not isinstance(provider, Provider):
                missing.append(member.provider_id)
                continue
            provider_type = str(
                getattr(provider, "provider_config", {}).get("type") or ""
            ).strip()
            if provider_type == POOL_PROVIDER_TYPE:
                ignored_pool_members.append(member.provider_id)
                continue
            candidates.append((member, provider))

        if not candidates:
            details: list[str] = []
            if missing:
                details.append(f"未加载或类型不正确：{', '.join(missing)}")
            if ignored_pool_members:
                details.append(
                    "被忽略的模型池自引用：" + ", ".join(ignored_pool_members)
                )
            suffix = "；".join(details)
            raise ImageCaptionPoolError(
                "模型池没有可用的真实 Provider。"
                "请确认成员已启用且 Provider ID 正确。"
                + (f" {suffix}。" if suffix else "")
            )
        return candidates, plugin_config

    async def _invoke_single_candidate(
        self,
        *,
        member: PoolMember,
        provider: Any,
        prompt: str | None,
        session_id: str | None,
        image_urls: list[str] | None,
        audio_urls: list[str] | None,
        func_tool: ToolSet | None,
        contexts: list[Message] | list[dict] | None,
        system_prompt: str | None,
        tool_calls_result: ToolCallsResult | list[ToolCallsResult] | None,
        extra_user_content_parts: list[ContentPart] | None,
        tool_choice: Literal["auto", "required"],
        timeout_seconds: float,
        call_kwargs: dict[str, Any],
        plugin_config: Any,
        metrics_store: Any,
        circuit_breaker: Any,
        key_manager: Any,
        attempt_index: int,
        max_attempts: int,
        verbose: bool,
    ) -> LLMResponse:
        """Execute a single candidate node with active_requests tracking, key retry, and metrics."""
        provider_id = member.provider_id
        start_time = time.time()
        st = circuit_breaker.get_state(provider_id) if circuit_breaker else None
        if st:
            st.active_requests += 1

        components = get_components()
        router = components[2] if len(components) >= 3 else None
        if router and hasattr(router, "mark_dispatched"):
            router.mark_dispatched(member)

        # Prepare arguments (extra_body is managed by AstrBot Provider's custom_extra_body config)
        merged_kwargs = dict(call_kwargs)

        # Key management: get provider keys and pick valid keys
        raw_keys = getattr(provider, "get_keys", lambda: [])() or []
        all_keys = [str(k).strip() for k in raw_keys if str(k).strip()]
        valid_keys = key_manager.get_valid_keys(provider_id, all_keys) if key_manager else all_keys
        keys_to_try = valid_keys if valid_keys else [""]
        last_member_exc: Exception | None = None

        try:
            for key_idx, active_key in enumerate(keys_to_try, start=1):
                if active_key and hasattr(provider, "set_key"):
                    try:
                        provider.set_key(active_key)
                    except Exception:
                        pass

                try:
                    response = await asyncio.wait_for(
                        provider.text_chat(
                            prompt=prompt,
                            session_id=session_id,
                            image_urls=image_urls,
                            audio_urls=audio_urls,
                            func_tool=func_tool,
                            contexts=contexts,
                            system_prompt=system_prompt,
                            tool_calls_result=tool_calls_result,
                            model=None,
                            extra_user_content_parts=extra_user_content_parts,
                            tool_choice=tool_choice,
                            request_max_retries=1,
                            **merged_kwargs,
                        ),
                        timeout=timeout_seconds,
                    )
                    if not str(getattr(response, "completion_text", "") or "").strip():
                        raise LLMPoolError("上游模型返回了空内容。")

                    latency_ms = (time.time() - start_time) * 1000.0
                    if key_manager and active_key:
                        await key_manager.mark_key_success(provider_id, active_key)
                    if circuit_breaker:
                        await circuit_breaker.mark_success(provider_id, latency_ms=latency_ms)
                    if metrics_store:
                        await metrics_store.record_request(
                            group_name=member.group_name or "default",
                            node_id=provider_id,
                            model_name=member.model_name or str(getattr(provider, "get_model", lambda: "")() or ""),
                            is_stream=False,
                            total_latency_ms=latency_ms,
                            status="SUCCESS",
                        )
                    if verbose:
                        logger.info(
                            "llm pool success: member=%s attempt=%d/%d latency=%.1fms",
                            provider_id,
                            attempt_index,
                            max_attempts,
                            latency_ms,
                        )
                    return response

                except asyncio.CancelledError:
                    # Cancelled due to hedging competitor completing first; re-raise cleanly without error marking
                    raise
                except Exception as exc:
                    # Context sanitization rescue: if 400 format/MessageContent deserialization error,
                    # sanitize contexts to plain text and retry immediately once on current node
                    if is_context_format_error(exc) and contexts:
                        cleaned_ctx = sanitize_contexts(contexts)
                        if cleaned_ctx != contexts:
                            logger.warning(
                                "llm pool: detected context format 400 error on %s, attempting sanitized plaintext retry...",
                                provider_id,
                            )
                            try:
                                response = await asyncio.wait_for(
                                    provider.text_chat(
                                        prompt=prompt,
                                        session_id=session_id,
                                        image_urls=image_urls,
                                        audio_urls=audio_urls,
                                        func_tool=func_tool,
                                        contexts=cleaned_ctx,
                                        system_prompt=system_prompt,
                                        tool_calls_result=tool_calls_result,
                                        model=None,
                                        extra_user_content_parts=None,
                                        tool_choice=tool_choice,
                                        request_max_retries=1,
                                        **merged_kwargs,
                                    ),
                                    timeout=timeout_seconds,
                                )
                                if str(getattr(response, "completion_text", "") or "").strip():
                                    latency_ms = (time.time() - start_time) * 1000.0
                                    if key_manager and active_key:
                                        await key_manager.mark_key_success(provider_id, active_key)
                                    if circuit_breaker:
                                        await circuit_breaker.mark_success(provider_id, latency_ms=latency_ms)
                                    if metrics_store:
                                        await metrics_store.record_request(
                                            group_name=member.group_name or "default",
                                            node_id=provider_id,
                                            model_name=member.model_name or str(getattr(provider, "get_model", lambda: "")() or ""),
                                            is_stream=False,
                                            total_latency_ms=latency_ms,
                                            status="SUCCESS",
                                        )
                                    logger.info("llm pool: sanitized plaintext retry succeeded on %s!", provider_id)
                                    return response
                            except Exception as retry_exc:
                                exc = retry_exc

                    last_member_exc = exc
                    if key_manager and active_key:
                        quarantined = await key_manager.mark_key_failure(provider_id, active_key, exc)
                        if quarantined and key_idx < len(keys_to_try):
                            logger.warning(
                                "llm pool: key failed with auth error on %s, trying next key (%d/%d)",
                                provider_id,
                                key_idx,
                                len(keys_to_try),
                            )
                            continue
                    break

            # If loop finished without returning, record failure
            latency_ms = (time.time() - start_time) * 1000.0
            error_text = safe_error("", last_member_exc)

            if is_vision_unsupported_error(last_member_exc or ""):
                logger.warning(
                    "llm pool: member %s reported image unsupported. Auto-marking supports_vision=False",
                    provider_id,
                )
                object.__setattr__(member, "supports_vision", False)
                for m_cfg in plugin_config.get("members", []):
                    if isinstance(m_cfg, dict) and m_cfg.get("provider_id") == provider_id:
                        m_cfg["supports_vision"] = False
                        if hasattr(plugin_config, "save_config"):
                            try:
                                plugin_config.save_config()
                            except Exception:
                                pass
                        break

            cat_name = "OtherError"
            if circuit_breaker:
                effective_cd = member.get_effective_cooldown(circuit_breaker.cooldown_time)
                custom_cd = effective_cd if (member.aggressive_mode or member.cooldown_seconds > 0.0) else None
                _, cat, _ = await circuit_breaker.mark_failure(
                    provider_id,
                    last_member_exc or "UnknownError",
                    custom_cooldown=custom_cd,
                    latency_ms=latency_ms,
                    is_aggressive=member.aggressive_mode,
                )
                cat_name = cat.value
            if metrics_store:
                await metrics_store.record_request(
                    group_name=member.group_name or "default",
                    node_id=provider_id,
                    model_name=member.model_name or "",
                    is_stream=False,
                    total_latency_ms=latency_ms,
                    status="FAILED",
                    error_type=cat_name,
                    error_msg=error_text[:300],
                )

            logger.warning(
                "llm pool member failed: member=%s attempt=%d/%d error=%s",
                provider_id,
                attempt_index,
                max_attempts,
                error_text,
            )
            raise last_member_exc or LLMPoolError(error_text)
        finally:
            if st:
                st.active_requests = max(0, st.active_requests - 1)

    async def text_chat(
        self,
        prompt: str | None = None,
        session_id: str | None = None,
        image_urls: list[str] | None = None,
        audio_urls: list[str] | None = None,
        func_tool: ToolSet | None = None,
        contexts: list[Message] | list[dict] | None = None,
        system_prompt: str | None = None,
        tool_calls_result: ToolCallsResult | list[ToolCallsResult] | None = None,
        model: str | None = None,
        extra_user_content_parts: list[ContentPart] | None = None,
        tool_choice: Literal["auto", "required"] = "auto",
        request_max_retries: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        metrics_store, circuit_breaker, _, key_manager = get_components()
        has_images = request_contains_image(
            image_urls=image_urls,
            contexts=contexts,
            extra_user_content_parts=extra_user_content_parts,
        )
        candidates, plugin_config = await self._resolve_candidates(
            require_vision=has_images,
            requested_model=model,
        )

        failover = bool(plugin_config.get("failover", True))
        try:
            timeout_seconds = float(plugin_config.get("request_timeout_seconds", 22.0))
        except (TypeError, ValueError):
            timeout_seconds = 22.0
        # Apply per-pool override
        if self._pool_request_timeout and self._pool_request_timeout > 0:
            timeout_seconds = self._pool_request_timeout
        # Hard cap single attempt timeout at 30.0s to prevent 50s+ indefinite stalling
        timeout_seconds = min(max(timeout_seconds, 5.0), 30.0)

        try:
            global_budget_seconds = float(plugin_config.get("global_timeout_seconds", 35.0))
        except (TypeError, ValueError):
            global_budget_seconds = 35.0
        global_budget_seconds = min(max(global_budget_seconds, 5.0), 180.0)
        overall_start_time = time.time()

        enable_hedging = bool(plugin_config.get("enable_hedging", True))
        try:
            hedging_delay = float(plugin_config.get("hedging_delay_seconds", 14.0))
        except (TypeError, ValueError):
            hedging_delay = 14.0
        hedging_delay = min(max(hedging_delay, 0.1), 60.0)

        try:
            retry_after_seconds = float(plugin_config.get("retry_after_seconds", 0.0) or 0.0)
        except (TypeError, ValueError):
            retry_after_seconds = 0.0
        retry_after_seconds = min(max(retry_after_seconds, 0.0), 30.0)

        enable_weighted_failover = bool(plugin_config.get("enable_weighted_failover", False))

        # Candidates are already sliced per-group by max_attempts_per_group in fallback order.
        # If failover is disabled, only attempt the very first candidate.
        max_attempts = len(candidates) if failover else 1
        remaining_candidates = list(candidates[:max_attempts])

        verbose = bool(plugin_config.get("verbose_logging", False))
        failures: list[tuple[str, str]] = []
        attempt_count = 0

        while remaining_candidates:
            elapsed_total = time.time() - overall_start_time
            remaining_budget = global_budget_seconds - elapsed_total
            if remaining_budget <= 1.0:
                logger.warning(
                    "llm pool: global deadline budget exhausted (%.1fs elapsed of %.1fs limit)",
                    elapsed_total,
                    global_budget_seconds,
                )
                break

            # Fast-path rescue: If time budget is tight (<= 8.0s) and multiple distinct groups remain,
            # jump directly to standby tiers in sequential order (e.g. agnes first, then ds-official)
            distinct_remaining_groups = []
            for m, _ in remaining_candidates:
                grp = m.group_name or "default"
                if grp not in distinct_remaining_groups:
                    distinct_remaining_groups.append(grp)

            if remaining_budget <= 8.0 and len(distinct_remaining_groups) > 1:
                split_idx = len(distinct_remaining_groups) // 2 if len(distinct_remaining_groups) >= 4 else -1
                standby_groups = (
                    distinct_remaining_groups[split_idx:]
                    if isinstance(split_idx, int) and split_idx > 0
                    else distinct_remaining_groups[-1:]
                )

                standby_candidates = [c for c in remaining_candidates if (c[0].group_name or "default") in standby_groups]
                other_candidates = [c for c in remaining_candidates if (c[0].group_name or "default") not in standby_groups]
                if standby_candidates and other_candidates:
                    logger.info(
                        "llm pool: time budget tight (%.1fs remaining). Fast-jumping directly to standby tiers %s",
                        remaining_budget,
                        standby_groups,
                    )
                    remaining_candidates = standby_candidates + other_candidates

            # Weighted failover: if enabled and this is a retry, re-pick from same-group candidates by weight
            if enable_weighted_failover and attempt_count > 0 and remaining_candidates:
                current_grp = remaining_candidates[0][0].group_name or "default"
                same_group = [(i, c) for i, c in enumerate(remaining_candidates) if (c[0].group_name or "default") == current_grp]
                if len(same_group) > 1:
                    import random as _rng
                    weights = [max(1, c[0][0].weight) for _, c in same_group]
                    total_w = sum(weights)
                    needle = _rng.random() * total_w
                    upto = 0.0
                    chosen_list_idx = 0
                    for li, (orig_idx, _) in enumerate(same_group):
                        upto += weights[li]
                        if needle < upto:
                            chosen_list_idx = li
                            break
                    orig_idx, chosen = same_group[chosen_list_idx]
                    remaining_candidates.pop(orig_idx)
                    member, provider = chosen
                else:
                    member, provider = remaining_candidates.pop(0)
            else:
                member, provider = remaining_candidates.pop(0)

            # retry_after: wait before retrying after a failure (skip on first attempt)
            if attempt_count > 0 and retry_after_seconds > 0.0:
                remaining_budget -= retry_after_seconds
                if remaining_budget <= 1.0:
                    logger.warning(
                        "llm pool: retry_after delay (%.1fs) exhausted remaining budget",
                        retry_after_seconds,
                    )
                    break
                if verbose:
                    logger.info(
                        "llm pool: waiting %.1fs before retry (retry_after)",
                        retry_after_seconds,
                    )
                await asyncio.sleep(retry_after_seconds)

            attempt_count += 1
            current_group = member.group_name or "default"

            # Node-specific timeout override: use member's effective timeout if specified/aggressive
            effective_node_timeout = member.get_effective_timeout(timeout_seconds)
            attempt_timeout = min(effective_node_timeout, max(3.0, remaining_budget))

            # Check if there is a same-group sibling available for delayed hedging
            hedge_sibling = None
            if enable_hedging and failover and attempt_timeout > hedging_delay:
                for idx, (cand_m, cand_p) in enumerate(remaining_candidates):
                    if (cand_m.group_name or "default") == current_group:
                        hedge_sibling = remaining_candidates.pop(idx)
                        break

            if verbose:
                logger.info(
                    "llm pool request: group=%s member=%s attempt=%d/%d timeout=%.1fs (budget=%.1fs) images=%d",
                    current_group,
                    member.provider_id,
                    attempt_count,
                    max_attempts,
                    attempt_timeout,
                    remaining_budget,
                    len(image_urls or []),
                )

            attempt_start_time = time.time()
            task_primary = asyncio.create_task(
                self._invoke_single_candidate(
                    member=member,
                    provider=provider,
                    prompt=prompt,
                    session_id=session_id,
                    image_urls=image_urls,
                    audio_urls=audio_urls,
                    func_tool=func_tool,
                    contexts=contexts,
                    system_prompt=system_prompt,
                    tool_calls_result=tool_calls_result,
                    extra_user_content_parts=extra_user_content_parts,
                    tool_choice=tool_choice,
                    timeout_seconds=attempt_timeout,
                    call_kwargs=dict(kwargs),
                    plugin_config=plugin_config,
                    metrics_store=metrics_store,
                    circuit_breaker=circuit_breaker,
                    key_manager=key_manager,
                    attempt_index=attempt_count,
                    max_attempts=max_attempts,
                    verbose=verbose,
                )
            )

            if hedge_sibling is None:
                # No same-group sibling for hedging, run primary directly
                try:
                    return await task_primary
                except Exception as exc:
                    failures.append((member.provider_id, safe_error("", exc)))
                    continue

            # We have a same-group sibling! Wait up to hedging_delay for primary
            effective_hedge_delay = hedging_delay
            done, pending = await asyncio.wait({task_primary}, timeout=effective_hedge_delay)
            if task_primary in done:
                try:
                    return task_primary.result()
                except Exception as exc:
                    failures.append((member.provider_id, safe_error("", exc)))
                    # Primary failed fast (e.g. 429). Now run the sibling immediately!
                    sibling_m, sibling_p = hedge_sibling
                    attempt_count += 1
                    sibling_timeout = min(sibling_m.get_effective_timeout(timeout_seconds), max(3.0, remaining_budget))
                    try:
                        return await self._invoke_single_candidate(
                            member=sibling_m,
                            provider=sibling_p,
                            prompt=prompt,
                            session_id=session_id,
                            image_urls=image_urls,
                            audio_urls=audio_urls,
                            func_tool=func_tool,
                            contexts=contexts,
                            system_prompt=system_prompt,
                            tool_calls_result=tool_calls_result,
                            extra_user_content_parts=extra_user_content_parts,
                            tool_choice=tool_choice,
                            timeout_seconds=sibling_timeout,
                            call_kwargs=dict(kwargs),
                            plugin_config=plugin_config,
                            metrics_store=metrics_store,
                            circuit_breaker=circuit_breaker,
                            key_manager=key_manager,
                            attempt_index=attempt_count,
                            max_attempts=max_attempts,
                            verbose=verbose,
                        )
                    except Exception as sib_exc:
                        failures.append((sibling_m.provider_id, safe_error("", sib_exc)))
                        continue

            # Primary is still running after effective_hedge_delay seconds!
            # Launch the hedge sibling task to race with primary.
            sibling_m, sibling_p = hedge_sibling
            attempt_count += 1
            sibling_timeout = min(sibling_m.get_effective_timeout(timeout_seconds), max(3.0, remaining_budget))
            if verbose:
                logger.info(
                    "llm pool: primary %s slow (>%.1fs), launching speculative hedge request on sibling %s",
                    member.provider_id,
                    effective_hedge_delay,
                    sibling_m.provider_id,
                )

            task_hedge = asyncio.create_task(
                self._invoke_single_candidate(
                    member=sibling_m,
                    provider=sibling_p,
                    prompt=prompt,
                    session_id=session_id,
                    image_urls=image_urls,
                    audio_urls=audio_urls,
                    func_tool=func_tool,
                    contexts=contexts,
                    system_prompt=system_prompt,
                    tool_calls_result=tool_calls_result,
                    extra_user_content_parts=extra_user_content_parts,
                    tool_choice=tool_choice,
                    timeout_seconds=sibling_timeout,
                    call_kwargs=dict(kwargs),
                    plugin_config=plugin_config,
                    metrics_store=metrics_store,
                    circuit_breaker=circuit_breaker,
                    key_manager=key_manager,
                    attempt_index=attempt_count,
                    max_attempts=max_attempts,
                    verbose=verbose,
                )
            )

            racing_tasks = {
                task_primary: member.provider_id,
                task_hedge: sibling_m.provider_id,
            }

            winner_response = None
            winner_node_id = None
            while racing_tasks:
                done_set, pending_set = await asyncio.wait(
                    set(racing_tasks.keys()),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for finished_task in done_set:
                    node_id = racing_tasks.pop(finished_task)
                    try:
                        resp = finished_task.result()
                        if winner_response is None:
                            winner_response = resp
                            winner_node_id = node_id
                            # Cancel the remaining racing task
                            for pending_task in pending_set:
                                pending_task.cancel()
                    except Exception as task_exc:
                        failures.append((node_id, safe_error("", task_exc)))

                if winner_response is not None:
                    # Clean up pending cancellations
                    if pending_set:
                        await asyncio.gather(*pending_set, return_exceptions=True)
                    if metrics_store and hasattr(metrics_store, "record_hedge") and winner_node_id:
                        sibling_won = (winner_node_id == sibling_m.provider_id)
                        elapsed_s = time.time() - attempt_start_time
                        saved_ms = max(0.0, (attempt_timeout - elapsed_s) * 1000.0) if sibling_won else 0.0
                        try:
                            await metrics_store.record_hedge(
                                group_name=current_group,
                                primary_node=member.provider_id,
                                sibling_node=sibling_m.provider_id,
                                winner_node=winner_node_id,
                                sibling_won=sibling_won,
                                saved_ms=saved_ms,
                            )
                        except Exception:
                            pass
                    return winner_response

            # Both primary and hedge failed; loop continues to next group
            continue

        details = "; ".join(
            f"{provider_id}: {error}" for provider_id, error in failures
        )
        if not details:
            details = "没有候选 Provider 被执行"

        logger.error(
            "llm pool: all candidate providers failed (%d attempts): %s",
            attempt_count,
            details,
        )

        suppress_error = bool(plugin_config.get("suppress_error_raise", True))
        if suppress_error:
            # Silent fallback: return empty response so AstrBot suppresses chat error output
            return LLMResponse(role="assistant", completion_text="")

        raise ImageCaptionPoolError(
            f"模型池本次请求全部失败（尝试 {attempt_count} 个成员）：{details}"
        )

    async def text_chat_stream(
        self,
        prompt: str | None = None,
        session_id: str | None = None,
        image_urls: list[str] | None = None,
        audio_urls: list[str] | None = None,
        func_tool: ToolSet | None = None,
        contexts: list[Message] | list[dict] | None = None,
        system_prompt: str | None = None,
        tool_calls_result: ToolCallsResult | list[ToolCallsResult] | None = None,
        model: str | None = None,
        extra_user_content_parts: list[ContentPart] | None = None,
        tool_choice: Literal["auto", "required"] = "auto",
        request_max_retries: int | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[LLMResponse, None]:
        metrics_store, circuit_breaker, _, key_manager = get_components()
        has_images = request_contains_image(
            image_urls=image_urls,
            contexts=contexts,
            extra_user_content_parts=extra_user_content_parts,
        )
        candidates, plugin_config = await self._resolve_candidates(
            require_vision=has_images,
            requested_model=model,
        )

        failover = bool(plugin_config.get("failover", True))
        try:
            base_stream_timeout = float(plugin_config.get("stream_timeout", 15))
        except (TypeError, ValueError):
            base_stream_timeout = 15.0
        # Apply per-pool override
        if self._pool_stream_timeout and self._pool_stream_timeout > 0:
            base_stream_timeout = self._pool_stream_timeout
        base_stream_timeout = min(max(base_stream_timeout, 3.0), 300.0)

        # Chunk interval watchdog (stall detector)
        try:
            chunk_stall_timeout = float(plugin_config.get("chunk_stall_timeout", 8.0))
        except (TypeError, ValueError):
            chunk_stall_timeout = 8.0
        chunk_stall_timeout = min(max(chunk_stall_timeout, 2.0), 60.0)

        try:
            retry_after_seconds = float(plugin_config.get("retry_after_seconds", 0.0) or 0.0)
        except (TypeError, ValueError):
            retry_after_seconds = 0.0
        retry_after_seconds = min(max(retry_after_seconds, 0.0), 30.0)

        # Candidates are already sliced per-group by max_attempts_per_group in fallback order.
        # If failover is disabled, only attempt the very first candidate.
        max_attempts = len(candidates) if failover else 1

        verbose = bool(plugin_config.get("verbose_logging", False))
        failures: list[tuple[str, str]] = []

        for attempt_index, (member, provider) in enumerate(
            candidates[:max_attempts],
            start=1,
        ):
            # retry_after: wait before retrying after a failure (skip on first attempt)
            if attempt_index > 1 and retry_after_seconds > 0.0:
                if verbose:
                    logger.info(
                        "llm pool stream: waiting %.1fs before retry (retry_after)",
                        retry_after_seconds,
                    )
                await asyncio.sleep(retry_after_seconds)

            provider_id = member.provider_id
            start_time = time.time()
            first_token_time: float | None = None

            # Ladder TTFT timeout: First attempt gets base_stream_timeout (15s);
            # subsequent attempts get slight ladder increment (+3s per attempt), capped at 30s
            attempt_stream_timeout = min(base_stream_timeout + (attempt_index - 1) * 3.0, 30.0)
            attempt_stream_timeout = min(member.get_effective_timeout(attempt_stream_timeout), 30.0)

            if verbose:
                logger.info(
                    "llm pool stream request: group=%s member=%s attempt=%d/%d timeout=%.1fs",
                    member.group_name or "default",
                    provider_id,
                    attempt_index,
                    max_attempts,
                    attempt_stream_timeout,
                )

            st = circuit_breaker.get_state(provider_id) if circuit_breaker else None
            if st:
                st.active_requests += 1

            components = get_components()
            router = components[2] if len(components) >= 3 else None
            if router and hasattr(router, "mark_dispatched"):
                router.mark_dispatched(member)

            # Prepare arguments (extra_body is managed by AstrBot Provider's custom_extra_body config)
            call_kwargs = dict(kwargs)

            # Key management: get provider keys and filter valid keys
            raw_keys = getattr(provider, "get_keys", lambda: [])() or []
            all_keys = [str(k).strip() for k in raw_keys if str(k).strip()]
            valid_keys = key_manager.get_valid_keys(provider_id, all_keys) if key_manager else all_keys
            keys_to_try = valid_keys if valid_keys else [""]

            member_succeeded = False
            last_member_exc = None

            try:
                for key_idx, active_key in enumerate(keys_to_try, start=1):
                    if active_key and hasattr(provider, "set_key"):
                        try:
                            provider.set_key(active_key)
                        except Exception:
                            pass

                    gen = None
                    try:
                        gen = provider.text_chat_stream(
                            prompt=prompt,
                            session_id=session_id,
                            image_urls=image_urls,
                            audio_urls=audio_urls,
                            func_tool=func_tool,
                            contexts=contexts,
                            system_prompt=system_prompt,
                            tool_calls_result=tool_calls_result,
                            model=None,
                            extra_user_content_parts=extra_user_content_parts,
                            tool_choice=tool_choice,
                            # Force request_max_retries=1 to disable AstrBot's slow internal tenacity retries
                            request_max_retries=1,
                            **call_kwargs,
                        )

                        # Attempt to get the very first chunk under ladder TTFT timeout
                        first_chunk = await asyncio.wait_for(
                            gen.__anext__(),
                            timeout=attempt_stream_timeout,
                        )
                        first_token_time = time.time()
                        ttft_ms = (first_token_time - start_time) * 1000.0

                        # First token arrived! Yield it and lock onto this provider.
                        yield first_chunk

                        # Continue consuming remaining chunks with watchdog for mid-stream stalls
                        while True:
                            try:
                                next_chunk = await asyncio.wait_for(
                                    gen.__anext__(),
                                    timeout=chunk_stall_timeout,
                                )
                                yield next_chunk
                            except StopAsyncIteration:
                                break

                        total_latency_ms = (time.time() - start_time) * 1000.0
                        if key_manager and active_key:
                            await key_manager.mark_key_success(provider_id, active_key)
                        if circuit_breaker:
                            await circuit_breaker.mark_success(
                                provider_id,
                                latency_ms=total_latency_ms,
                                ttft_ms=ttft_ms,
                            )
                        if metrics_store:
                            await metrics_store.record_request(
                                group_name=member.group_name or "default",
                                node_id=provider_id,
                                model_name=member.model_name or str(getattr(provider, "get_model", lambda: "")() or ""),
                                is_stream=True,
                                ttft_ms=ttft_ms,
                                total_latency_ms=total_latency_ms,
                                status="SUCCESS",
                            )
                        member_succeeded = True
                        return

                    except StopAsyncIteration:
                        total_latency_ms = (time.time() - start_time) * 1000.0
                        if first_token_time is not None:
                            member_succeeded = True
                            return
                        error_text = "Upstream stream ended with no content"
                        last_member_exc = LLMPoolError(error_text)
                        failures.append((provider_id, error_text))
                    except Exception as exc:
                        if first_token_time is not None:
                            # Stall error or disconnect occurred AFTER first token was already sent
                            logger.error(
                                "llm pool stream error/stall mid-generation on member=%s: %s",
                                provider_id,
                                safe_error("", exc),
                            )
                            raise

                        last_member_exc = exc
                        # Check if key failed with 401/auth; quarantine and retry next key in provider!
                        if key_manager and active_key:
                            quarantined = await key_manager.mark_key_failure(provider_id, active_key, exc)
                            if quarantined and key_idx < len(keys_to_try):
                                logger.warning(
                                    "llm pool stream: key auth error on %s, trying next key (%d/%d)",
                                    provider_id,
                                    key_idx,
                                    len(keys_to_try),
                                )
                                continue
                        break
                    finally:
                        if gen is not None and hasattr(gen, "aclose"):
                            try:
                                await gen.aclose()
                            except Exception:
                                pass

                if not member_succeeded:
                    total_latency_ms = (time.time() - start_time) * 1000.0
                    error_text = safe_error("", last_member_exc)
                    failures.append((provider_id, error_text))

                    # Auto-learn: If upstream model specifically does not support image input,
                    # dynamically disable supports_vision for this node so future image requests bypass it directly!
                    if is_vision_unsupported_error(last_member_exc or ""):
                        logger.warning(
                            "llm pool stream: member %s reported image unsupported. Auto-marking supports_vision=False",
                            provider_id,
                        )
                        object.__setattr__(member, "supports_vision", False)
                        for m_cfg in plugin_config.get("members", []):
                            if isinstance(m_cfg, dict) and m_cfg.get("provider_id") == provider_id:
                                m_cfg["supports_vision"] = False
                                if hasattr(plugin_config, "save_config"):
                                    try:
                                        plugin_config.save_config()
                                    except Exception:
                                        pass
                                break

                    cat_name = "OtherError"
                    if circuit_breaker:
                        effective_cd = member.get_effective_cooldown(circuit_breaker.cooldown_time)
                        custom_cd = effective_cd if (member.aggressive_mode or member.cooldown_seconds > 0.0) else None
                        _, cat, _ = await circuit_breaker.mark_failure(
                            provider_id,
                            last_member_exc or "UnknownError",
                            custom_cooldown=custom_cd,
                            latency_ms=total_latency_ms,
                            is_aggressive=member.aggressive_mode,
                        )
                        cat_name = cat.value
                    if metrics_store:
                        await metrics_store.record_request(
                            group_name=member.group_name or "default",
                            node_id=provider_id,
                            model_name=member.model_name or "",
                            is_stream=True,
                            total_latency_ms=total_latency_ms,
                            status="FAILED",
                            error_type=cat_name,
                            error_msg=error_text[:300],
                        )

                    logger.warning(
                        "llm pool member stream failed before first token: member=%s attempt=%d/%d error=%s",
                        provider_id,
                        attempt_index,
                        max_attempts,
                        error_text,
                    )
                    continue
            finally:
                if st:
                    st.active_requests = max(0, st.active_requests - 1)

        details = "; ".join(
            f"{provider_id}: {error}" for provider_id, error in failures
        )
        logger.error(
            "llm pool stream: all candidate providers failed (%d attempts): %s",
            max_attempts,
            details,
        )

        suppress_error = bool(plugin_config.get("suppress_error_raise", True))
        if suppress_error:
            # Silent fallback: yield empty response so AstrBot stream finishes silently
            yield LLMResponse(role="assistant", completion_text="")
            return

        raise LLMPoolError(
            f"模型池流式请求全部失败（尝试 {max_attempts} 个成员）：{details}"
        )


# Backwards compatibility alias
ProviderImageCaptionPool = ProviderLLMPool
