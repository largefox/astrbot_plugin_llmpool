"""AstrBot plugin lifecycle for the image-caption routing Provider."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from astrbot.api import AstrBotConfig, logger, star
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.web import error_response, json_response, request
from astrbot.core.message.components import Plain
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path

# Register the Provider type before ProviderManager loads configured instances.
from . import providers as _providers  # noqa: F401
from .circuit_breaker import CircuitBreaker
from .constants import (
    POOL_MODEL_NAME,
    POOL_PROVIDER_ID,
)
from .key_manager import KeyManager
from .lifecycle import (
    migrate_caption_references,
    reconcile_default_config,
    sanitize_plugin_members,
    sync_pool_runtime,
)
from .metrics_store import MetricsStore
from .routing import UsageBasedRouter, normalize_members
from .runtime import bind_runtime, unbind_runtime
from .vision_utils import mask_node_id

PLUGIN_NAME = "astrbot_plugin_llmpool"
FALLBACK_PLUGIN_NAME = "astrbot_plugin_llm_pool"


def _normalize_fallbacks_dict(raw: Any) -> dict[str, list[str]]:
    """Convert template_list or dict fallbacks into normalized dict[str, list[str]]."""
    if isinstance(raw, dict):
        result: dict[str, list[str]] = {}
        for k, v in raw.items():
            if isinstance(v, list):
                result[str(k).strip()] = [str(x).strip() for x in v if str(x).strip()]
            elif isinstance(v, str) and v.strip():
                result[str(k).strip()] = [v.strip()]
        return result
    if isinstance(raw, list):
        result = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            src = str(item.get("source_group") or "").strip()
            tgt = str(item.get("target_group") or "").strip()
            if src and tgt:
                result.setdefault(src, []).append(tgt)
        return result
    return {}


def _normalize_alias_dict(raw: Any) -> dict[str, str]:
    """Convert template_list or dict aliases into normalized dict[str, str]."""
    if isinstance(raw, dict):
        return {str(k).strip(): str(v).strip() for k, v in raw.items() if str(k).strip() and str(v).strip()}
    if isinstance(raw, list):
        result = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            alias = str(item.get("alias_name") or "").strip()
            target = str(item.get("target_group") or "").strip()
            if alias and target:
                result[alias] = target
        return result
    return {}


class LLMPoolPlugin(star.Star):
    """Expose one stable Provider ID for AstrBot's main and image-caption slots."""

    def __init__(
        self,
        context: star.Context,
        config: AstrBotConfig,
    ) -> None:
        super().__init__(context)
        self.config = config

        db_dir = Path(get_astrbot_plugin_data_path()) / PLUGIN_NAME
        self.metrics_store = MetricsStore(db_dir / "pool_metrics.db")

        cooldown_time = float(self.config.get("cooldown_seconds", 1800.0) or 1800.0)
        daily_reset_hour = int(self.config.get("daily_reset_hour", 2) or 2)
        ratelimit_cooldown = float(self.config.get("ratelimit_cooldown_seconds", 90.0) or 90.0)
        aggressive_allowed_fails = int(self.config.get("aggressive_allowed_fails", 3) or 3)
        self.circuit_breaker = CircuitBreaker(
            cooldown_time=cooldown_time,
            allowed_fails=int(self.config.get("allowed_fails", 1) or 1),
            aggressive_allowed_fails=aggressive_allowed_fails,
            allowed_fails_policy={
                "UpstreamDeadError": 0,
                "InsufficientQuotaError": 0,
                "TimeoutError": int(self.config.get("timeout_allowed_fails", 2) or 2),
                "RateLimitError": int(self.config.get("ratelimit_allowed_fails", 0) or 0),
                "BadRequestError": int(self.config.get("badrequest_allowed_fails", 2) or 2),
            },
            persistence_path=db_dir / "circuit_state.json",
            daily_reset_hour=daily_reset_hour,
            ratelimit_cooldown_seconds=ratelimit_cooldown,
        )

        model_group_alias = _normalize_alias_dict(self.config.get("model_group_alias"))
        fallbacks = _normalize_fallbacks_dict(self.config.get("fallbacks"))
        model_request_interval = float(self.config.get("model_request_interval_seconds", 3.0) or 0.0)
        self.router = UsageBasedRouter(
            self.circuit_breaker,
            model_group_alias=model_group_alias or None,
            fallbacks=fallbacks or None,
            model_request_interval_seconds=model_request_interval,
        )

        self.key_manager = KeyManager(quarantine_duration=3600.0)

        bind_runtime(
            context,
            config,
            metrics_store=self.metrics_store,
            circuit_breaker=self.circuit_breaker,
            router=self.router,
            key_manager=self.key_manager,
        )
        self._register_web_apis()

    def _register_web_apis(self) -> None:
        routes = [
            ("stats", self.api_get_stats, ["GET"], "Get high-level summary KPIs and per-node statistics"),
            ("hourly", self.api_get_hourly_profile, ["GET"], "Get 24-hour availability and latency profile"),
            ("logs", self.api_get_logs, ["GET"], "Get recent request logs with error details"),
            ("circuit/status", self.api_get_circuit_status, ["GET"], "Get real-time circuit breaker status for all nodes"),
            ("circuit/reset", self.api_reset_circuit, ["POST"], "Reset circuit breaker status for a node or all nodes"),
            ("routing/status", self.api_get_routing_status, ["GET"], "Get current active model group and promotion status"),
            ("routing/set_active_group", self.api_set_active_group, ["POST"], "Set or reset the current active model group"),
            ("stats/cost_analysis", self.api_get_cost_analysis, ["GET"], "Get depth cost and tier absorption analytics"),
            ("stats/error_attribution", self.api_get_error_attribution, ["GET"], "Get detailed error type distribution and node failure attribution"),
            ("stats/hedging", self.api_get_hedging_stats, ["GET"], "Get speculative hedging racing statistics and saved latency"),
            ("stats/heatmap", self.api_get_weekly_heatmap, ["GET"], "Get 7x24 weekly traffic and congestion heatmap matrix"),
            ("stats/key_fleet", self.api_get_key_fleet, ["GET"], "Get multi-key fleet health and quarantine cooldown status"),
            ("config/get", self.api_get_config, ["GET"], "Get current plugin config and available AstrBot providers"),
            ("config/save", self.api_save_config, ["POST"], "Save updated plugin configuration visually"),
            ("config/import_providers", self.api_import_providers, ["POST"], "One-click import AstrBot chat providers as pool members"),
            ("config/export", self.api_export_config, ["GET"], "Export full plugin config as downloadable JSON"),
            ("config/import_json", self.api_import_config, ["POST"], "Import plugin config from JSON payload"),
            ("config/clear_members", self.api_clear_members, ["POST"], "Clear all pool members"),
            ("config/reset_all", self.api_reset_all_config, ["POST"], "Reset all plugin config to schema defaults"),
        ]
        # Register for both metadata name ("astrbot_plugin_llmpool") and directory name ("astrbot_plugin_llm_pool")
        names_to_register = {PLUGIN_NAME, FALLBACK_PLUGIN_NAME}
        for name in names_to_register:
            for subpath, handler, methods, desc in routes:
                self.context.register_web_api(
                    f"/{name}/{subpath}",
                    handler,
                    methods,
                    desc,
                )

    async def api_get_routing_status(self) -> Any:
        members = normalize_members(self.config.get("members", []))
        threshold = float(self.config.get("congestion_latency_threshold", 18.0) or 18.0) * 1000.0
        enable_promotion = bool(self.config.get("enable_congestion_promotion", True))
        status = await self.router.get_routing_status(
            members,
            enable_congestion_promotion=enable_promotion,
            congestion_latency_threshold_ms=threshold,
        )
        return json_response(status)

    async def api_set_active_group(self) -> Any:
        payload = await request.json(default={})
        if not isinstance(payload, dict):
            payload = {}
        action = str(payload.get("action") or "set").lower()
        group_name = payload.get("group_name")
        if action == "reset" or not group_name:
            await self.router.reset_active_group()
            # 严格保留所有已熔断死节点的封锁状态，仅将未熔断健康节点的 QoS 延迟复位，重置模型组提权
            await self.circuit_breaker.soft_reset_healthy_qos_metrics()
            msg = "已重置模型组为自动模式并复位提权状态（已熔断节点保持封锁）"
        else:
            await self.router.set_manual_active_group(str(group_name).strip())
            msg = f"已将活跃组手动切换为：{group_name}"
        members = normalize_members(self.config.get("members", []))
        threshold = float(self.config.get("congestion_latency_threshold", 18.0) or 18.0) * 1000.0
        enable_promotion = bool(self.config.get("enable_congestion_promotion", True))
        status = await self.router.get_routing_status(
            members,
            enable_congestion_promotion=enable_promotion,
            congestion_latency_threshold_ms=threshold,
        )
        return json_response({"ok": True, "message": msg, "routing_status": status})

    async def api_get_cost_analysis(self) -> Any:
        window_seconds = request.query.get("window_seconds", 86400, type=int)
        members = normalize_members(self.config.get("members", []))
        threshold = float(self.config.get("congestion_latency_threshold", 18.0) or 18.0) * 1000.0
        enable_promotion = bool(self.config.get("enable_congestion_promotion", True))
        routing_status = await self.router.get_routing_status(
            members,
            enable_congestion_promotion=enable_promotion,
            congestion_latency_threshold_ms=threshold,
        )
        fallback_chain = routing_status.get("configured_chain") or routing_status.get("active_chain") or []
        data = await self.metrics_store.get_cost_analysis(
            window_seconds=float(window_seconds),
            fallback_chain=fallback_chain,
        )
        return json_response(data)

    async def api_get_error_attribution(self) -> Any:
        window_seconds = request.query.get("window_seconds", 86400, type=int)
        data = await self.metrics_store.get_error_attribution(window_seconds=float(window_seconds))
        return json_response(data)

    async def api_get_hedging_stats(self) -> Any:
        window_seconds = request.query.get("window_seconds", 86400, type=int)
        data = await self.metrics_store.get_hedging_stats(window_seconds=float(window_seconds))
        return json_response(data)

    async def api_get_weekly_heatmap(self) -> Any:
        days = request.query.get("days", 7, type=int)
        data = await self.metrics_store.get_weekly_heatmap(days=days)
        return json_response(data)

    async def api_get_key_fleet(self) -> Any:
        manager = getattr(self.context, "provider_manager", None)
        inst_map = getattr(manager, "inst_map", {}) if manager else {}
        members = normalize_members(self.config.get("members", []))
        configured_member_pids = {m.provider_id: m for m in members if m.provider_id}

        provider_keys_map = {}
        provider_group_map = {}
        if isinstance(inst_map, dict):
            for pid, m in configured_member_pids.items():
                prov = inst_map.get(pid)
                if not prov:
                    continue
                keys = getattr(prov, "get_keys", lambda: [])() or getattr(prov, "provider_config", {}).get("key", [])
                clean_keys = []
                if isinstance(keys, list):
                    clean_keys = [str(k).strip() for k in keys if str(k).strip()]
                elif isinstance(keys, str) and keys.strip():
                    clean_keys = [keys.strip()]
                if clean_keys:
                    provider_keys_map[pid] = clean_keys
                    provider_group_map[pid] = m.group_name or "default"

        data = await self.key_manager.get_fleet_summary(provider_keys_map)
        for node in data.get("nodes", []):
            node["group_name"] = provider_group_map.get(node["provider_id"], "default")
        return json_response(data)

    async def api_get_stats(self) -> Any:
        window_seconds = request.query.get("window_seconds", 86400, type=int)
        summary = await self.metrics_store.get_summary_stats(window_seconds=float(window_seconds))
        nodes = await self.metrics_store.get_node_stats(window_seconds=float(window_seconds))
        circuit_status = await self.circuit_breaker.get_all_status()
        members = normalize_members(self.config.get("members", []))
        threshold = float(self.config.get("congestion_latency_threshold", 18.0) or 18.0) * 1000.0
        enable_promotion = bool(self.config.get("enable_congestion_promotion", True))
        routing_status = await self.router.get_routing_status(
            members,
            enable_congestion_promotion=enable_promotion,
            congestion_latency_threshold_ms=threshold,
        )

        # Detect whether llm-pool is selected as chat model or fallback model
        pool_selection = {"is_chat_model": False, "is_fallback_model": False}
        try:
            acm = getattr(self.context, "astrbot_config_mgr", None)
            default_conf = getattr(acm, "default_conf", None)
            if default_conf is not None:
                agent_runner = default_conf.get("agent_runner", {})
                model_cfg = agent_runner.get("config", {}).get("model", {})
                provider_id = str(model_cfg.get("provider_id") or "").strip()
                fallback_ids = model_cfg.get("fallback_provider_ids") or []
                # Collect all pool IDs (primary + dedicated)
                pool_ids = {POOL_PROVIDER_ID}
                try:
                    from .lifecycle import resolve_dedicated_pools
                except ImportError:
                    from lifecycle import resolve_dedicated_pools
                for p in resolve_dedicated_pools(self.config):
                    pool_ids.add(p["id"])
                pool_selection["is_chat_model"] = provider_id in pool_ids
                if isinstance(fallback_ids, list):
                    pool_selection["is_fallback_model"] = any(
                        str(fid).strip() in pool_ids for fid in fallback_ids
                    )
        except Exception:
            pass

        return json_response(
            {
                "summary": summary,
                "nodes": nodes,
                "circuit_status": circuit_status,
                "routing_status": routing_status,
                "pool_selection": pool_selection,
            }
        )

    async def api_get_hourly_profile(self) -> Any:
        node_id = request.query.get("node_id")
        days = request.query.get("days", 7, type=int)
        profile = await self.metrics_store.get_hourly_profile(node_id=node_id, days=days)
        return json_response({"profile": profile})

    async def api_get_logs(self) -> Any:
        page = request.query.get("page", 1, type=int)
        page_size = request.query.get("page_size", 50, type=int)
        status_filter = request.query.get("status")
        node_filter = request.query.get("node_id")
        window_seconds = request.query.get("window_seconds", None, type=float)
        start_time = request.query.get("start_time", None, type=float)
        end_time = request.query.get("end_time", None, type=float)
        keyword = request.query.get("keyword")

        data = await self.metrics_store.get_recent_logs(
            page=page,
            page_size=page_size,
            status_filter=status_filter,
            node_filter=node_filter,
            window_seconds=window_seconds,
            start_time=start_time,
            end_time=end_time,
            keyword=keyword,
        )
        return json_response(data)

    async def api_get_circuit_status(self) -> Any:
        status = await self.circuit_breaker.get_all_status()
        return json_response({"status": status})

    async def api_reset_circuit(self) -> Any:
        payload = await request.json(default={})
        if not isinstance(payload, dict):
            payload = {}
        node_id = payload.get("node_id")
        clear_history = bool(payload.get("clear_history", False))
        await self.circuit_breaker.reset(node_id, reset_metrics=True)
        if hasattr(self.router, "reset_dispatch_history"):
            self.router.reset_dispatch_history(node_id)
        deleted_count = 0
        if clear_history:
            deleted_count = await self.metrics_store.clear_metrics(node_id)
        return json_response({
            "ok": True,
            "reset_node": node_id or "all",
            "clear_history": clear_history,
            "deleted_records": deleted_count,
        })

    async def api_get_config(self) -> Any:
        manager = getattr(self.context, "provider_manager", None)
        inst_map = getattr(manager, "inst_map", {}) if manager else {}
        available_providers = []
        seen_pids = set()

        # Exclude primary pool and all dedicated pool IDs
        excluded_ids = {POOL_PROVIDER_ID}
        try:
            from .lifecycle import resolve_dedicated_pools
        except ImportError:
            from lifecycle import resolve_dedicated_pools
        for p in resolve_dedicated_pools(self.config):
            excluded_ids.add(p["id"])
        excluded_ids.discard("")

        # 1. Collect from inst_map (live instances)
        if isinstance(inst_map, dict):
            for pid, prov in inst_map.items():
                pid_str = str(pid).strip()
                if not pid_str or pid_str in excluded_ids or pid_str in seen_pids:
                    continue
                available_providers.append(
                    {
                        "provider_id": pid_str,
                        "type": str(getattr(prov, "provider_config", {}).get("type") or ""),
                        "model": str(getattr(prov, "get_model", lambda: "")() or ""),
                    }
                )
                seen_pids.add(pid_str)

        # 2. Collect from AstrBot default_conf (configured providers that may not be in inst_map yet)
        try:
            acm = getattr(self.context, "astrbot_config_mgr", None)
            default_conf = getattr(acm, "default_conf", None)
            if default_conf is not None:
                for k in ("provider", "providers"):
                    p_list = default_conf.get(k, []) if hasattr(default_conf, "get") else []
                    if isinstance(p_list, list):
                        for item in p_list:
                            if isinstance(item, dict):
                                pid_str = str(item.get("id") or item.get("provider") or "").strip()
                                if pid_str and pid_str not in excluded_ids and pid_str not in seen_pids:
                                    available_providers.append(
                                        {
                                            "provider_id": pid_str,
                                            "type": str(item.get("type") or ""),
                                            "model": str(item.get("model") or ""),
                                        }
                                    )
                                    seen_pids.add(pid_str)
        except Exception:
            pass

        # 3. Ensure all currently configured pool members are in available_providers so they always display
        for m in self.config.get("members", []):
            if isinstance(m, dict):
                pid_str = str(m.get("provider_id") or "").strip()
                if pid_str and pid_str not in seen_pids:
                    available_providers.append(
                        {
                            "provider_id": pid_str,
                            "type": "configured",
                            "model": "",
                        }
                    )
                    seen_pids.add(pid_str)

        return json_response(
            {
                "config": dict(self.config),
                "available_providers": available_providers,
            }
        )

    async def api_import_providers(self) -> Any:
        """One-click import AstrBot chat providers as pool members.

        Reads all enabled chat_completion providers from the ProviderManager and config,
        excludes pool/dedicated IDs, and merges them into the existing members
        list (skipping duplicates by provider_id).
        """
        manager = getattr(self.context, "provider_manager", None)
        inst_map = getattr(manager, "inst_map", {}) if manager else {}

        # Build excluded IDs (primary pool + dedicated pools)
        excluded_ids = {POOL_PROVIDER_ID}
        try:
            from .lifecycle import resolve_dedicated_pools
        except ImportError:
            from lifecycle import resolve_dedicated_pools
        for p in resolve_dedicated_pools(self.config):
            excluded_ids.add(p["id"])
        excluded_ids.discard("")

        # Collect existing member provider_ids to skip duplicates
        existing_members = self.config.get("members", [])
        if not isinstance(existing_members, list):
            existing_members = []
        existing_pids = {
            str(m.get("provider_id") or "").strip()
            for m in existing_members
            if isinstance(m, dict)
        }

        # Non-chat provider types to ignore during import
        ignored_types = {"tts", "stt", "whisper", "dall-e", "embedding", "embeddings", "rerank"}

        # Discover importable providers from live inst_map
        imported = []
        if isinstance(inst_map, dict):
            for pid, prov in inst_map.items():
                pid_str = str(pid).strip()
                if not pid_str or pid_str in excluded_ids or pid_str in existing_pids:
                    continue
                ptype = str(getattr(prov, "provider_config", {}).get("type") or "").lower()
                if any(bad in ptype for bad in ignored_types):
                    continue
                model = str(getattr(prov, "get_model", lambda: "")() or "")
                group_name = model or pid_str
                imported.append({
                    "__template_key": "provider",
                    "provider_id": pid_str,
                    "group_name": group_name,
                    "enabled": True,
                })
                existing_pids.add(pid_str)

        # Also discover from AstrBot default_conf
        try:
            acm = getattr(self.context, "astrbot_config_mgr", None)
            default_conf = getattr(acm, "default_conf", None)
            if default_conf is not None:
                for k in ("provider", "providers"):
                    p_list = default_conf.get(k, []) if hasattr(default_conf, "get") else []
                    if isinstance(p_list, list):
                        for item in p_list:
                            if isinstance(item, dict):
                                pid_str = str(item.get("id") or item.get("provider") or "").strip()
                                if not pid_str or pid_str in excluded_ids or pid_str in existing_pids:
                                    continue
                                ptype = str(item.get("type") or "").lower()
                                if any(bad in ptype for bad in ignored_types):
                                    continue
                                model = str(item.get("model") or "")
                                group_name = model or pid_str
                                imported.append({
                                    "__template_key": "provider",
                                    "provider_id": pid_str,
                                    "group_name": group_name,
                                    "enabled": True,
                                })
                                existing_pids.add(pid_str)
        except Exception:
            pass

        if not imported:
            return json_response({
                "ok": False,
                "message": "没有可导入的新 Provider（已全部在成员列表中，或不存在 chat_completion 类型的 Provider）",
                "imported_count": 0,
            })

        # Merge into config
        new_members = list(existing_members) + imported
        self.config["members"] = new_members
        self._safe_save_plugin_config()

        return json_response({
            "ok": True,
            "message": f"成功导入 {len(imported)} 个 Provider：{', '.join(m['provider_id'] for m in imported)}",
            "imported_count": len(imported),
            "imported": imported,
        })

    async def api_export_config(self) -> Any:
        """Export full plugin config as a downloadable JSON response."""
        from flask import Response
        import json as _json
        config_dict = dict(self.config)
        # Remove internal keys
        config_dict.pop("__template_key", None)
        json_str = _json.dumps(config_dict, indent=2, ensure_ascii=False)
        return Response(
            json_str,
            mimetype="application/json",
            headers={"Content-Disposition": "attachment; filename=llm_pool_config.json"},
        )

    async def api_import_config(self) -> Any:
        """Import plugin config from a JSON payload."""
        payload = await request.json(default={})
        if not isinstance(payload, dict):
            return error_response("Payload must be a JSON object", status_code=400)

        # Validate: must have at least 'members' key
        if "members" not in payload and "strategy" not in payload:
            return error_response("JSON 中未找到 members 或 strategy 字段，无法识别为有效配置", status_code=400)

        # Merge imported keys into config (overwrite existing)
        for k, v in payload.items():
            if k.startswith("_"):
                continue
            self.config[k] = v

        # Ensure template_list entries have __template_key
        for m in self.config.get("members", []):
            if isinstance(m, dict) and not m.get("__template_key"):
                m["__template_key"] = "provider"

        for dp in self.config.get("dedicated_pools", []):
            if isinstance(dp, dict) and not dp.get("__template_key"):
                dp["__template_key"] = "pool"

        self._safe_save_plugin_config()

        # Sync runtime
        try:
            acm = getattr(self.context, "astrbot_config_mgr", None)
            default_conf = getattr(acm, "default_conf", None)
            manager = getattr(self.context, "provider_manager", None)
            if default_conf is not None and manager is not None:
                reconciliation = reconcile_default_config(default_conf, self.config)
                if not reconciliation.conflict and reconciliation.canonical_configs:
                    if reconciliation.changed:
                        default_conf.save_config()
                    providers = default_conf.get("provider", [])
                    provider_sources = default_conf.get("provider_sources", [])
                    await sync_pool_runtime(
                        manager,
                        providers=providers,
                        provider_sources=provider_sources,
                        canonical_configs=reconciliation.canonical_configs,
                        duplicate_provider_ids=reconciliation.duplicate_provider_ids,
                    )
        except Exception as sync_exc:
            logger.warning("llm pool: api_import_config runtime sync failed (%s)", sync_exc)

        return json_response({"ok": True, "message": "配置已导入并保存", "config": dict(self.config)})

    async def api_clear_members(self) -> Any:
        """Clear all pool members."""
        old_count = len(self.config.get("members", []))
        self.config["members"] = []
        self._safe_save_plugin_config()
        return json_response({"ok": True, "message": f"已清空 {old_count} 个成员", "cleared_count": old_count})

    async def api_reset_all_config(self) -> Any:
        """Reset all plugin config to schema defaults."""
        import json as _json
        from pathlib import Path as _Path

        schema_path = _Path(__file__).parent / "_conf_schema.json"
        try:
            schema = _json.loads(schema_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return error_response(f"无法读取配置模板：{exc}", status_code=500)

        # Build default config from schema
        def get_default(field_schema):
            ftype = field_schema.get("type", "string")
            if ftype in ("bool",):
                return field_schema.get("default", False)
            if ftype in ("int",):
                return field_schema.get("default", 0)
            if ftype in ("float",):
                return field_schema.get("default", 0.0)
            if ftype in ("list", "template_list"):
                return field_schema.get("default", [])
            if ftype in ("object",):
                return field_schema.get("default", {})
            return field_schema.get("default", "")

        for key, field_schema in schema.items():
            if key.startswith("_"):
                continue
            self.config[key] = get_default(field_schema)

        self._safe_save_plugin_config()

        # Sync runtime
        try:
            acm = getattr(self.context, "astrbot_config_mgr", None)
            default_conf = getattr(acm, "default_conf", None)
            manager = getattr(self.context, "provider_manager", None)
            if default_conf is not None and manager is not None:
                reconciliation = reconcile_default_config(default_conf, self.config)
                if not reconciliation.conflict and reconciliation.canonical_configs:
                    if reconciliation.changed:
                        default_conf.save_config()
                    providers = default_conf.get("provider", [])
                    provider_sources = default_conf.get("provider_sources", [])
                    await sync_pool_runtime(
                        manager,
                        providers=providers,
                        provider_sources=provider_sources,
                        canonical_configs=reconciliation.canonical_configs,
                        duplicate_provider_ids=reconciliation.duplicate_provider_ids,
                    )
        except Exception as sync_exc:
            logger.warning("llm pool: api_reset_all_config runtime sync failed (%s)", sync_exc)

        return json_response({"ok": True, "message": "所有配置已重置为默认值", "config": dict(self.config)})

    def _safe_save_plugin_config(self) -> None:
        """Safely persist plugin configuration with Windows atomic write retry/fallback."""
        try:
            self.config.save_config()
        except OSError as exc:
            # Handle Windows [WinError 5] Access Denied on atomic replace
            logger.warning(
                "llm pool: standard save_config failed (%s), attempting direct fallback write",
                exc,
            )
            config_path = getattr(self.config, "config_path", None)
            if config_path:
                import json
                try:
                    with open(config_path, "w", encoding="utf-8-sig") as f:
                        json.dump(dict(self.config), f, indent=2, ensure_ascii=False)
                except Exception as inner_exc:
                    logger.error("llm pool: fallback config write failed: %s", inner_exc)

    async def api_save_config(self) -> Any:
        payload = await request.json(default={})
        if not isinstance(payload, dict):
            return error_response("Payload must be a JSON object", status_code=400)

        # Update and save
        for k, v in payload.items():
            self.config[k] = v

        # Convert fallbacks from dict to template_list format if saved as dict, preserving schema validity
        raw_fb = self.config.get("fallbacks")
        if isinstance(raw_fb, dict):
            template_fallbacks = []
            for src, tgts in raw_fb.items():
                if isinstance(tgts, list):
                    for tgt in tgts:
                        template_fallbacks.append({
                            "__template_key": "fallback_rule",
                            "source_group": str(src).strip(),
                            "target_group": str(tgt).strip(),
                        })
                elif isinstance(tgts, str) and tgts.strip():
                    template_fallbacks.append({
                        "__template_key": "fallback_rule",
                        "source_group": str(src).strip(),
                        "target_group": str(tgts).strip(),
                    })
            self.config["fallbacks"] = template_fallbacks

        # Convert model_group_alias from dict to template_list format if saved as dict
        raw_alias = self.config.get("model_group_alias")
        if isinstance(raw_alias, dict):
            template_alias = []
            for al, tgt in raw_alias.items():
                template_alias.append({
                    "__template_key": "alias_rule",
                    "alias_name": str(al).strip(),
                    "target_group": str(tgt).strip(),
                })
            self.config["model_group_alias"] = template_alias

        # Ensure all members have __template_key="provider"
        for m in self.config.get("members", []):
            if isinstance(m, dict) and not m.get("__template_key"):
                m["__template_key"] = "provider"

        for dp in self.config.get("dedicated_pools", []):
            if isinstance(dp, dict) and not dp.get("__template_key"):
                dp["__template_key"] = "pool"

        self._safe_save_plugin_config()

        # Update circuit breaker and router policies dynamically
        if "cooldown_seconds" in payload:
            self.circuit_breaker.cooldown_time = float(payload["cooldown_seconds"])
        if "allowed_fails" in payload:
            self.circuit_breaker.allowed_fails = int(payload["allowed_fails"])
        if "aggressive_allowed_fails" in payload:
            self.circuit_breaker.aggressive_allowed_fails = int(payload["aggressive_allowed_fails"] or 3)
        if "fallbacks" in payload:
            self.router.fallbacks = _normalize_fallbacks_dict(payload["fallbacks"])
        if "model_group_alias" in payload:
            self.router.model_group_alias = _normalize_alias_dict(payload["model_group_alias"])
        if "model_request_interval_seconds" in payload:
            self.router.model_request_interval_seconds = float(payload["model_request_interval_seconds"] or 0.0)

        # Dynamically synchronize AstrBot Provider runtime if secondary pool or provider settings were updated
        try:
            acm = getattr(self.context, "astrbot_config_mgr", None)
            default_conf = getattr(acm, "default_conf", None)
            manager = getattr(self.context, "provider_manager", None)
            if default_conf is not None and manager is not None:
                reconciliation = reconcile_default_config(default_conf, self.config)
                if not reconciliation.conflict and reconciliation.canonical_configs:
                    if reconciliation.changed:
                        default_conf.save_config()
                    providers = default_conf.get("provider", [])
                    provider_sources = default_conf.get("provider_sources", [])
                    await sync_pool_runtime(
                        manager,
                        providers=providers,
                        provider_sources=provider_sources,
                        canonical_configs=reconciliation.canonical_configs,
                        duplicate_provider_ids=reconciliation.duplicate_provider_ids,
                    )
        except Exception as sync_exc:
            logger.warning("llm pool: api_save_config runtime sync failed (%s)", sync_exc)

        return json_response({"ok": True, "config": dict(self.config)})

    # ==================== User Chat Commands ====================

    async def _render_pool_status_result(self) -> str:
        """Helper to resolve and format pool_status output text."""
        summary = await self.metrics_store.get_summary_stats(window_seconds=86400.0)
        circuit_status = await self.circuit_breaker.get_all_status()

        total_nodes = len(circuit_status)
        open_nodes = [s for s in circuit_status.values() if s.get("status") == "OPEN"]
        half_open_nodes = [s for s in circuit_status.values() if s.get("status") == "HALF-OPEN"]
        healthy_nodes_count = total_nodes - len(open_nodes) - len(half_open_nodes)

        tot_req = summary.get("total_requests", 0)
        suc_req = summary.get("success_requests", 0)
        fail_req = summary.get("failed_requests", 0)
        avail_pct = summary.get("availability_pct", 100.0)
        avg_lat = summary.get("avg_latency_ms", 0.0)
        avg_ttft = summary.get("avg_ttft_ms", 0.0)

        lines = [
            "⚡ 【LLM 模型池简要报告】",
            f"• 24h 请求总量: {tot_req} 次 (成功: {suc_req} | 失败: {fail_req})",
            f"• 24h 服务可用率: {avail_pct}%",
            f"• 平均总耗时: {avg_lat} ms",
            f"• 平均首字(TTFT): {avg_ttft} ms",
            f"• 节点健康概况: 正常 {healthy_nodes_count} / 熔断 {len(open_nodes)} / 试探 {len(half_open_nodes)} (总数 {total_nodes})",
        ]

        blocked_nodes = [
            s for s in circuit_status.values()
            if s.get("status") in {"OPEN", "DEEP-SLEEP"} or s.get("is_open") or s.get("is_deep_sleep")
        ]
        if blocked_nodes:
            lines.append("\n⚠️ 当前处于熔断冷却的节点:")
            for on in blocked_nodes:
                rem = on.get("cooldown_remaining_seconds", 0)
                masked_id = mask_node_id(on["node_id"], "")
                status_label = "深度休眠" if on.get("status") == "DEEP-SLEEP" else f"冷却剩余 {rem}s"
                reason_str = on.get("trip_reason") or on.get("last_error_category") or "异常超限"
                err_detail = on.get("trip_error_msg") or on.get("last_error_msg") or ""
                err_short = f" | {err_detail[:60]}..." if err_detail else ""
                lines.append(f"  - {masked_id} ({status_label}): {reason_str}{err_short}")
        return "\n".join(lines)

    @filter.command("pool_status")
    async def pool_status_command(self, event: AstrMessageEvent):
        """查看当前 LLM 模型池运行情况与健康报告。"""
        text = await self._render_pool_status_result()
        yield event.plain_result(text)

    @filter.command("pool_reset")
    async def pool_reset_command(self, event: AstrMessageEvent):
        """重置所有 LLM 节点的熔断冷却状态。"""
        await self.circuit_breaker.reset()
        yield event.plain_result("⚡ 已成功重置所有模型的熔断冷却状态！全部节点已恢复为健康状态。")

    @filter.on_llm_request(priority=-99999)
    async def intercept_commands_before_llm(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
    ) -> None:
        """Intercept pool_status and pool_reset inside reply/quote messages before LLM pipeline calls LLM."""
        msg_str = str(event.message_str or "").strip()
        raw_msg = str(getattr(getattr(event, "message_obj", None), "raw_message", "") or "")
        combined = f"{msg_str} {raw_msg}"

        # Detect command triggers even when embedded after quote/reply markers
        import re
        match = re.search(r"(?:^|\s|[#//！!])(pool_status|pool_reset)\b", combined, re.IGNORECASE)
        if not match:
            return

        cmd = match.group(1).lower()
        if cmd == "pool_reset":
            await self.circuit_breaker.reset()
            text = "⚡ 已成功重置所有模型的熔断冷却状态！全部节点已恢复为健康状态。"
        else:
            text = await self._render_pool_status_result()

        # Reply directly to chat and stop event from reaching LLM
        await event.send(MessageChain([Plain(text)]))
        event.stop_event()

    async def initialize(self) -> None:
        acm = self.context.astrbot_config_mgr
        default_conf = acm.default_conf
        reconciliation = reconcile_default_config(default_conf, self.config)
        if reconciliation.conflict:
            logger.error("image caption pool: %s", reconciliation.conflict)
            return
        assert reconciliation.canonical_config is not None

        changed_profiles: list[str] = []
        if reconciliation.changed:
            changed_profiles.append("default")

        for conf_id, conf in acm.confs.items():
            if conf is default_conf:
                continue
            updates = migrate_caption_references(conf, reconciliation.alias_ids)
            if updates:
                conf.save_config()
                changed_profiles.append(str(conf_id))

        if reconciliation.changed:
            default_conf.save_config()

        # Ensure all template_list entries in plugin config have valid __template_key
        config_modified = False
        for m in self.config.get("members", []):
            if isinstance(m, dict) and not m.get("__template_key"):
                m["__template_key"] = "provider"
                config_modified = True

        for fb in self.config.get("fallbacks", []):
            if isinstance(fb, dict) and not fb.get("__template_key"):
                fb["__template_key"] = "fallback_rule"
                config_modified = True

        for al in self.config.get("model_group_alias", []):
            if isinstance(al, dict) and not al.get("__template_key"):
                al["__template_key"] = "alias_rule"
                config_modified = True

        for dp in self.config.get("dedicated_pools", []):
            if isinstance(dp, dict) and not dp.get("__template_key"):
                dp["__template_key"] = "pool"
                config_modified = True

        if config_modified:
            self._safe_save_plugin_config()

        removed_members = sanitize_plugin_members(
            self.config,
            reconciliation.alias_ids,
        )
        if removed_members:
            self.config.save_config()
            logger.warning(
                "image caption pool: removed %d accidental self-reference(s) "
                "from plugin members",
                removed_members,
            )

        providers = default_conf.get("provider", [])
        provider_sources = default_conf.get("provider_sources")
        if not isinstance(provider_sources, list):
            provider_sources = []
            default_conf["provider_sources"] = provider_sources
            default_conf.save_config()

        runtime = await sync_pool_runtime(
            self.context.provider_manager,
            providers=providers,
            provider_sources=provider_sources,
            canonical_configs=reconciliation.canonical_configs,
            duplicate_provider_ids=reconciliation.duplicate_provider_ids,
        )
        if runtime.conflict:
            logger.error(
                "image caption pool runtime state=%s: %s",
                runtime.state,
                runtime.conflict,
            )
            return

        members = normalize_members(self.config.get("members", []))
        try:
            from .lifecycle import resolve_dedicated_pools
        except ImportError:
            from lifecycle import resolve_dedicated_pools
        dedicated = resolve_dedicated_pools(self.config)
        dedicated_summary = ", ".join(f"{p['id']}({p['model_name']})" for p in dedicated) or "none"
        logger.info(
            "llm pool ready: primary=%s(%s) dedicated=[%s] runtime=%s "
            "members=%d strategy=%s removed_duplicates=%d migrated_profiles=%d",
            POOL_PROVIDER_ID,
            POOL_MODEL_NAME,
            dedicated_summary,
            runtime.state,
            len(members),
            str(self.config.get("strategy", "usage-based-routing-v2")),
            len(reconciliation.duplicate_provider_ids),
            len(changed_profiles),
        )

    async def terminate(self) -> None:
        unbind_runtime(self.context)

        manager = self.context.provider_manager
        inst_map = getattr(manager, "inst_map", {})
        terminate_provider = getattr(manager, "terminate_provider", None)
        ids_to_term = [POOL_PROVIDER_ID]
        try:
            from .lifecycle import resolve_dedicated_pools
        except ImportError:
            from lifecycle import resolve_dedicated_pools
        for p in resolve_dedicated_pools(self.config):
            ids_to_term.append(p["id"])
        for pid in ids_to_term:
            if isinstance(inst_map, dict) and pid in inst_map and callable(terminate_provider):
                try:
                    await terminate_provider(pid)
                except Exception:
                    logger.exception(
                        "llm pool: failed to terminate runtime Provider %s", pid
                    )


# Backwards compatibility alias
ImageCaptionPoolPlugin = LLMPoolPlugin
