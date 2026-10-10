from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

try:
    from .constants import (
        IMAGE_CAPTION_REFERENCE_KEYS,
        POOL_DEFAULT_CONFIG,
        POOL_MODEL_NAME,
        POOL_PROVIDER_ID,
        POOL_PROVIDER_TYPE,
    )
except ImportError:  # Standalone unit tests from the repository root.
    from constants import (
        IMAGE_CAPTION_REFERENCE_KEYS,
        POOL_DEFAULT_CONFIG,
        POOL_MODEL_NAME,
        POOL_PROVIDER_ID,
        POOL_PROVIDER_TYPE,
    )


@dataclass(frozen=True, slots=True)
class ConfigReconcileResult:
    changed: bool
    canonical_config: dict[str, Any] | None
    alias_ids: frozenset[str]
    duplicate_provider_ids: tuple[str, ...]
    removed_source_ids: tuple[str, ...]
    migrated_references: int
    conflict: str | None = None
    canonical_configs: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class RuntimeSyncResult:
    state: str
    terminated_provider_ids: tuple[str, ...] = ()
    conflict: str | None = None


def _clean_id(value: object) -> str:
    return str(value or "").strip()


def _is_pool_source_config(value: object) -> bool:
    return isinstance(value, dict) and _clean_id(value.get("type")) == POOL_PROVIDER_TYPE


def _is_pool_provider_config(
    value: object,
    pool_source_ids: set[str] | frozenset[str],
) -> bool:
    if not isinstance(value, dict):
        return False
    if _clean_id(value.get("type")) == POOL_PROVIDER_TYPE:
        return True
    source_id = _clean_id(value.get("provider_source_id"))
    return bool(
        source_id
        and (
            source_id in pool_source_ids
            or source_id == POOL_PROVIDER_ID
        )
    )


def _is_pool_reference(value: object, aliases: set[str] | frozenset[str]) -> bool:
    provider_id = _clean_id(value)
    if not provider_id:
        return False
    return (
        provider_id == POOL_PROVIDER_ID
        or provider_id.startswith(f"{POOL_PROVIDER_ID}/")
        or provider_id in aliases
    )


def build_canonical_provider_config() -> dict[str, Any]:
    return dict(POOL_DEFAULT_CONFIG)


def _cfg_get(plugin_config: Any, key: str, default: Any = None) -> Any:
    """Read a key from plugin_config that may be a dict or have .get()."""
    if plugin_config is None:
        return default
    if hasattr(plugin_config, "get"):
        return plugin_config.get(key, default)
    if isinstance(plugin_config, dict):
        return plugin_config.get(key, default)
    return default


def resolve_dedicated_pools(plugin_config: Any = None) -> list[dict[str, Any]]:
    """Return a list of dedicated pool configs.

    Each item has keys: id, group, model_name, strict_group, strategy,
    max_attempts, request_timeout_seconds, stream_timeout.
    """
    raw_list = _cfg_get(plugin_config, "dedicated_pools", [])
    pools: list[dict[str, Any]] = []

    if isinstance(raw_list, list):
        for item in raw_list:
            if not isinstance(item, dict):
                continue
            pid = str(item.get("id") or "").strip()
            if not pid:
                continue
            pools.append(
                {
                    "id": pid,
                    "group": str(item.get("group") or pid).strip(),
                    "model_name": str(item.get("model_name") or f"{pid}-router").strip(),
                    "strict_group": bool(item.get("strict_group", False)),
                    "strategy": str(item.get("strategy") or "").strip(),
                    "max_attempts": int(item.get("max_attempts") or 0),
                    "request_timeout_seconds": float(item.get("request_timeout_seconds") or 0),
                    "stream_timeout": float(item.get("stream_timeout") or 0),
                }
            )

    return pools


def build_canonical_dedicated_configs(plugin_config: Any = None) -> list[dict[str, Any]]:
    """Build canonical AstrBot provider configs for all dedicated pools."""
    pools = resolve_dedicated_pools(plugin_config)
    configs: list[dict[str, Any]] = []
    for p in pools:
        cfg = {
            "id": p["id"],
            "provider": p["id"],
            "type": POOL_PROVIDER_TYPE,
            "provider_type": "chat_completion",
            "enable": True,
            "model": p["model_name"],
            "key": [],
            "hint": f"插件管理的专用节点池路由入口 ({p['id']})；无需填写 API Key 或 API Base URL。",
        }
        configs.append(cfg)
    return configs


def migrate_caption_references(
    root: object,
    aliases: set[str] | frozenset[str],
) -> int:
    """Replace pool aliases only in image-caption provider reference fields."""
    changed = 0

    def visit(value: object) -> None:
        nonlocal changed
        if isinstance(value, dict):
            for key, item in list(value.items()):
                if (
                    key in IMAGE_CAPTION_REFERENCE_KEYS
                    and _is_pool_reference(item, aliases)
                    and item != POOL_PROVIDER_ID
                ):
                    value[key] = POOL_PROVIDER_ID
                    changed += 1
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(root)
    return changed


def reconcile_default_config(
    config: Any,
    plugin_config: Any = None,
) -> ConfigReconcileResult:
    """Keep canonical pool Providers and remove UI-created duplicates."""
    providers = config.get("provider", []) if hasattr(config, "get") else None
    if not isinstance(providers, list):
        return ConfigReconcileResult(
            changed=False,
            canonical_config=None,
            alias_ids=frozenset({POOL_PROVIDER_ID}),
            duplicate_provider_ids=(),
            removed_source_ids=(),
            migrated_references=0,
            conflict="AstrBot 配置字段 `provider` 不是列表，无法维护模型池入口。",
        )

    provider_sources = config.get("provider_sources", []) if hasattr(config, "get") else []
    sources_are_list = isinstance(provider_sources, list)
    sources = provider_sources if sources_are_list else []

    # Build canonical configs
    canonical_primary = build_canonical_provider_config()
    canonical_list: list[dict[str, Any]] = [canonical_primary]

    # Add all dedicated pools (supports multiple)
    canonical_dedicated = build_canonical_dedicated_configs(plugin_config)
    canonical_list.extend(canonical_dedicated)

    canonical_ids = {
        _clean_id(c.get("id"))
        for c in canonical_list
        if _clean_id(c.get("id"))
    }

    # Discover all plugin-owned source IDs before mutating anything. In 0.1.0,
    # Dashboard-created model cards referenced these IDs but omitted their type.
    pool_source_ids = {
        source_id
        for source_id in (
            _clean_id(source.get("id"))
            for source in sources
            if _is_pool_source_config(source)
        )
        if source_id
    }
    pool_source_ids.update(canonical_ids)

    for c in canonical_list:
        target_id = _clean_id(c.get("id"))
        foreign_canonical_source = next(
            (
                source
                for source in sources
                if isinstance(source, dict)
                and _clean_id(source.get("id")) == target_id
                and not _is_pool_source_config(source)
            ),
            None,
        )
        if foreign_canonical_source is not None:
            foreign_type = _clean_id(foreign_canonical_source.get("type")) or "<empty>"
            return ConfigReconcileResult(
                changed=False,
                canonical_config=None,
                alias_ids=frozenset(canonical_ids),
                duplicate_provider_ids=(),
                removed_source_ids=(),
                migrated_references=0,
                conflict=(
                    f"Provider Source ID `{target_id}` 已被其他类型 "
                    f"`{foreign_type}` 占用；插件没有覆盖该配置。"
                ),
            )

        foreign_canonical = next(
            (
                item
                for item in providers
                if isinstance(item, dict)
                and _clean_id(item.get("id")) == target_id
                and not _is_pool_provider_config(item, pool_source_ids)
            ),
            None,
        )
        if foreign_canonical is not None:
            foreign_type = _clean_id(foreign_canonical.get("type")) or "<empty>"
            return ConfigReconcileResult(
                changed=False,
                canonical_config=None,
                alias_ids=frozenset(canonical_ids),
                duplicate_provider_ids=(),
                removed_source_ids=(),
                migrated_references=0,
                conflict=(
                    f"Provider ID `{target_id}` 已被其他类型 `{foreign_type}` 占用；"
                    "插件没有覆盖该配置。"
                ),
            )

    pool_entries = [
        item
        for item in providers
        if _is_pool_provider_config(item, pool_source_ids)
    ]
    pool_provider_ids = {
        provider_id
        for provider_id in (
            _clean_id(item.get("id")) for item in pool_entries
        )
        if provider_id
    }

    alias_ids = {*canonical_ids, *pool_source_ids, *pool_provider_ids}
    duplicate_provider_ids = tuple(
        provider_id
        for provider_id in sorted(pool_provider_ids)
        if provider_id not in canonical_ids
    )
    removed_source_ids = tuple(
        sorted(
            {
                _clean_id(source.get("id"))
                for source in sources
                if _is_pool_source_config(source) and _clean_id(source.get("id"))
            }
        )
    )

    first_pool_index = next(
        (
            index
            for index, item in enumerate(providers)
            if _is_pool_provider_config(item, pool_source_ids)
        ),
        len(providers),
    )
    retained_providers = [
        item
        for item in providers
        if not _is_pool_provider_config(item, pool_source_ids)
    ]
    insertion_index = min(first_pool_index, len(retained_providers))
    normalized_providers = [
        *retained_providers[:insertion_index],
        *canonical_list,
        *retained_providers[insertion_index:],
    ]

    retained_sources = [
        source for source in sources if not _is_pool_source_config(source)
    ]

    changed = normalized_providers != providers
    if changed:
        providers[:] = normalized_providers

    if sources_are_list and retained_sources != sources:
        provider_sources[:] = retained_sources
        changed = True

    migrated = migrate_caption_references(config, alias_ids)
    changed = changed or migrated > 0

    return ConfigReconcileResult(
        changed=changed,
        canonical_config=canonical_primary,
        alias_ids=frozenset(alias_ids),
        duplicate_provider_ids=duplicate_provider_ids,
        removed_source_ids=removed_source_ids,
        migrated_references=migrated,
        canonical_configs=tuple(canonical_list),
    )


def sanitize_plugin_members(
    plugin_config: Any,
    aliases: set[str] | frozenset[str],
) -> int:
    """Remove accidental self-references from the configured member list."""
    members = plugin_config.get("members", []) if hasattr(plugin_config, "get") else None
    if not isinstance(members, list):
        return 0

    retained: list[Any] = []
    removed = 0
    for item in members:
        provider_id = _clean_id(item.get("provider_id")) if isinstance(item, dict) else ""
        if _is_pool_reference(provider_id, aliases):
            removed += 1
            continue
        retained.append(item)

    if removed:
        members[:] = retained
    return removed


def provider_manager_is_initialized(manager: Any) -> bool:
    """Capability probe for AstrBot 4.27.x startup vs. hot-install lifecycle."""
    if getattr(manager, "_mcp_init_task", None) is not None:
        return True
    if getattr(manager, "inst_map", None):
        return True
    return bool(getattr(manager, "provider_insts", None))


def _instance_config(instance: Any) -> dict[str, Any]:
    config = getattr(instance, "provider_config", None)
    return config if isinstance(config, dict) else {}


def _instance_is_owned_pool(instance: Any) -> bool:
    return _clean_id(_instance_config(instance).get("type")) == POOL_PROVIDER_TYPE


def _instance_matches_canonical(
    instance: Any,
    canonical_config: dict[str, Any] | None = None,
) -> bool:
    config = _instance_config(instance)
    if not _instance_is_owned_pool(instance):
        return False
    target_id = _clean_id((canonical_config or {}).get("id")) or POOL_PROVIDER_ID
    target_model = _clean_id((canonical_config or {}).get("model")) or POOL_MODEL_NAME
    if _clean_id(config.get("id")) != target_id:
        return False
    configured_model = _clean_id(config.get("model"))
    get_model = getattr(instance, "get_model", None)
    runtime_model = _clean_id(get_model()) if callable(get_model) else configured_model
    return configured_model == target_model and runtime_model == target_model


@asynccontextmanager
async def _optional_lock(lock: Any):
    if lock is None:
        yield
        return
    async with lock:
        yield


async def sync_pool_runtime(
    manager: Any,
    *,
    providers: list[Any],
    provider_sources: list[Any],
    canonical_config: dict[str, Any] | None = None,
    canonical_configs: tuple[dict[str, Any], ...] | list[dict[str, Any]] | None = None,
    duplicate_provider_ids: tuple[str, ...],
) -> RuntimeSyncResult:
    """Synchronize ProviderManager after hot install without double-loading at startup."""
    manager.providers_config = providers
    manager.provider_sources_config = provider_sources

    if not provider_manager_is_initialized(manager):
        return RuntimeSyncResult(state="staged")

    cfgs: list[dict[str, Any]] = (
        list(canonical_configs)
        if canonical_configs is not None
        else ([dict(canonical_config)] if canonical_config is not None else [])
    )

    terminated: list[str] = []
    lock = getattr(manager, "resource_lock", None)
    async with _optional_lock(lock):
        inst_map = getattr(manager, "inst_map", {})
        terminate_provider = getattr(manager, "terminate_provider", None)
        load_provider = getattr(manager, "load_provider", None)
        if not isinstance(inst_map, dict) or not callable(terminate_provider) or not callable(
            load_provider
        ):
            return RuntimeSyncResult(
                state="unsupported",
                conflict="当前 AstrBot ProviderManager 缺少热加载接口，请完整重启 AstrBot。",
            )

        for provider_id in duplicate_provider_ids:
            if provider_id in inst_map:
                await terminate_provider(provider_id)
                terminated.append(provider_id)

        if not cfgs:
            return RuntimeSyncResult(
                state="ready",
                terminated_provider_ids=tuple(terminated),
            )

        overall_state = "ready"
        for cfg in cfgs:
            target_id = _clean_id(cfg.get("id"))
            if not target_id:
                continue

            current = inst_map.get(target_id)
            if current is not None and not _instance_is_owned_pool(current):
                return RuntimeSyncResult(
                    state="conflict",
                    terminated_provider_ids=tuple(terminated),
                    conflict=(
                        f"运行中的 Provider ID `{target_id}` 不属于本插件；"
                        "未执行覆盖。"
                    ),
                )

            if current is not None and not _instance_matches_canonical(current, cfg):
                await terminate_provider(target_id)
                terminated.append(target_id)
                current = None
                if overall_state == "ready":
                    overall_state = "reloaded"
            elif current is None:
                if overall_state == "ready":
                    overall_state = "loaded"

            if current is None:
                await load_provider(dict(cfg))
                if target_id not in getattr(manager, "inst_map", {}):
                    return RuntimeSyncResult(
                        state="load_failed",
                        terminated_provider_ids=tuple(terminated),
                        conflict=(
                            f"AstrBot 没有把 `{target_id}` 放入运行时 Provider 表；"
                            "请查看紧邻的 Provider 加载日志。"
                        ),
                    )

        return RuntimeSyncResult(
            state=overall_state,
            terminated_provider_ids=tuple(terminated),
        )
