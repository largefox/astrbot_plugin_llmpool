from __future__ import annotations

from typing import Any

_runtime_context: Any = None
_plugin_config: Any = None
_metrics_store: Any = None
_circuit_breaker: Any = None
_router: Any = None
_key_manager: Any = None
_inspector: Any = None


def bind_runtime(
    context: Any,
    config: Any,
    metrics_store: Any = None,
    circuit_breaker: Any = None,
    router: Any = None,
    key_manager: Any = None,
    inspector: Any = None,
) -> None:
    global _runtime_context, _plugin_config, _metrics_store, _circuit_breaker, _router, _key_manager, _inspector
    _runtime_context = context
    _plugin_config = config
    _metrics_store = metrics_store
    _circuit_breaker = circuit_breaker
    _router = router
    _key_manager = key_manager
    _inspector = inspector


def unbind_runtime(context: Any) -> None:
    global _runtime_context, _plugin_config, _metrics_store, _circuit_breaker, _router, _key_manager, _inspector
    if _runtime_context is context:
        _runtime_context = None
        _plugin_config = None
        _metrics_store = None
        _circuit_breaker = None
        _router = None
        _key_manager = None
        _inspector = None


def get_runtime() -> tuple[Any, Any]:
    if _runtime_context is None or _plugin_config is None:
        raise RuntimeError(
            "模型池运行时尚未绑定。请确认插件已启用；"
            "若刚刚更新或停用过插件，请重新启用或重启 AstrBot。"
        )
    return _runtime_context, _plugin_config


def get_components() -> tuple[Any, Any, Any, Any]:
    """Retrieve (metrics_store, circuit_breaker, router, key_manager)."""
    return _metrics_store, _circuit_breaker, _router, _key_manager


def get_inspector() -> Any:
    """Retrieve the prompt inspector instance if bound."""
    return _inspector
