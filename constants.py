from __future__ import annotations

from typing import Final

POOL_PROVIDER_TYPE: Final = "llm_pool"
POOL_PROVIDER_ID: Final = "llm-pool"
POOL_MODEL_NAME: Final = "llm-pool-router"
POOL_DISPLAY_NAME: Final = "LLM 智能模型池（路由入口）"

POOL_DEFAULT_CONFIG: Final[dict[str, object]] = {
    "id": POOL_PROVIDER_ID,
    "provider": POOL_PROVIDER_ID,
    "type": POOL_PROVIDER_TYPE,
    "provider_type": "chat_completion",
    "enable": True,
    "model": POOL_MODEL_NAME,
    "key": [],
    "hint": "插件管理的虚拟路由入口；无需填写 API Key 或 API Base URL。",
}

# Default per-pool overrideable settings.  When a dedicated pool config does
# not specify a field, the corresponding global value is used.
DEDICATED_POOL_DEFAULTS: Final[dict[str, object]] = {
    "strategy": "",                 # empty = inherit global strategy
    "max_attempts": 0,              # 0 = inherit global max_attempts
    "request_timeout_seconds": 0,   # 0 = inherit global
    "stream_timeout": 0,            # 0 = inherit global
    "strict_group": False,
}

IMAGE_CAPTION_REFERENCE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "image_caption_provider_id",
        "default_image_caption_provider_id",
    }
)
