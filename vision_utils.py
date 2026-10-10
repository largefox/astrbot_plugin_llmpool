"""Vision detection and error categorization helpers."""

from __future__ import annotations

from typing import Any


def request_contains_image(
    image_urls: list[str] | None = None,
    contexts: Any | None = None,
    extra_user_content_parts: Any | None = None,
) -> bool:
    """Check if the incoming request contains any image content across all input formats."""
    if image_urls and len(image_urls) > 0:
        return True

    if extra_user_content_parts:
        for part in extra_user_content_parts:
            part_type = getattr(part, "type", None)
            if not part_type and isinstance(part, dict):
                part_type = part.get("type")
            if part_type == "image_url":
                return True

    if contexts:
        for msg in contexts:
            content = msg.content if hasattr(msg, "content") else (msg.get("content") if isinstance(msg, dict) else None)
            if isinstance(content, list):
                for part in content:
                    ptype = getattr(part, "type", None)
                    if not ptype and isinstance(part, dict):
                        ptype = part.get("type")
                    if ptype == "image_url":
                        return True

    return False


def is_vision_unsupported_error(exc: Exception | str) -> bool:
    """Check if upstream specifically rejected the request due to lack of image/vision support."""
    msg = str(exc).lower()
    return any(
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
    )


def is_context_format_error(exc: Exception | str) -> bool:
    """Check if upstream rejected the request due to malformed message context (e.g. 400 MessageContent deserialization)."""
    msg = str(exc).lower()
    return any(
        k in msg
        for k in (
            "messagecontent",
            "invalid content",
            "failed to deserialize the json body",
            "json_parse_error",
            "目标接口不支持的字段",
            "invalid_request_error",
            "invalid_request",
            "unsupported content",
        )
    )


def sanitize_contexts(contexts: Any) -> Any:
    """Sanitize message contexts into clean plain-text messages to rescue 400 deserialization errors."""
    if not isinstance(contexts, list):
        return contexts

    sanitized = []
    for msg in contexts:
        if isinstance(msg, dict):
            role = str(msg.get("role") or "user")
            content = msg.get("content")
            if isinstance(content, str):
                sanitized.append({"role": role, "content": content})
            elif isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        if item.get("type") == "text":
                            t = item.get("text", "")
                            if t:
                                parts.append(str(t))
                        elif item.get("type") == "image_url":
                            parts.append("[图片]")
                        elif "text" in item:
                            parts.append(str(item["text"]))
                    elif hasattr(item, "text"):
                        parts.append(str(getattr(item, "text")))
                sanitized.append({"role": role, "content": "\n".join(parts).strip() or "[消息]"})
            else:
                sanitized.append({"role": role, "content": str(content) if content is not None else ""})
        elif hasattr(msg, "role") and hasattr(msg, "content"):
            role = str(getattr(msg, "role", "user"))
            content = getattr(msg, "content", "")
            if isinstance(content, str):
                sanitized.append({"role": role, "content": content})
            elif isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        if item.get("type") == "text":
                            t = item.get("text", "")
                            if t:
                                parts.append(str(t))
                        elif item.get("type") == "image_url":
                            parts.append("[图片]")
                        elif "text" in item:
                            parts.append(str(item["text"]))
                    elif hasattr(item, "text"):
                        parts.append(str(getattr(item, "text")))
                sanitized.append({"role": role, "content": "\n".join(parts).strip() or "[消息]"})
            else:
                sanitized.append({"role": role, "content": str(content) if content is not None else ""})
        else:
            sanitized.append(msg)
    return sanitized


def mask_model_name(node_id: str, model_name: str) -> str:
    """Mask model names according to hack rules: gemini -> hy3, gpt -> hy4 (without suffix)."""
    combined = f"{node_id} {model_name}".lower()
    if "gemini" in combined:
        return "hy3"
    if "gpt" in combined:
        return "hy4"
    name = model_name or node_id or "hy3"
    return name.split("/")[-1] if "/" in name else name


def mask_node_id(node_id: str, model_name: str) -> str:
    """Mask node id: strip everything before the last slash, and hack gemini -> hy3, gpt -> hy4."""
    combined = f"{node_id} {model_name}".lower()
    if "gemini" in combined:
        return "hy3"
    if "gpt" in combined:
        return "hy4"
    # Do not display content before the last slash (e.g. flxapi/deepseek-v4.1-flash -> deepseek-v4.1-flash)
    return node_id.split("/")[-1] if "/" in node_id else node_id
