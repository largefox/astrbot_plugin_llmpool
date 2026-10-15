"""Inspector module for capturing full LLM prompts and responses on-demand.

Zero overhead when no active inspection session is open:
- Heartbeat lease mechanism: Only when a frontend user has the Inspector page
  open and active (sending heartbeats within lease duration), detailed prompt
  contexts, system prompts, multimodal attachments, and full responses are recorded.
- In-memory bounded ring buffer: Retains at most maxlen records (default 100),
  without persisting massive text/image payloads to disk database.
"""

from __future__ import annotations

import collections
import logging
import time
from datetime import datetime
from typing import Any

logger = logging.getLogger("astrbot_plugin_llmpool.inspector")


class PromptInspector:
    """Bounded in-memory store for complete LLM request and response records."""

    def __init__(self, max_records: int = 100, lease_seconds: float = 60.0) -> None:
        self.max_records = max_records
        self.lease_seconds = lease_seconds
        self._last_heartbeat: float = 0.0
        self._paused: bool = False
        self._records: collections.deque[dict[str, Any]] = collections.deque(maxlen=max_records)
        self._counter: int = 0

    def heartbeat(self, paused: bool | None = None) -> None:
        """Called by frontend polling to keep the recording lease active."""
        prev_active = self.is_active()
        self._last_heartbeat = time.time()
        if paused is not None:
            self._paused = bool(paused)
        if not prev_active and self.is_active():
            logger.info("llm pool inspector lease activated (lease=%.1fs)", self.lease_seconds)

    def is_active(self) -> bool:
        """Return True if an inspector page is currently open and recording is not paused."""
        if self._paused:
            return False
        return (time.time() - self._last_heartbeat) < self.lease_seconds

    def get_status(self) -> dict[str, Any]:
        """Return current inspector operational state."""
        active = self.is_active()
        return {
            "active": active,
            "paused": self._paused,
            "total_recorded": len(self._records),
            "max_records": self.max_records,
            "seconds_since_heartbeat": round(max(0.0, time.time() - self._last_heartbeat), 1)
            if self._last_heartbeat > 0
            else None,
        }

    def clear(self) -> None:
        """Clear all in-memory inspection records."""
        self._records.clear()

    @staticmethod
    def _truncate_long_data(value: Any, max_len: int = 1000) -> Any:
        """Truncate excessive base64 blobs to avoid bloating browser memory while keeping inspection useful."""
        if isinstance(value, str):
            if value.startswith("data:image/") or value.startswith("base64://"):
                if len(value) > 200:
                    prefix = value[:40]
                    return f"{prefix}... [base64 data truncated, total {len(value)} chars]"
            elif len(value) > 30000:
                return value[:30000] + f"... [truncated {len(value) - 30000} chars]"
            return value
        if isinstance(value, list):
            return [PromptInspector._truncate_long_data(x, max_len) for x in value]
        if isinstance(value, dict):
            return {k: PromptInspector._truncate_long_data(v, max_len) for k, v in value.items()}
        return value

    @classmethod
    def sanitize_contexts(cls, contexts: Any) -> list[dict[str, Any]]:
        """Normalize messages / contexts from AstrBot into clean, JSON-serializable dictionaries."""
        if not contexts:
            return []
        result: list[dict[str, Any]] = []
        if isinstance(contexts, list):
            for item in contexts:
                if isinstance(item, dict):
                    result.append(cls._truncate_long_data(item))
                elif hasattr(item, "role"):
                    # AstrBot Message entity
                    content = getattr(item, "content", "")
                    # If content is a list of ContentPart
                    if isinstance(content, list):
                        parts = []
                        for p in content:
                            if hasattr(p, "type"):
                                p_dict = {"type": getattr(p, "type", "text")}
                                if hasattr(p, "text"):
                                    p_dict["text"] = getattr(p, "text", "")
                                if hasattr(p, "image_url"):
                                    p_dict["image_url"] = getattr(p, "image_url", "")
                                parts.append(p_dict)
                            elif isinstance(p, dict):
                                parts.append(p)
                            else:
                                parts.append(str(p))
                        content = parts
                    result.append({
                        "role": str(getattr(item, "role", "user")),
                        "content": cls._truncate_long_data(content),
                    })
                else:
                    result.append({"role": "unknown", "content": cls._truncate_long_data(str(item))})
        return result

    def record_interaction(
        self,
        *,
        model: str | None = None,
        group_name: str | None = None,
        hit_node: str | None = None,
        is_stream: bool = False,
        prompt: str | None = None,
        contexts: Any = None,
        system_prompt: str | None = None,
        image_urls: list[str] | None = None,
        response_text: str | None = None,
        status: str = "SUCCESS",
        total_latency_ms: float = 0.0,
        ttft_ms: float | None = None,
        error_msg: str | None = None,
        attempts: list[dict[str, Any]] | None = None,
    ) -> None:
        """Record an end-to-end LLM submission and response."""
        if not self.is_active():
            elapsed = round(time.time() - self._last_heartbeat, 1) if self._last_heartbeat > 0 else "never"
            logger.debug("llm pool inspector: interaction skipped (inactive, last_hb=%s)", elapsed)
            return

        self._counter += 1
        now = time.time()
        dt_str = datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")

        cleaned_contexts = self.sanitize_contexts(contexts)
        cleaned_images = [self._truncate_long_data(u) for u in (image_urls or [])]

        entry: dict[str, Any] = {
            "id": self._counter,
            "timestamp": now,
            "time_str": dt_str,
            "model": model or "llm-pool",
            "group_name": group_name or "default",
            "hit_node": hit_node or "unknown",
            "is_stream": is_stream,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "contexts": cleaned_contexts,
            "image_urls": cleaned_images,
            "response_text": response_text,
            "status": status,
            "total_latency_ms": round(total_latency_ms, 1),
            "ttft_ms": round(ttft_ms, 1) if ttft_ms is not None else None,
            "error_msg": error_msg,
            "attempts": attempts or [],
        }

        self._records.append(entry)
        logger.info(
            "llm pool inspector: captured #%d model=%s hit=%s status=%s prompt_len=%d",
            self._counter,
            entry["model"],
            entry["hit_node"],
            status,
            len(str(prompt or "")),
        )

    def get_records(self, limit: int = 50, since_id: int | None = None) -> list[dict[str, Any]]:
        """Retrieve recent inspection records."""
        items = list(self._records)
        if since_id is not None:
            items = [item for item in items if item["id"] > since_id]
        if limit and limit > 0:
            items = items[-limit:]
        return items
