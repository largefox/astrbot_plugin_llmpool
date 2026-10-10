"""Key-level management, rotation, and bad-key quarantine for Providers."""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Any


def hash_key(key: str) -> str:
    """Return a short prefix/hash safe for logging and tracking."""
    if not key:
        return "empty"
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
    prefix = key[:6] if len(key) >= 6 else key
    return f"{prefix}..{h}"


def is_auth_error(exc: Exception | str) -> bool:
    """Check if exception represents an invalid/expired key (401/403/invalid api key)."""
    msg = str(exc).lower()
    return any(
        k in msg
        for k in (
            "401",
            "unauthorized",
            "invalid_api_key",
            "invalid api key",
            "authentication",
            "incorrect api key",
            "api_key_invalid",
            "403",
            "forbidden",
            "permission denied",
        )
    )


class KeyManager:
    """Manages active keys, fair rotation, and bad-key quarantine per Provider."""

    def __init__(self, quarantine_duration: float = 3600.0) -> None:
        self.quarantine_duration = quarantine_duration
        self._provider_cursors: dict[str, int] = {}
        # provider_id -> {key: quarantine_until_monotonic}
        self._quarantined_keys: dict[str, dict[str, float]] = {}
        self._lock = asyncio.Lock()

    def get_valid_keys(self, provider_id: str, all_keys: list[str]) -> list[str]:
        """Return non-quarantined keys for a provider."""
        if not all_keys:
            return [""]
        now = time.monotonic()
        quarantined = self._quarantined_keys.get(provider_id, {})
        valid = [k for k in all_keys if quarantined.get(k, 0.0) <= now]
        return valid if valid else list(all_keys)

    async def pick_next_key(self, provider_id: str, all_keys: list[str]) -> str:
        """Round-robin pick a valid key for the provider."""
        clean_keys = [str(k).strip() for k in all_keys if str(k).strip()]
        if not clean_keys:
            return ""

        async with self._lock:
            valid_keys = self.get_valid_keys(provider_id, clean_keys)
            cursor = self._provider_cursors.get(provider_id, 0)
            chosen_key = valid_keys[cursor % len(valid_keys)]
            self._provider_cursors[provider_id] = (cursor + 1) % max(1, len(valid_keys))
            return chosen_key

    async def mark_key_failure(
        self,
        provider_id: str,
        key: str,
        exc: Exception | str,
        duration: float | None = None,
    ) -> bool:
        """Quarantine a key if it suffered an auth/permanent failure. Returns True if quarantined."""
        if not key:
            return False

        if is_auth_error(exc):
            q_time = duration if duration is not None else self.quarantine_duration
            now = time.monotonic()
            async with self._lock:
                if provider_id not in self._quarantined_keys:
                    self._quarantined_keys[provider_id] = {}
                self._quarantined_keys[provider_id][key] = now + q_time
            return True
        return False

    async def mark_key_success(self, provider_id: str, key: str) -> None:
        """Clear quarantine on success."""
        if not key:
            return
        async with self._lock:
            if provider_id in self._quarantined_keys and key in self._quarantined_keys[provider_id]:
                del self._quarantined_keys[provider_id][key]

    async def get_key_status(self, provider_id: str, all_keys: list[str]) -> list[dict[str, Any]]:
        now = time.monotonic()
        async with self._lock:
            quarantined = self._quarantined_keys.get(provider_id, {})
            results = []
            for k in all_keys:
                clean_k = str(k).strip()
                if not clean_k:
                    continue
                q_until = quarantined.get(clean_k, 0.0)
                remaining = max(0.0, q_until - now)
                results.append(
                    {
                        "key_hash": hash_key(clean_k),
                        "status": "QUARANTINED" if remaining > 0 else "HEALTHY",
                        "quarantine_remaining_seconds": round(remaining, 1),
                    }
                )
            return results

    async def get_fleet_summary(self, provider_keys_map: dict[str, list[str]]) -> dict[str, Any]:
        """Aggregate health and quarantine status across all API keys in the fleet."""
        now = time.monotonic()
        total_keys = 0
        healthy_keys = 0
        quarantined_keys = 0
        node_details = []

        async with self._lock:
            for pid, keys in provider_keys_map.items():
                clean_keys = [str(k).strip() for k in keys if str(k).strip()]
                if not clean_keys:
                    continue
                q_map = self._quarantined_keys.get(pid, {})
                node_q = 0
                node_h = 0
                key_items = []
                for k in clean_keys:
                    total_keys += 1
                    rem = max(0.0, q_map.get(k, 0.0) - now)
                    if rem > 0:
                        quarantined_keys += 1
                        node_q += 1
                        key_items.append({
                            "key_hash": hash_key(k),
                            "status": "QUARANTINED",
                            "quarantine_remaining_seconds": round(rem, 1),
                        })
                    else:
                        healthy_keys += 1
                        node_h += 1
                        key_items.append({
                            "key_hash": hash_key(k),
                            "status": "HEALTHY",
                            "quarantine_remaining_seconds": 0.0,
                        })

                node_details.append({
                    "provider_id": pid,
                    "total_keys": len(clean_keys),
                    "healthy_keys": node_h,
                    "quarantined_keys": node_q,
                    "keys": key_items,
                })

        return {
            "total_keys": total_keys,
            "healthy_keys": healthy_keys,
            "quarantined_keys": quarantined_keys,
            "health_pct": round(healthy_keys / total_keys * 100.0, 1) if total_keys > 0 else 100.0,
            "nodes": node_details,
        }
