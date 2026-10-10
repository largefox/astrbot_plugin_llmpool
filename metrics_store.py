"""SQLite-based persistent metrics storage and time-series analytics for LLM Pool."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
import sqlite3
import time
from typing import Any


class MetricsStore:
    """Stores request logs and provides time-window, availability, and latency analytics."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS request_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    hour_of_day INTEGER NOT NULL,
                    group_name TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    is_stream BOOLEAN NOT NULL,
                    ttft_ms REAL,
                    total_latency_ms REAL NOT NULL,
                    status TEXT NOT NULL,
                    http_code INTEGER,
                    error_type TEXT,
                    error_msg TEXT
                );
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_req_time
                ON request_metrics(timestamp);
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_req_node_hour
                ON request_metrics(node_id, hour_of_day);
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_req_group
                ON request_metrics(group_name);
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS hedge_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    group_name TEXT NOT NULL,
                    primary_node TEXT NOT NULL,
                    sibling_node TEXT NOT NULL,
                    winner_node TEXT NOT NULL,
                    sibling_won BOOLEAN NOT NULL,
                    saved_ms REAL NOT NULL
                );
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_hedge_time
                ON hedge_metrics(timestamp);
                """
            )
            conn.commit()
        finally:
            conn.close()

    async def record_request(
        self,
        *,
        group_name: str,
        node_id: str,
        model_name: str,
        is_stream: bool,
        total_latency_ms: float,
        status: str,
        ttft_ms: float | None = None,
        http_code: int | None = None,
        error_type: str | None = None,
        error_msg: str | None = None,
    ) -> None:
        """Asynchronously record one request metric into the database."""
        now = time.time()
        hour_of_day = datetime.now().hour

        def _insert() -> None:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO request_metrics (
                        timestamp, hour_of_day, group_name, node_id, model_name,
                        is_stream, ttft_ms, total_latency_ms, status, http_code,
                        error_type, error_msg
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        now,
                        hour_of_day,
                        group_name,
                        node_id,
                        model_name,
                        1 if is_stream else 0,
                        ttft_ms,
                        total_latency_ms,
                        status,
                        http_code,
                        error_type,
                        error_msg,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

        async with self._lock:
            await asyncio.to_thread(_insert)

    async def get_summary_stats(self, window_seconds: float = 86400.0) -> dict[str, Any]:
        """Get high-level summary KPIs for the last window_seconds (default 24h)."""
        cutoff = time.time() - window_seconds

        def _query() -> dict[str, Any]:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT
                        COUNT(*) as total_requests,
                        SUM(CASE WHEN status = 'SUCCESS' THEN 1 ELSE 0 END) as success_requests,
                        SUM(CASE WHEN status != 'SUCCESS' THEN 1 ELSE 0 END) as failed_requests,
                        AVG(total_latency_ms) as avg_latency_ms,
                        AVG(CASE WHEN is_stream = 1 AND ttft_ms IS NOT NULL THEN ttft_ms ELSE NULL END) as avg_ttft_ms
                    FROM request_metrics
                    WHERE timestamp >= ?
                    """,
                    (cutoff,),
                )
                row = cursor.fetchone()
                total = int(row["total_requests"] or 0)
                success = int(row["success_requests"] or 0)
                failed = int(row["failed_requests"] or 0)
                avail_rate = (success / total * 100.0) if total > 0 else 100.0
                return {
                    "total_requests": total,
                    "success_requests": success,
                    "failed_requests": failed,
                    "availability_pct": round(avail_rate, 2),
                    "avg_latency_ms": round(float(row["avg_latency_ms"] or 0.0), 1),
                    "avg_ttft_ms": round(float(row["avg_ttft_ms"] or 0.0), 1),
                }
            finally:
                conn.close()

        async with self._lock:
            return await asyncio.to_thread(_query)

    async def get_node_stats(self, window_seconds: float = 86400.0) -> list[dict[str, Any]]:
        """Get per-node statistics (success rate, latency, request count) in the last window."""
        cutoff = time.time() - window_seconds

        def _query() -> list[dict[str, Any]]:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT
                        node_id,
                        group_name,
                        model_name,
                        COUNT(*) as total,
                        SUM(CASE WHEN status = 'SUCCESS' THEN 1 ELSE 0 END) as success,
                        SUM(CASE WHEN status != 'SUCCESS' THEN 1 ELSE 0 END) as failed,
                        AVG(total_latency_ms) as avg_latency_ms,
                        AVG(CASE WHEN is_stream = 1 AND ttft_ms IS NOT NULL THEN ttft_ms ELSE NULL END) as avg_ttft_ms
                    FROM request_metrics
                    WHERE timestamp >= ?
                    GROUP BY node_id
                    ORDER BY total DESC
                    """,
                    (cutoff,),
                )
                results = []
                for row in cursor.fetchall():
                    total = int(row["total"] or 0)
                    success = int(row["success"] or 0)
                    avail = (success / total * 100.0) if total > 0 else 100.0
                    results.append(
                        {
                            "node_id": row["node_id"],
                            "group_name": row["group_name"],
                            "model_name": row["model_name"],
                            "total": total,
                            "success": success,
                            "failed": int(row["failed"] or 0),
                            "availability_pct": round(avail, 2),
                            "avg_latency_ms": round(float(row["avg_latency_ms"] or 0.0), 1),
                            "avg_ttft_ms": round(float(row["avg_ttft_ms"] or 0.0), 1),
                        }
                    )
                return results
            finally:
                conn.close()

        async with self._lock:
            return await asyncio.to_thread(_query)

    async def get_hourly_profile(
        self,
        node_id: str | None = None,
        days: int = 7,
    ) -> list[dict[str, Any]]:
        """Calculate availability and latency per hour-of-day (0-23) across recent days."""
        cutoff = time.time() - (days * 86400.0)

        def _query() -> list[dict[str, Any]]:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                if node_id:
                    cursor.execute(
                        """
                        SELECT
                            hour_of_day,
                            COUNT(*) as total,
                            SUM(CASE WHEN status = 'SUCCESS' THEN 1 ELSE 0 END) as success,
                            AVG(total_latency_ms) as avg_latency_ms,
                            AVG(CASE WHEN is_stream = 1 AND ttft_ms IS NOT NULL THEN ttft_ms ELSE NULL END) as avg_ttft_ms
                        FROM request_metrics
                        WHERE timestamp >= ? AND node_id = ?
                        GROUP BY hour_of_day
                        ORDER BY hour_of_day ASC
                        """,
                        (cutoff, node_id),
                    )
                else:
                    cursor.execute(
                        """
                        SELECT
                            hour_of_day,
                            COUNT(*) as total,
                            SUM(CASE WHEN status = 'SUCCESS' THEN 1 ELSE 0 END) as success,
                            AVG(total_latency_ms) as avg_latency_ms,
                            AVG(CASE WHEN is_stream = 1 AND ttft_ms IS NOT NULL THEN ttft_ms ELSE NULL END) as avg_ttft_ms
                        FROM request_metrics
                        WHERE timestamp >= ?
                        GROUP BY hour_of_day
                        ORDER BY hour_of_day ASC
                        """,
                        (cutoff,),
                    )
                hour_map = {row["hour_of_day"]: row for row in cursor.fetchall()}
                full_hours = []
                for h in range(24):
                    if h in hour_map:
                        r = hour_map[h]
                        tot = int(r["total"] or 0)
                        suc = int(r["success"] or 0)
                        full_hours.append(
                            {
                                "hour": h,
                                "total": tot,
                                "success": suc,
                                "availability_pct": round(suc / tot * 100.0, 2) if tot > 0 else 100.0,
                                "avg_latency_ms": round(float(r["avg_latency_ms"] or 0.0), 1),
                                "avg_ttft_ms": round(float(r["avg_ttft_ms"] or 0.0), 1),
                            }
                        )
                    else:
                        full_hours.append(
                            {
                                "hour": h,
                                "total": 0,
                                "success": 0,
                                "availability_pct": 100.0,
                                "avg_latency_ms": 0.0,
                                "avg_ttft_ms": 0.0,
                            }
                        )
                return full_hours
            finally:
                conn.close()

        async with self._lock:
            return await asyncio.to_thread(_query)

    async def get_recent_logs(
        self,
        page: int = 1,
        page_size: int = 50,
        status_filter: str | None = None,
        node_filter: str | None = None,
        window_seconds: float | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        keyword: str | None = None,
    ) -> dict[str, Any]:
        """Retrieve paginated request logs with multi-condition filtering."""
        page = max(1, page)
        page_size = min(max(page_size, 10), 200)
        offset = (page - 1) * page_size

        def _query() -> dict[str, Any]:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                where_clauses: list[str] = ["1=1"]
                params: list[Any] = []

                if start_time is not None and start_time > 0:
                    where_clauses.append("timestamp >= ?")
                    params.append(start_time)
                elif window_seconds is not None and window_seconds > 0:
                    cutoff = time.time() - window_seconds
                    where_clauses.append("timestamp >= ?")
                    params.append(cutoff)

                if end_time is not None and end_time > 0:
                    where_clauses.append("timestamp <= ?")
                    params.append(end_time)

                if status_filter:
                    where_clauses.append("status = ?")
                    params.append(status_filter)

                if node_filter:
                    where_clauses.append("node_id = ?")
                    params.append(node_filter)

                if keyword:
                    like_kw = f"%{keyword.strip()}%"
                    where_clauses.append("(node_id LIKE ? OR group_name LIKE ? OR model_name LIKE ? OR error_msg LIKE ?)")
                    params.extend([like_kw, like_kw, like_kw, like_kw])

                where_sql = " AND ".join(where_clauses)

                # Total count
                cursor.execute(f"SELECT COUNT(*) as total_count FROM request_metrics WHERE {where_sql}", tuple(params))
                total_count = int(cursor.fetchone()["total_count"] or 0)

                # Data rows
                query_sql = f"SELECT * FROM request_metrics WHERE {where_sql} ORDER BY id DESC LIMIT ? OFFSET ?"
                query_params = list(params)
                query_params.extend([page_size, offset])
                cursor.execute(query_sql, tuple(query_params))

                logs = []
                for row in cursor.fetchall():
                    logs.append(
                        {
                            "id": row["id"],
                            "timestamp": row["timestamp"],
                            "group_name": row["group_name"],
                            "node_id": row["node_id"],
                            "model_name": row["model_name"],
                            "is_stream": bool(row["is_stream"]),
                            "ttft_ms": round(float(row["ttft_ms"]), 1) if row["ttft_ms"] is not None else None,
                            "total_latency_ms": round(float(row["total_latency_ms"] or 0.0), 1),
                            "status": row["status"],
                            "http_code": row["http_code"],
                            "error_type": row["error_type"],
                            "error_msg": row["error_msg"],
                        }
                    )
                return {
                    "total": total_count,
                    "page": page,
                    "page_size": page_size,
                    "total_pages": max(1, (total_count + page_size - 1) // page_size),
                    "logs": logs,
                }
            finally:
                conn.close()

        async with self._lock:
            return await asyncio.to_thread(_query)

    async def get_latest_records(self, limit: int = 5) -> list[dict[str, Any]]:
        """Return the most recent N successful/failed records, newest first."""
        def _query() -> list[dict[str, Any]]:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM request_metrics ORDER BY id DESC LIMIT ?",
                    (limit,),
                )
                results = []
                for row in cursor.fetchall():
                    results.append(
                        {
                            "id": row["id"],
                            "timestamp": row["timestamp"],
                            "time_diff": 0.0,
                            "group_name": row["group_name"],
                            "node_id": row["node_id"],
                            "model_name": row["model_name"],
                            "is_stream": bool(row["is_stream"]),
                            "ttft_ms": round(float(row["ttft_ms"]), 1) if row["ttft_ms"] is not None else None,
                            "total_latency_ms": round(float(row["total_latency_ms"] or 0.0), 1),
                            "status": row["status"],
                            "error_type": row["error_type"],
                            "error_msg": row["error_msg"],
                        }
                    )
                return results
            finally:
                conn.close()

        async with self._lock:
            return await asyncio.to_thread(_query)

    async def get_logs_near_timestamp(
        self,
        target_timestamp: float,
        window_seconds: float = 15.0,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Query request records closest to a specific timestamp within +/- window_seconds."""
        start_t = target_timestamp - window_seconds
        end_t = target_timestamp + window_seconds

        def _query() -> list[dict[str, Any]]:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT *, ABS(timestamp - ?) as time_diff
                    FROM request_metrics
                    WHERE timestamp >= ? AND timestamp <= ?
                    ORDER BY time_diff ASC
                    LIMIT ?
                    """,
                    (target_timestamp, start_t, end_t, limit),
                )
                results = []
                for row in cursor.fetchall():
                    results.append(
                        {
                            "id": row["id"],
                            "timestamp": row["timestamp"],
                            "time_diff": round(float(row["time_diff"]), 2),
                            "group_name": row["group_name"],
                            "node_id": row["node_id"],
                            "model_name": row["model_name"],
                            "is_stream": bool(row["is_stream"]),
                            "ttft_ms": round(float(row["ttft_ms"]), 1) if row["ttft_ms"] is not None else None,
                            "total_latency_ms": round(float(row["total_latency_ms"] or 0.0), 1),
                            "status": row["status"],
                            "http_code": row["http_code"],
                            "error_type": row["error_type"],
                            "error_msg": row["error_msg"],
                        }
                    )
                return results
            finally:
                conn.close()

        async with self._lock:
            return await asyncio.to_thread(_query)

    async def cleanup_old_metrics(self, retention_days: int = 14) -> int:
        """Prune records older than retention_days."""
        cutoff = time.time() - (retention_days * 86400.0)

        def _cleanup() -> int:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM request_metrics WHERE timestamp < ?",
                    (cutoff,),
                )
                deleted = cursor.rowcount
                conn.commit()
                return deleted
            finally:
                conn.close()

        async with self._lock:
            return await asyncio.to_thread(_cleanup)

    async def clear_metrics(self, node_id: str | None = None) -> int:
        """Clear historical database records for a specific node or all nodes."""
        def _clear() -> int:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                if node_id:
                    cursor.execute("DELETE FROM request_metrics WHERE node_id = ?", (node_id,))
                else:
                    cursor.execute("DELETE FROM request_metrics")
                deleted = cursor.rowcount
                conn.commit()
                return deleted
            finally:
                conn.close()

        async with self._lock:
            return await asyncio.to_thread(_clear)

    async def get_cost_analysis(
        self,
        window_seconds: float = 86400.0,
        fallback_chain: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """Compute deep cost reduction, tier absorption funnel, and waste analysis dynamically."""
        cutoff = time.time() - window_seconds

        def _query() -> dict[str, Any]:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()

                # 1. Group / Tier aggregation
                cursor.execute(
                    """
                    SELECT
                        group_name,
                        COUNT(*) as total,
                        SUM(CASE WHEN status = 'SUCCESS' THEN 1 ELSE 0 END) as success,
                        SUM(CASE WHEN status != 'SUCCESS' THEN 1 ELSE 0 END) as failed,
                        AVG(total_latency_ms) as avg_latency_ms
                    FROM request_metrics
                    WHERE timestamp >= ?
                    GROUP BY group_name
                    ORDER BY total DESC
                    """,
                    (cutoff,),
                )
                clean_chain = [str(x).strip() for x in (fallback_chain or []) if str(x).strip()]

                def _norm_gn(raw_gn: str) -> str:
                    raw = str(raw_gn or "").strip()
                    if not raw:
                        return "default"
                    import re
                    parts = [p.strip() for p in re.split(r"[,;，；/|]+", raw) if p.strip()]
                    if len(parts) <= 1:
                        return parts[0] if parts else "default"
                    for p in parts:
                        if p in clean_chain:
                            return p
                    for p in parts:
                        if not any(k in p.lower() for k in ("light", "dedicated")):
                            return p
                    return parts[0]

                raw_groups_data: dict[str, dict[str, Any]] = {}
                total_all_reqs = 0
                for row in cursor.fetchall():
                    tot = int(row["total"] or 0)
                    succ = int(row["success"] or 0)
                    fail = int(row["failed"] or 0)
                    total_all_reqs += tot
                    raw_gn = str(row["group_name"] or "")
                    norm_gn = _norm_gn(raw_gn)
                    if norm_gn not in raw_groups_data:
                        raw_groups_data[norm_gn] = {
                            "group_name": norm_gn,
                            "total": 0,
                            "success": 0,
                            "failed": 0,
                            "total_latency_ms": 0.0,
                        }
                    raw_groups_data[norm_gn]["total"] += tot
                    raw_groups_data[norm_gn]["success"] += succ
                    raw_groups_data[norm_gn]["failed"] += fail
                    raw_groups_data[norm_gn]["total_latency_ms"] += float(row["avg_latency_ms"] or 0.0) * tot

                groups_data = []
                for gn, gd in sorted(raw_groups_data.items(), key=lambda x: x[1]["total"], reverse=True):
                    avg_lat = gd["total_latency_ms"] / gd["total"] if gd["total"] > 0 else 0.0
                    groups_data.append({
                        "group_name": gn,
                        "total": gd["total"],
                        "success": gd["success"],
                        "failed": gd["failed"],
                        "avg_latency_ms": round(avg_lat, 1),
                    })

                active_group_names = [str(g["group_name"]) for g in groups_data]

                # Main chain strictly excludes dedicated/light groups
                main_chain = [
                    gn for gn in clean_chain
                    if not any(k in gn.lower() for k in ("light", "dedicated"))
                ]

                group_roles: dict[str, str] = {}
                # Dedicated groups (e.g. light_llm_pool) are ALWAYS dedicated, NEVER terminal/保底!
                for gn in active_group_names:
                    if any(k in gn.lower() for k in ("light", "dedicated")):
                        group_roles[gn] = "dedicated"

                if main_chain and len(main_chain) >= 2:
                    primary_grp = main_chain[0]
                    terminal_grp = main_chain[-1]
                    buffer_grps = set(main_chain[1:-1])

                    for gn in active_group_names:
                        if gn in group_roles:
                            continue
                        if gn == primary_grp:
                            group_roles[gn] = "primary"
                        elif gn == terminal_grp:
                            group_roles[gn] = "terminal"
                        elif gn in buffer_grps:
                            group_roles[gn] = "buffer"
                        else:
                            gn_lower = gn.lower()
                            if any(k in gn_lower for k in ("official", "paid", "final", "terminal")):
                                group_roles[gn] = "terminal"
                            else:
                                group_roles[gn] = "buffer"
                elif main_chain and len(main_chain) == 1:
                    primary_grp = main_chain[0]
                    for gn in active_group_names:
                        if gn in group_roles:
                            continue
                        if gn == primary_grp:
                            group_roles[gn] = "primary"
                        else:
                            group_roles[gn] = "buffer"
                else:
                    # No explicit main chain provided: infer from keywords or natural order
                    non_dedicated_names = [
                        gn for gn in active_group_names
                        if not any(k in gn.lower() for k in ("light", "dedicated"))
                    ]
                    has_keywords = any(
                        any(k in gn.lower() for k in ("fallback", "official", "relay", "agnes"))
                        for gn in non_dedicated_names
                    )
                    if has_keywords:
                        for gn in non_dedicated_names:
                            gn_lower = gn.lower()
                            if "official" in gn_lower or "ds-official" in gn_lower:
                                group_roles[gn] = "terminal"
                            elif "agnes" in gn_lower or "relay" in gn_lower or "fallback" in gn_lower:
                                group_roles[gn] = "buffer"
                            else:
                                group_roles[gn] = "primary"
                    elif len(non_dedicated_names) >= 2:
                        sorted_names = sorted(non_dedicated_names)
                        group_roles[sorted_names[0]] = "primary"
                        group_roles[sorted_names[-1]] = "terminal"
                        for gn in sorted_names[1:-1]:
                            group_roles[gn] = "buffer"
                    elif non_dedicated_names:
                        group_roles[non_dedicated_names[0]] = "primary"

                primary_reqs = sum(g["total"] for g in groups_data if group_roles.get(g["group_name"]) == "primary")
                buffer_reqs = sum(g["total"] for g in groups_data if group_roles.get(g["group_name"]) == "buffer")
                terminal_reqs = sum(g["total"] for g in groups_data if group_roles.get(g["group_name"]) == "terminal")
                dedicated_reqs = sum(g["total"] for g in groups_data if group_roles.get(g["group_name"]) == "dedicated")

                # Rates for main-chain tiers exclude dedicated requests from the denominator
                # so that primary/leakage percentages reflect the main funnel only.
                main_chain_reqs = total_all_reqs - dedicated_reqs
                primary_rate = round((primary_reqs / main_chain_reqs * 100.0), 1) if main_chain_reqs > 0 else 100.0
                buffer_rate = round((buffer_reqs / main_chain_reqs * 100.0), 1) if main_chain_reqs > 0 else 0.0
                leakage_rate = round((terminal_reqs / main_chain_reqs * 100.0), 1) if main_chain_reqs > 0 else 0.0
                dedicated_rate = round((dedicated_reqs / total_all_reqs * 100.0), 1) if total_all_reqs > 0 else 0.0

                primary_names = [gn for gn, r in group_roles.items() if r == "primary"]
                terminal_names = [gn for gn, r in group_roles.items() if r == "terminal"]
                buffer_names = [gn for gn, r in group_roles.items() if r == "buffer"]
                primary_label = ", ".join(primary_names) if primary_names else "主力层"
                terminal_label = ", ".join(terminal_names) if terminal_names else "保底层"
                buffer_label = ", ".join(buffer_names) if buffer_names else "后备层"

                # Dynamic Funnel Tiers
                role_priority = {"primary": 0, "buffer": 1, "terminal": 2, "dedicated": 3}
                role_colors = {
                    "primary": "seg-free",
                    "buffer": "seg-fallback",
                    "terminal": "seg-official",
                    "dedicated": "seg-relay",
                }
                role_desc_map = {
                    "primary": "主力层",
                    "buffer": "后备层",
                    "terminal": "保底层",
                    "dedicated": "专用层",
                }

                def _tier_sort_key(gn: str) -> tuple[int, int]:
                    role = group_roles.get(gn, "buffer")
                    role_ord = role_priority.get(role, 9)
                    chain_ord = clean_chain.index(gn) if gn in clean_chain else 999
                    return (role_ord, chain_ord)

                funnel_tiers = []
                for gn in sorted(active_group_names, key=_tier_sort_key):
                    g_info = next((x for x in groups_data if x["group_name"] == gn), None)
                    if not g_info:
                        continue
                    role = group_roles.get(gn, "buffer")
                    role_desc = role_desc_map.get(role, "梯队")
                    rate = round((g_info["total"] / total_all_reqs * 100.0), 1) if total_all_reqs > 0 else 0.0
                    funnel_tiers.append({
                        "group_name": gn,
                        "role": role,
                        "name": f"{gn} ({role_desc})",
                        "count": g_info["total"],
                        "rate": rate,
                        "avg_latency_ms": g_info["avg_latency_ms"],
                        "cls": role_colors.get(role, "seg-fallback"),
                    })

                # 2. Hourly distribution across 0-23
                cursor.execute(
                    """
                    SELECT
                        hour_of_day,
                        group_name,
                        COUNT(*) as count
                    FROM request_metrics
                    WHERE timestamp >= ?
                    GROUP BY hour_of_day, group_name
                    ORDER BY hour_of_day ASC
                    """,
                    (cutoff,),
                )
                hourly_map: dict[int, dict[str, Any]] = {
                    h: {
                        "primary": 0,
                        "buffer": 0,
                        "dedicated": 0,
                        "terminal": 0,
                        "total": 0,
                        "by_group": {},
                    }
                    for h in range(24)
                }
                for row in cursor.fetchall():
                    h = int(row["hour_of_day"])
                    raw_gn = str(row["group_name"] or "")
                    norm_gn = _norm_gn(raw_gn)
                    cnt = int(row["count"] or 0)
                    if h in hourly_map:
                        hourly_map[h]["total"] += cnt
                        hourly_map[h]["by_group"][norm_gn] = hourly_map[h]["by_group"].get(norm_gn, 0) + cnt
                        role = group_roles.get(norm_gn, "buffer")
                        hourly_map[h][role] = hourly_map[h].get(role, 0) + cnt

                hourly_tiers = []
                for h in range(24):
                    d = hourly_map[h]
                    tot = d["total"]
                    hourly_tiers.append(
                        {
                            "hour": h,
                            "total": tot,
                            "free": d["primary"],
                            "fallback": d["buffer"],
                            "relay": d["dedicated"],
                            "official": d["terminal"],
                            "free_pct": round(d["primary"] / tot * 100.0, 1) if tot > 0 else 100.0,
                            "by_group": d.get("by_group", {}),
                        }
                    )

                # 3. Top Contributors (Hero nodes)
                cursor.execute(
                    """
                    SELECT
                        node_id,
                        group_name,
                        model_name,
                        COUNT(*) as total,
                        SUM(CASE WHEN status = 'SUCCESS' THEN 1 ELSE 0 END) as success,
                        AVG(total_latency_ms) as avg_latency_ms
                    FROM request_metrics
                    WHERE timestamp >= ?
                    GROUP BY node_id
                    ORDER BY success DESC, total DESC
                    LIMIT 5
                    """,
                    (cutoff,),
                )
                top_contributors = []
                for row in cursor.fetchall():
                    tot = int(row["total"] or 0)
                    succ = int(row["success"] or 0)
                    top_contributors.append(
                        {
                            "node_id": row["node_id"],
                            "group_name": _norm_gn(str(row["group_name"] or "")),
                            "model_name": row["model_name"],
                            "total": tot,
                            "success": succ,
                            "avail_pct": round(succ / tot * 100.0, 1) if tot > 0 else 0.0,
                            "avg_latency_ms": round(float(row["avg_latency_ms"] or 0.0), 1),
                        }
                    )

                # 4. Leakage Culprits (Nodes causing frequent failures)
                cursor.execute(
                    """
                    SELECT
                        node_id,
                        group_name,
                        model_name,
                        COUNT(*) as total,
                        SUM(CASE WHEN status != 'SUCCESS' THEN 1 ELSE 0 END) as failed,
                        AVG(total_latency_ms) as avg_latency_ms
                    FROM request_metrics
                    WHERE timestamp >= ?
                    GROUP BY node_id
                    HAVING failed > 0
                    ORDER BY failed DESC, total DESC
                    LIMIT 5
                    """,
                    (cutoff,),
                )
                leakage_culprits = []
                for row in cursor.fetchall():
                    tot = int(row["total"] or 0)
                    fail = int(row["failed"] or 0)
                    fail_rate = round(fail / tot * 100.0, 1) if tot > 0 else 0.0
                    leakage_culprits.append(
                        {
                            "node_id": row["node_id"],
                            "group_name": _norm_gn(str(row["group_name"] or "")),
                            "model_name": row["model_name"],
                            "total": tot,
                            "failed": fail,
                            "fail_rate_pct": fail_rate,
                            "avg_latency_ms": round(float(row["avg_latency_ms"] or 0.0), 1),
                        }
                    )

                # 5. Smart Insights
                insights = []
                if primary_rate >= 80.0:
                    insights.append(
                        {
                            "type": "success",
                            "title": "主力梯队拦截极为高效",
                            "desc": f"近 24 小时主力组 ({primary_label}) 成功消化了 {primary_rate:.1f}% 的流量，避免了向下渗漏至后备保底组。",
                        }
                    )
                elif primary_rate < 50.0:
                    insights.append(
                        {
                            "type": "warning",
                            "title": "主力梯队拦截率偏低",
                            "desc": f"主力组 ({primary_label}) 仅消化了 {primary_rate:.1f}% 的请求，超过一半流量下渗到后备层 ({buffer_label})，建议检查主力组健康节点或增加配额。",
                        }
                    )

                if leakage_culprits:
                    worst = leakage_culprits[0]
                    if worst["fail_rate_pct"] > 50.0 and worst["failed"] >= 3:
                        insights.append(
                            {
                                "type": "danger",
                                "title": f"发现高损耗节点：{worst['node_id']} (组: {worst['group_name']})",
                                "desc": f"该节点累计失败 {worst['failed']} 次（失败率 {worst['fail_rate_pct']}%），易导致请求向下梯队击穿，建议开启【⚡ 激进抢跑】或从池中移除。",
                            }
                        )

                if leakage_rate > 10.0:
                    insights.append(
                        {
                            "type": "danger",
                            "title": f"末级保底梯队 ({terminal_label}) 承接偏高",
                            "desc": f"末级保底组承接了 {leakage_rate:.1f}% 的请求（超过 10% 警戒线），说明前序各层发生较多重试或熔断，建议补充前序层健康节点。",
                        }
                    )
                elif total_all_reqs > 0:
                    insights.append(
                        {
                            "type": "info",
                            "title": "梯队后备防线稳固",
                            "desc": f"末级保底组 ({terminal_label}) 仅承接 {leakage_rate:.1f}% 流量，绝大部分请求已在前序主力与缓冲梯队被平稳消化。",
                        }
                    )

                return {
                    "total_requests": total_all_reqs,
                    "primary_requests": primary_reqs,
                    "buffer_requests": buffer_reqs,
                    "terminal_requests": terminal_reqs,
                    "dedicated_requests": dedicated_reqs,
                    "primary_rate": primary_rate,
                    "buffer_rate": buffer_rate,
                    "leakage_rate": leakage_rate,
                    "dedicated_rate": dedicated_rate,
                    "primary_label": primary_label,
                    "terminal_label": terminal_label,
                    "buffer_label": buffer_label,
                    "funnel_tiers": funnel_tiers,
                    # Backward compatibility keys for existing tests & UI
                    "free_requests": primary_reqs,
                    "fallback_requests": buffer_reqs,
                    "relay_requests": dedicated_reqs,
                    "official_requests": terminal_reqs,
                    "free_rate": primary_rate,
                    "fallback_rate": buffer_rate,
                    "relay_rate": dedicated_rate,
                    "groups_data": groups_data,
                    "hourly_tiers": hourly_tiers,
                    "top_contributors": top_contributors,
                    "leakage_culprits": leakage_culprits,
                    "insights": insights,
                }
            finally:
                conn.close()

        async with self._lock:
            return await asyncio.to_thread(_query)

    async def record_hedge(
        self,
        *,
        group_name: str,
        primary_node: str,
        sibling_node: str,
        winner_node: str,
        sibling_won: bool,
        saved_ms: float = 0.0,
    ) -> None:
        """Record a speculative hedging racing event and whether the sibling won."""
        now = time.time()

        def _insert() -> None:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO hedge_metrics (
                        timestamp, group_name, primary_node, sibling_node, winner_node, sibling_won, saved_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (now, group_name, primary_node, sibling_node, winner_node, 1 if sibling_won else 0, max(0.0, saved_ms)),
                )
                conn.commit()
            finally:
                conn.close()

        async with self._lock:
            await asyncio.to_thread(_insert)

    async def get_error_attribution(self, window_seconds: float = 86400.0) -> dict[str, Any]:
        """Compute error type distribution and per-node failure breakdown."""
        cutoff = time.time() - window_seconds

        def _query() -> dict[str, Any]:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT
                        error_type,
                        COUNT(*) as count
                    FROM request_metrics
                    WHERE timestamp >= ? AND status != 'SUCCESS'
                    GROUP BY error_type
                    ORDER BY count DESC
                    """,
                    (cutoff,),
                )
                type_rows = cursor.fetchall()
                total_errors = sum(int(r["count"] or 0) for r in type_rows)

                cat_labels = {
                    "RateLimitError": {"name": "429 频率超限 (Rate Limit)", "icon": "🟡", "color": "#d97706"},
                    "InsufficientQuotaError": {"name": "账户欠费/额度耗尽 (Quota Exceeded)", "icon": "🔴", "color": "#dc2626"},
                    "TimeoutError": {"name": "请求超时 (Timeout)", "icon": "⏱️", "color": "#e11d48"},
                    "UpstreamDeadError": {"name": "上游渠道故障 (502/503)", "icon": "💀", "color": "#7c3aed"},
                    "BadRequestError": {"name": "参数格式错误 (400 Bad Request)", "icon": "⚠️", "color": "#f59e0b"},
                    "VisionUnsupportedError": {"name": "不支持视觉图片输入", "icon": "🖼️", "color": "#0ea5e9"},
                    "OtherError": {"name": "其他异常 (Other Error)", "icon": "❓", "color": "#6b7280"},
                }

                categories = []
                for r in type_rows:
                    et = str(r["error_type"] or "OtherError")
                    cnt = int(r["count"] or 0)
                    meta = cat_labels.get(et, {"name": et, "icon": "⚠️", "color": "#6b7280"})
                    categories.append({
                        "error_type": et,
                        "name": meta["name"],
                        "icon": meta["icon"],
                        "color": meta["color"],
                        "count": cnt,
                        "percentage": round(cnt / total_errors * 100.0, 1) if total_errors > 0 else 0.0,
                    })

                cursor.execute(
                    """
                    SELECT
                        node_id,
                        group_name,
                        error_type,
                        COUNT(*) as count
                    FROM request_metrics
                    WHERE timestamp >= ? AND status != 'SUCCESS'
                    GROUP BY node_id, error_type
                    ORDER BY count DESC
                    """,
                    (cutoff,),
                )
                node_fails: dict[str, dict[str, Any]] = {}
                for r in cursor.fetchall():
                    nid = str(r["node_id"])
                    et = str(r["error_type"] or "OtherError")
                    cnt = int(r["count"] or 0)
                    if nid not in node_fails:
                        node_fails[nid] = {
                            "node_id": nid,
                            "group_name": str(r["group_name"] or "default"),
                            "total_errors": 0,
                            "error_breakdown": {},
                        }
                    node_fails[nid]["total_errors"] += cnt
                    node_fails[nid]["error_breakdown"][et] = cnt

                node_rankings = []
                for nf in sorted(node_fails.values(), key=lambda x: x["total_errors"], reverse=True)[:8]:
                    top_err = max(nf["error_breakdown"].items(), key=lambda x: x[1])[0] if nf["error_breakdown"] else "OtherError"
                    top_meta = cat_labels.get(top_err, {"name": top_err, "icon": "⚠️"})
                    node_rankings.append({
                        "node_id": nf["node_id"],
                        "group_name": nf["group_name"],
                        "total_errors": nf["total_errors"],
                        "primary_error": top_err,
                        "primary_error_name": top_meta["name"],
                        "primary_error_icon": top_meta["icon"],
                        "error_breakdown": nf["error_breakdown"],
                    })

                return {
                    "total_errors": total_errors,
                    "categories": categories,
                    "node_rankings": node_rankings,
                }
            finally:
                conn.close()

        async with self._lock:
            return await asyncio.to_thread(_query)

    async def get_hedging_stats(self, window_seconds: float = 86400.0) -> dict[str, Any]:
        """Compute speculative hedging racing stats and saved latency."""
        cutoff = time.time() - window_seconds

        def _query() -> dict[str, Any]:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT
                        COUNT(*) as total_races,
                        SUM(CASE WHEN sibling_won = 1 THEN 1 ELSE 0 END) as sibling_won_count,
                        SUM(saved_ms) as total_saved_ms
                    FROM hedge_metrics
                    WHERE timestamp >= ?
                    """,
                    (cutoff,),
                )
                row = cursor.fetchone()
                total_races = int(row["total_races"] or 0) if row else 0
                sibling_won = int(row["sibling_won_count"] or 0) if row else 0
                total_saved_ms = float(row["total_saved_ms"] or 0.0) if row else 0.0

                overtake_rate = round(sibling_won / total_races * 100.0, 1) if total_races > 0 else 0.0
                saved_seconds = round(total_saved_ms / 1000.0, 1)

                cursor.execute(
                    """
                    SELECT
                        timestamp,
                        group_name,
                        primary_node,
                        sibling_node,
                        winner_node,
                        sibling_won,
                        saved_ms
                    FROM hedge_metrics
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                    LIMIT 10
                    """,
                    (cutoff,),
                )
                recent_events = []
                for r in cursor.fetchall():
                    recent_events.append({
                        "timestamp": float(r["timestamp"]),
                        "group_name": str(r["group_name"]),
                        "primary_node": str(r["primary_node"]),
                        "sibling_node": str(r["sibling_node"]),
                        "winner_node": str(r["winner_node"]),
                        "sibling_won": bool(r["sibling_won"]),
                        "saved_seconds": round(float(r["saved_ms"] or 0.0) / 1000.0, 2),
                    })

                return {
                    "total_races": total_races,
                    "sibling_won_count": sibling_won,
                    "overtake_rate": overtake_rate,
                    "saved_seconds": saved_seconds,
                    "recent_events": recent_events,
                }
            finally:
                conn.close()

        async with self._lock:
            return await asyncio.to_thread(_query)

    async def get_weekly_heatmap(self, days: int = 7) -> dict[str, Any]:
        """Compute 7x24 weekly heatmap matrix for request traffic and latency."""
        cutoff = time.time() - (days * 86400.0)

        def _query() -> dict[str, Any]:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT
                        timestamp,
                        hour_of_day,
                        total_latency_ms,
                        status
                    FROM request_metrics
                    WHERE timestamp >= ?
                    """,
                    (cutoff,),
                )

                matrix = [
                    [{"requests": 0, "failures": 0, "total_latency": 0.0} for _ in range(24)]
                    for _ in range(7)
                ]

                max_requests = 0
                peak_day = 0
                peak_hour = 0
                total_volume = 0

                for row in cursor.fetchall():
                    ts = float(row["timestamp"])
                    import datetime
                    dt = datetime.datetime.fromtimestamp(ts)
                    w = dt.weekday()
                    h = int(row["hour_of_day"]) % 24
                    lat = float(row["total_latency_ms"] or 0.0)
                    is_fail = 1 if row["status"] != "SUCCESS" else 0

                    cell = matrix[w][h]
                    cell["requests"] += 1
                    cell["failures"] += is_fail
                    cell["total_latency"] += lat
                    total_volume += 1

                    if cell["requests"] > max_requests:
                        max_requests = cell["requests"]
                        peak_day = w
                        peak_hour = h

                weekday_labels = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
                formatted_matrix = []
                for w in range(7):
                    row_data = []
                    for h in range(24):
                        c = matrix[w][h]
                        reqs = c["requests"]
                        fails = c["failures"]
                        avg_lat = round(c["total_latency"] / reqs, 1) if reqs > 0 else 0.0
                        fail_rate = round(fails / reqs * 100.0, 1) if reqs > 0 else 0.0
                        intensity = round(reqs / max_requests, 3) if max_requests > 0 else 0.0
                        row_data.append({
                            "hour": h,
                            "requests": reqs,
                            "failures": fails,
                            "fail_rate": fail_rate,
                            "avg_latency_ms": avg_lat,
                            "intensity": intensity,
                        })
                    formatted_matrix.append({
                        "weekday_index": w,
                        "weekday_label": weekday_labels[w],
                        "hours": row_data,
                    })

                return {
                    "total_requests": total_volume,
                    "max_cell_requests": max_requests,
                    "peak_weekday": weekday_labels[peak_day],
                    "peak_hour": f"{peak_hour:02d}:00",
                    "matrix": formatted_matrix,
                }
            finally:
                conn.close()

        async with self._lock:
            return await asyncio.to_thread(_query)
