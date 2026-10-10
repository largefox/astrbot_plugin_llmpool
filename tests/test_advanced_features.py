"""Unit tests for CircuitBreaker, MetricsStore, and UsageBasedRouter."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from circuit_breaker import CircuitBreaker, ErrorCategory, classify_error
from key_manager import KeyManager, is_auth_error
from metrics_store import MetricsStore
from routing import PoolMember, UsageBasedRouter
from vision_utils import (
    is_context_format_error,
    is_vision_unsupported_error,
    request_contains_image,
    sanitize_contexts,
)


class CircuitBreakerTests(unittest.IsolatedAsyncioTestCase):
    async def test_classify_error(self) -> None:
        self.assertEqual(classify_error("HTTP 429: rate limit exceeded"), ErrorCategory.RATE_LIMIT)
        self.assertEqual(classify_error("Request timed out after 30s"), ErrorCategory.TIMEOUT)
        self.assertEqual(classify_error("400 Bad Request: missing param"), ErrorCategory.BAD_REQUEST)
        self.assertEqual(classify_error("Internal server error 500"), ErrorCategory.OTHER)
        self.assertEqual(classify_error("Error: insufficient_quota"), ErrorCategory.INSUFFICIENT_QUOTA)
        self.assertEqual(classify_error("Your balance is not enough, 账户已欠费"), ErrorCategory.INSUFFICIENT_QUOTA)
        self.assertEqual(classify_error("You exceeded your current quota"), ErrorCategory.INSUFFICIENT_QUOTA)

    async def test_circuit_breaker_records_trip_error_and_distinguishes_reasons(self) -> None:
        cb = CircuitBreaker()
        node_quota = "node-quota-test"
        quota_err = "Error code: 400 - insufficient_quota: You exceeded your current quota"
        tripped, cat, cd = await cb.mark_failure(node_quota, quota_err)
        self.assertTrue(tripped)
        self.assertEqual(cat, ErrorCategory.INSUFFICIENT_QUOTA)
        st_q = cb.get_state(node_quota)
        self.assertTrue(st_q.is_deep_sleep)
        self.assertIn("欠费/额度耗尽", st_q.trip_reason)
        self.assertEqual(st_q.trip_error_msg, quota_err)
        self.assertEqual(st_q.last_error_category, "InsufficientQuotaError")

        # Check in get_all_status
        all_status = await cb.get_all_status()
        self.assertEqual(all_status[node_quota]["status"], "DEEP-SLEEP")
        self.assertIn("欠费/额度耗尽", all_status[node_quota]["trip_reason"])
        self.assertEqual(all_status[node_quota]["trip_error_msg"], quota_err)

        # Rate limit node
        node_rl = "node-rl-test"
        rl_err = "429 Too Many Requests: Rate limit reached for requests per minute"
        tripped_rl, cat_rl, cd_rl = await cb.mark_failure(node_rl, rl_err)
        self.assertTrue(tripped_rl)
        self.assertEqual(cat_rl, ErrorCategory.RATE_LIMIT)
        st_rl = cb.get_state(node_rl)
        self.assertFalse(st_rl.is_deep_sleep)
        self.assertTrue(st_rl.is_open)
        self.assertIn("429 限频保护", st_rl.trip_reason)
        self.assertEqual(st_rl.trip_error_msg, rl_err)
        self.assertEqual(st_rl.last_error_category, "RateLimitError")

        # Reset clears trip attribution
        await cb.reset(node_quota)
        self.assertEqual(st_q.trip_reason, "")
        self.assertEqual(st_q.trip_error_msg, "")
        self.assertEqual(st_q.last_error_msg, "")

    async def test_context_format_error_detection_and_sanitization(self) -> None:
        err_msg = 'Error code: 400 - {"error":{"message":"Invalid JSON data: Failed to deserialize the JSON body into the target type: messages[4]: data did not match any variant of untagged enum MessageContent"}}'
        self.assertTrue(is_context_format_error(err_msg))
        self.assertTrue(is_context_format_error("invalid content"))
        self.assertTrue(is_context_format_error("请求中存在目标接口不支持的字段"))

        # Test context sanitization
        bad_contexts = [
            {"role": "user", "content": "你好"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "发几张表情包"},
                    {"type": "image_url", "image_url": {"url": "http://example.com/bad.jpg"}},
                ],
            },
        ]
        cleaned = sanitize_contexts(bad_contexts)
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(cleaned[0]["content"], "你好")
        self.assertIn("发几张表情包", cleaned[1]["content"])
        self.assertIn("[图片]", cleaned[1]["content"])
        self.assertIsInstance(cleaned[1]["content"], str)

    async def test_circuit_breaker_rate_limit_trips_immediately(self) -> None:
        cb = CircuitBreaker(
            cooldown_time=1800.0,
            allowed_fails=2,
            allowed_fails_policy={
                ErrorCategory.RATE_LIMIT.value: 0,
                ErrorCategory.TIMEOUT.value: 2,
            },
        )
        node = "test-node-1"
        self.assertTrue(await cb.is_available(node))

        # 0 allowed fails for RATE_LIMIT -> 1 failure trips immediately
        tripped, category, cd = await cb.mark_failure(node, "429 Too Many Requests")
        self.assertTrue(tripped)
        self.assertEqual(category, ErrorCategory.RATE_LIMIT)
        self.assertFalse(await cb.is_available(node))

        # Resetting restores availability
        await cb.reset(node)
        self.assertTrue(await cb.is_available(node))

    async def test_circuit_breaker_reset_all_nodes(self) -> None:
        cb = CircuitBreaker(
            cooldown_time=1800.0,
            allowed_fails_policy={
                ErrorCategory.RATE_LIMIT.value: 0,
            },
        )
        node1 = "node-alpha"
        node2 = "node-beta"
        await cb.mark_failure(node1, "429 Too Many Requests")
        await cb.mark_failure(node2, "429 Quota Exceeded")
        self.assertFalse(await cb.is_available(node1))
        self.assertFalse(await cb.is_available(node2))

        # Reset all with None
        await cb.reset(None)
        self.assertTrue(await cb.is_available(node1))
        self.assertTrue(await cb.is_available(node2))

    async def test_circuit_breaker_timeout_tolerates_threshold(self) -> None:
        cb = CircuitBreaker(
            cooldown_time=1800.0,
            allowed_fails_policy={
                ErrorCategory.TIMEOUT.value: 2,
            },
        )
        node = "timeout-node"
        # First timeout: tolerated
        tripped, _, _ = await cb.mark_failure(node, "Read timeout")
        self.assertFalse(tripped)
        self.assertTrue(await cb.is_available(node))

        # Second timeout: tolerated (threshold is 2)
        tripped, _, _ = await cb.mark_failure(node, "Read timeout")
        self.assertFalse(tripped)
        self.assertTrue(await cb.is_available(node))

        # Third timeout: trips!
        tripped, _, _ = await cb.mark_failure(node, "Read timeout")
        self.assertTrue(tripped)
        self.assertFalse(await cb.is_available(node))

    async def test_cold_start_smoothing_and_outlier_clamping(self) -> None:
        cb = CircuitBreaker()
        node = "cold-start-node"
        st = cb.get_state(node)

        # 1. Cold start prior check (0 samples -> 95% neutral prior, not 0%)
        self.assertAlmostEqual(st.get_recent_availability(), 0.95, delta=0.01)

        # 2. Single failure out of 1 request should not drop availability to 0% (Bayesian smoothing)
        await cb.mark_failure(node, "Temporary glitch", latency_ms=60000.0)
        self.assertGreater(st.get_recent_availability(), 0.70)

        # 3. Outlier clamping: 60s (60000ms) latency should be clamped to 12000ms max in EWMA
        self.assertLess(st.ewma_latency_ms, 12000.0)

    async def test_circuit_state_persistence_across_restarts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p_path = Path(tmp) / "circuit_state.json"

            # Instance 1: Mark a failure and a success with EWMA
            cb1 = CircuitBreaker(
                cooldown_time=1800.0,
                allowed_fails_policy={ErrorCategory.RATE_LIMIT.value: 0},
                persistence_path=p_path,
            )
            await cb1.mark_failure("bad-node", "429 rate limit exceeded")
            await cb1.mark_success("fast-node", latency_ms=450.0, ttft_ms=180.0)

            self.assertFalse(await cb1.is_available("bad-node"))
            self.assertTrue(await cb1.is_available("fast-node"))
            self.assertLess(cb1.get_state("fast-node").ewma_ttft_ms, 1000.0)

            # Instance 2 (simulating restart): load from same persistence_path
            cb2 = CircuitBreaker(
                cooldown_time=1800.0,
                allowed_fails_policy={ErrorCategory.RATE_LIMIT.value: 0},
                persistence_path=p_path,
            )

            # bad-node should STILL be in cooldown (not reset to healthy)
            self.assertFalse(await cb2.is_available("bad-node"))
            # fast-node should keep its warm EWMA latency/ttft profile
            st_fast = cb2.get_state("fast-node")
            self.assertTrue(await cb2.is_available("fast-node"))
            self.assertAlmostEqual(st_fast.ewma_ttft_ms, cb1.get_state("fast-node").ewma_ttft_ms, delta=1.0)
            self.assertAlmostEqual(st_fast.ewma_latency_ms, cb1.get_state("fast-node").ewma_latency_ms, delta=1.0)


class MetricsStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_metrics_record_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test_metrics.db"
            store = MetricsStore(db_path)

            await store.record_request(
                group_name="gemini-3-flash",
                node_id="node-a",
                model_name="gemini-3.1-flash-lite",
                is_stream=True,
                ttft_ms=120.5,
                total_latency_ms=450.0,
                status="SUCCESS",
            )
            await store.record_request(
                group_name="gemini-3-flash",
                node_id="node-b",
                model_name="deepseek-v4-flash",
                is_stream=False,
                total_latency_ms=800.0,
                status="FAILED",
                error_type="RateLimitError",
                error_msg="Rate limit reached",
            )

            summary = await store.get_summary_stats(window_seconds=3600)
            self.assertEqual(summary["total_requests"], 2)
            self.assertEqual(summary["success_requests"], 1)
            self.assertEqual(summary["failed_requests"], 1)
            self.assertEqual(summary["availability_pct"], 50.0)

            nodes = await store.get_node_stats(window_seconds=3600)
            self.assertEqual(len(nodes), 2)

            hourly = await store.get_hourly_profile(days=1)
            self.assertEqual(len(hourly), 24)

            # Paginated logs check
            data = await store.get_recent_logs(page=1, page_size=10)
            self.assertEqual(data["total"], 2)
            self.assertEqual(len(data["logs"]), 2)
            self.assertEqual(data["logs"][0]["status"], "FAILED")

            # Keyword filtering check
            kw_data = await store.get_recent_logs(keyword="deepseek")
            self.assertEqual(kw_data["total"], 1)
            self.assertEqual(kw_data["logs"][0]["node_id"], "node-b")

    async def test_cost_analysis_tier_funnel_and_insights(self) -> None:
        """Verify deep cost analysis computes tier rates, hourly spillover, and smart insights."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "metrics.db"
            store = MetricsStore(db_path)

            # Record 3 free tier requests
            for _ in range(3):
                await store.record_request(
                    group_name="gemini-3-flash",
                    node_id="free-node-1",
                    model_name="flash",
                    is_stream=False,
                    total_latency_ms=1200.0,
                    status="SUCCESS",
                )
            # Record 1 fallback request
            await store.record_request(
                group_name="gemini-3-flash-fallback",
                node_id="fallback-node-1",
                model_name="flash-fallback",
                is_stream=False,
                total_latency_ms=2500.0,
                status="SUCCESS",
            )
            # Record 1 failed official request
            await store.record_request(
                group_name="deepseek-official",
                node_id="ds-official-node",
                model_name="deepseek-chat",
                is_stream=False,
                total_latency_ms=5000.0,
                status="FAILED",
                error_type="TimeoutError",
            )

            cost_data = await store.get_cost_analysis(window_seconds=3600)
            self.assertEqual(cost_data["total_requests"], 5)
            self.assertEqual(cost_data["free_requests"], 3)
            self.assertEqual(cost_data["fallback_requests"], 1)
            self.assertEqual(cost_data["official_requests"], 1)
            self.assertAlmostEqual(cost_data["free_rate"], 60.0, delta=0.1)
            self.assertAlmostEqual(cost_data["leakage_rate"], 20.0, delta=0.1)
            self.assertEqual(len(cost_data["hourly_tiers"]), 24)
            self.assertGreaterEqual(len(cost_data["top_contributors"]), 1)
            self.assertGreaterEqual(len(cost_data["leakage_culprits"]), 1)
            self.assertGreaterEqual(len(cost_data["insights"]), 1)

    async def test_cost_analysis_with_arbitrary_custom_group_names(self) -> None:
        """Verify cost analysis normalizes comma-separated groups and isolates dedicated light groups from terminal tiers."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "custom_metrics.db"
            store = MetricsStore(db_path)

            for _ in range(8):
                await store.record_request(
                    group_name="tier1",
                    node_id="t1-node",
                    model_name="flash",
                    is_stream=False,
                    total_latency_ms=1000.0,
                    status="SUCCESS",
                )
            for _ in range(2):
                await store.record_request(
                    group_name="tier2",
                    node_id="t2-node",
                    model_name="pro",
                    is_stream=False,
                    total_latency_ms=2000.0,
                    status="SUCCESS",
                )
            # Historical records with multi-group names: tier3,light_llm_pool
            for _ in range(3):
                await store.record_request(
                    group_name="tier3,light_llm_pool",
                    node_id="t3-node",
                    model_name="opus",
                    is_stream=False,
                    total_latency_ms=3000.0,
                    status="SUCCESS",
                )
            # Terminal tier4
            await store.record_request(
                group_name="tier4",
                node_id="t4-node",
                model_name="max",
                is_stream=False,
                total_latency_ms=4000.0,
                status="SUCCESS",
            )
            # Dedicated light_llm_pool
            await store.record_request(
                group_name="light_llm_pool",
                node_id="light-node",
                model_name="fast",
                is_stream=False,
                total_latency_ms=500.0,
                status="SUCCESS",
            )

            # Pass dynamic fallback chain ['tier1', 'tier2', 'tier3', 'tier4']
            cost_data = await store.get_cost_analysis(
                window_seconds=3600,
                fallback_chain=["tier1", "tier2", "tier3", "tier4"],
            )

            self.assertEqual(cost_data["total_requests"], 15)
            self.assertEqual(cost_data["primary_requests"], 8)
            # buffer = tier2 (2) + tier3 (3) = 5
            self.assertEqual(cost_data["buffer_requests"], 5)
            # terminal = tier4 (1)
            self.assertEqual(cost_data["terminal_requests"], 1)
            # dedicated = light_llm_pool (1)
            self.assertEqual(cost_data["dedicated_requests"], 1)
            self.assertEqual(cost_data["primary_label"], "tier1")
            self.assertEqual(cost_data["terminal_label"], "tier4")
            # Rates now exclude dedicated requests from the denominator:
            # main_chain_reqs = 15 - 1 (dedicated) = 14
            # primary_rate = 8/14 = 57.1%, leakage_rate = 1/14 = 7.1%
            self.assertAlmostEqual(cost_data["primary_rate"], 57.1, delta=0.1)
            self.assertAlmostEqual(cost_data["leakage_rate"], 7.1, delta=0.1)

            # Verify dynamic funnel tiers: tier1 (primary), tier2 (buffer), tier3 (buffer), tier4 (terminal), light_llm_pool (dedicated)
            funnel_tiers = cost_data["funnel_tiers"]
            tier_names = [t["group_name"] for t in funnel_tiers]
            self.assertIn("tier1", tier_names)
            self.assertIn("tier2", tier_names)
            self.assertIn("tier3", tier_names)
            self.assertIn("tier4", tier_names)
            self.assertIn("light_llm_pool", tier_names)
            self.assertNotIn("tier3,light_llm_pool", tier_names)

            t4_tier = next(t for t in funnel_tiers if t["group_name"] == "tier4")
            self.assertEqual(t4_tier["role"], "terminal")

            light_tier = next(t for t in funnel_tiers if t["group_name"] == "light_llm_pool")
            self.assertEqual(light_tier["role"], "dedicated")

    async def test_resolve_dedicated_pools_from_list(self) -> None:
        """Verify resolve_dedicated_pools reads the dedicated_pools list directly."""
        from lifecycle import resolve_dedicated_pools

        plugin_config = {
            "dedicated_pools": [
                {
                    "id": "pool_a",
                    "group": "group_a",
                    "model_name": "pool-a-router",
                    "strict_group": True,
                    "strategy": "round_robin",
                    "max_attempts": 5,
                    "request_timeout_seconds": 15,
                    "stream_timeout": 10,
                },
                {
                    "id": "pool_b",
                    "group": "group_b",
                    "model_name": "pool-b-router",
                },
            ],
        }
        pools = resolve_dedicated_pools(plugin_config)
        self.assertEqual(len(pools), 2)
        self.assertEqual(pools[0]["id"], "pool_a")
        self.assertEqual(pools[0]["group"], "group_a")
        self.assertTrue(pools[0]["strict_group"])
        self.assertEqual(pools[0]["strategy"], "round_robin")
        self.assertEqual(pools[0]["max_attempts"], 5)
        self.assertEqual(pools[0]["request_timeout_seconds"], 15.0)
        self.assertEqual(pools[1]["id"], "pool_b")
        self.assertFalse(pools[1]["strict_group"])

    async def test_resolve_dedicated_pools_none_config(self) -> None:
        """Verify no pools are returned when plugin_config is None."""
        from lifecycle import resolve_dedicated_pools

        pools = resolve_dedicated_pools(None)
        self.assertEqual(len(pools), 0)

    async def test_build_canonical_dedicated_configs_multiple(self) -> None:
        """Verify multiple dedicated pool canonical configs are built correctly."""
        from lifecycle import build_canonical_dedicated_configs

        plugin_config = {
            "dedicated_pools": [
                {"id": "pool_a", "group": "group_a", "model_name": "a-router"},
                {"id": "pool_b", "group": "group_b", "model_name": "b-router"},
            ],
        }
        configs = build_canonical_dedicated_configs(plugin_config)
        self.assertEqual(len(configs), 2)
        self.assertEqual(configs[0]["id"], "pool_a")
        self.assertEqual(configs[0]["model"], "a-router")
        self.assertEqual(configs[1]["id"], "pool_b")
        self.assertEqual(configs[1]["model"], "b-router")

    async def test_error_attribution_breakdown(self) -> None:
        """Verify error attribution groups errors by type and node."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "err_metrics.db"
            store = MetricsStore(db_path)

            await store.record_request(
                group_name="t1",
                node_id="node-a",
                model_name="m1",
                is_stream=False,
                total_latency_ms=100.0,
                status="FAILED",
                error_type="RateLimitError",
            )
            await store.record_request(
                group_name="t1",
                node_id="node-a",
                model_name="m1",
                is_stream=False,
                total_latency_ms=100.0,
                status="FAILED",
                error_type="RateLimitError",
            )
            await store.record_request(
                group_name="t1",
                node_id="node-b",
                model_name="m2",
                is_stream=False,
                total_latency_ms=5000.0,
                status="FAILED",
                error_type="TimeoutError",
            )

            err_data = await store.get_error_attribution(window_seconds=3600)
            self.assertEqual(err_data["total_errors"], 3)
            cats = {c["error_type"]: c["count"] for c in err_data["categories"]}
            self.assertEqual(cats["RateLimitError"], 2)
            self.assertEqual(cats["TimeoutError"], 1)
            self.assertEqual(len(err_data["node_rankings"]), 2)
            self.assertEqual(err_data["node_rankings"][0]["node_id"], "node-a")

    async def test_hedging_stats_recording_and_aggregation(self) -> None:
        """Verify hedge racing records and statistics aggregation."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "hedge_metrics.db"
            store = MetricsStore(db_path)

            await store.record_hedge(
                group_name="tier1",
                primary_node="node-slow",
                sibling_node="node-fast",
                winner_node="node-fast",
                sibling_won=True,
                saved_ms=8500.0,
            )
            await store.record_hedge(
                group_name="tier1",
                primary_node="node-slow",
                sibling_node="node-fast",
                winner_node="node-slow",
                sibling_won=False,
                saved_ms=0.0,
            )

            stats = await store.get_hedging_stats(window_seconds=3600)
            self.assertEqual(stats["total_races"], 2)
            self.assertEqual(stats["sibling_won_count"], 1)
            self.assertEqual(stats["overtake_rate"], 50.0)
            self.assertEqual(stats["saved_seconds"], 8.5)
            self.assertEqual(len(stats["recent_events"]), 2)

    async def test_weekly_heatmap_matrix(self) -> None:
        """Verify 7x24 weekly heatmap matrix generates correct shapes and metrics."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "hm_metrics.db"
            store = MetricsStore(db_path)

            await store.record_request(
                group_name="tier1",
                node_id="node-1",
                model_name="flash",
                is_stream=False,
                total_latency_ms=1200.0,
                status="SUCCESS",
            )

            hm_data = await store.get_weekly_heatmap(days=7)
            self.assertEqual(hm_data["total_requests"], 1)
            self.assertEqual(len(hm_data["matrix"]), 7)
            for row in hm_data["matrix"]:
                self.assertEqual(len(row["hours"]), 24)


class RouterTests(unittest.IsolatedAsyncioTestCase):
    async def test_least_busy_strategy_prioritizes_lowest_concurrency(self) -> None:
        """Verify least_busy strategy picks the node with lowest active_requests."""
        cb = CircuitBreaker()
        router = UsageBasedRouter(cb)
        m_busy = PoolMember(provider_id="node-busy", max_parallel_requests=6, weight=1)
        m_idle = PoolMember(provider_id="node-idle", max_parallel_requests=6, weight=1)

        cb.get_state("node-busy").active_requests = 5
        cb.get_state("node-idle").active_requests = 0

        ordered = await router.order_group_candidates([m_busy, m_idle], strategy="least_busy")
        self.assertEqual(ordered[0].provider_id, "node-idle")
        self.assertEqual(ordered[1].provider_id, "node-busy")

    async def test_latency_based_strategy_prioritizes_lowest_latency(self) -> None:
        """Verify latency_based strategy picks the node with lowest EWMA latency."""
        cb = CircuitBreaker()
        router = UsageBasedRouter(cb)
        m_fast = PoolMember(provider_id="node-fast", max_parallel_requests=6, weight=1)
        m_slow = PoolMember(provider_id="node-slow", max_parallel_requests=6, weight=1)

        st_fast = cb.get_state("node-fast")
        st_slow = cb.get_state("node-slow")
        st_fast.ewma_latency_ms = 800.0
        st_slow.ewma_latency_ms = 5000.0

        ordered = await router.order_group_candidates([m_slow, m_fast], strategy="latency_based")
        self.assertEqual(ordered[0].provider_id, "node-fast")
        self.assertEqual(ordered[1].provider_id, "node-slow")

    async def test_fallback_chain_and_alias(self) -> None:
        cb = CircuitBreaker()
        router = UsageBasedRouter(
            cb,
            model_group_alias={
                "deepseek-chat": "gemini-3-flash",
            },
            fallbacks={
                "gemini-3-flash": ["gemini-3-flash-fallback"],
                "gemini-3-flash-fallback": ["deepseek-official"],
            },
        )
        chain = router.get_fallback_chain("deepseek-chat")
        self.assertEqual(
            chain,
            ["gemini-3-flash", "gemini-3-flash-fallback", "deepseek-official"],
        )

    async def test_build_fallback_chain_excludes_dedicated_groups(self) -> None:
        cb = CircuitBreaker()
        router = UsageBasedRouter(
            cb,
            fallbacks={
                "tier1": ["tier2"],
                "tier2": ["tier3"],
                "tier3": ["tier4"],
            },
        )
        group_to_members = {
            "tier1": [PoolMember("n1")],
            "tier2": [PoolMember("n2")],
            "tier3": [PoolMember("n3")],
            "tier4": [PoolMember("n4")],
            "light_llm_pool": [PoolMember("n5")],
        }
        chain = router.build_fallback_chain("tier1", group_to_members)
        self.assertEqual(chain, ["tier1", "tier2", "tier3", "tier4"])
        self.assertNotIn("light_llm_pool", chain)

    async def test_usage_based_ordering_prioritizes_low_load(self) -> None:
        cb = CircuitBreaker()
        router = UsageBasedRouter(cb)
        m1 = PoolMember(provider_id="node-busy", max_parallel_requests=6, weight=1)
        m2 = PoolMember(provider_id="node-idle", max_parallel_requests=6, weight=1)

        cb.get_state("node-busy").active_requests = 5
        cb.get_state("node-idle").active_requests = 0

        ordered = await router.order_group_candidates([m1, m2], strategy="usage-based-routing-v2")
        self.assertEqual(ordered[0].provider_id, "node-idle")
        self.assertEqual(ordered[1].provider_id, "node-busy")

    async def test_usage_based_ordering_prioritizes_low_latency(self) -> None:
        cb = CircuitBreaker()
        router = UsageBasedRouter(cb)
        m_fast = PoolMember(provider_id="node-fast", max_parallel_requests=6, weight=1)
        m_slow = PoolMember(provider_id="node-slow", max_parallel_requests=6, weight=1)

        # Same active requests (0), but node-slow has 10s latency vs 1s latency
        st_fast = cb.get_state("node-fast")
        st_slow = cb.get_state("node-slow")
        st_fast.ewma_latency_ms = 1000.0
        st_slow.ewma_latency_ms = 10000.0
        # Warm samples
        st_fast.recent_call_times.extend([(1.0, True), (2.0, True), (3.0, True)])
        st_slow.recent_call_times.extend([(1.0, True), (2.0, True), (3.0, True)])

        ordered = await router.order_group_candidates([m_slow, m_fast], strategy="usage-based-routing-v2")
        self.assertEqual(ordered[0].provider_id, "node-fast")
        self.assertEqual(ordered[1].provider_id, "node-slow")

    async def test_order_candidates_with_fallback_per_group_cap(self) -> None:
        cb = CircuitBreaker()
        router = UsageBasedRouter(
            cb,
            fallbacks={
                "gemini-3-flash": ["gemini-3-flash-fallback"],
                "gemini-3-flash-fallback": ["deepseek-official"],
            },
        )
        members = [
            # Group 1: 5 members
            PoolMember(provider_id="g1-1", group_name="gemini-3-flash"),
            PoolMember(provider_id="g1-2", group_name="gemini-3-flash"),
            PoolMember(provider_id="g1-3", group_name="gemini-3-flash"),
            PoolMember(provider_id="g1-4", group_name="gemini-3-flash"),
            PoolMember(provider_id="g1-5", group_name="gemini-3-flash"),
            # Group 2: 2 members
            PoolMember(provider_id="g2-1", group_name="gemini-3-flash-fallback"),
            PoolMember(provider_id="g2-2", group_name="gemini-3-flash-fallback"),
            # Group 3: 1 member
            PoolMember(provider_id="g3-1", group_name="deepseek-official"),
        ]

        # max_attempts_per_group = 2 (e.g. LiteLLM num_retries=2)
        candidates = await router.order_candidates_with_fallback(
            members,
            primary_group="gemini-3-flash",
            max_attempts_per_group=2,
        )
        pids = [c.provider_id for c in candidates]

        # First 2 from g1
        self.assertTrue(all(p.startswith("g1-") for p in pids[:2]))
        # Next 2 from g2
        self.assertTrue(all(p.startswith("g2-") for p in pids[2:4]))
        # Next 1 from g3
        self.assertEqual(pids[4], "g3-1")
        # Total candidate count is 2 + 2 + 1 = 5, not all 8
        self.assertEqual(len(candidates), 5)

    async def test_order_candidates_with_fallback_strict_group(self) -> None:
        cb = CircuitBreaker()
        router = UsageBasedRouter(
            cb,
            fallbacks={
                "light_llm_pool": ["fallback-group"],
            },
        )
        members = [
            PoolMember(provider_id="light-1", group_name="light_llm_pool"),
            PoolMember(provider_id="light-2", group_name="light_llm_pool"),
            PoolMember(provider_id="fb-1", group_name="fallback-group"),
            PoolMember(provider_id="heavy-1", group_name="gemini-3-flash"),
        ]
        # When strict_group=True, ONLY light_llm_pool members are returned
        candidates = await router.order_candidates_with_fallback(
            members,
            primary_group="light_llm_pool",
            strict_group=True,
        )
        pids = [c.provider_id for c in candidates]
        self.assertEqual(set(pids), {"light-1", "light-2"})

    async def test_model_request_interval_defers_recently_dispatched_node(self) -> None:
        import time
        cb = CircuitBreaker()
        router = UsageBasedRouter(cb, model_request_interval_seconds=5.0)
        m_a = PoolMember(provider_id="node-a", group_name="grp", max_parallel_requests=6)
        m_b = PoolMember(provider_id="node-b", group_name="grp", max_parallel_requests=6)

        # Pre-mark node-a as dispatched 1 second ago
        router._last_dispatched_at["node-a"] = time.monotonic() - 1.0

        ordered = await router.order_group_candidates([m_a, m_b], strategy="usage-based-routing-v2")
        # node-b is fresh, so it should be prioritized ahead of node-a
        self.assertEqual(ordered[0].provider_id, "node-b")
        self.assertEqual(ordered[1].provider_id, "node-a")

    async def test_model_request_interval_falls_back_when_single_model(self) -> None:
        import time
        cb = CircuitBreaker()
        router = UsageBasedRouter(cb, model_request_interval_seconds=5.0)
        m_single = PoolMember(provider_id="only-node", group_name="grp")

        # Marked as dispatched 0.5s ago
        router._last_dispatched_at["only-node"] = time.monotonic() - 0.5

        # With no other model in group, single node is still returned
        ordered = await router.order_group_candidates([m_single], strategy="usage-based-routing-v2")
        self.assertEqual(len(ordered), 1)
        self.assertEqual(ordered[0].provider_id, "only-node")

    async def test_model_request_interval_rotates_when_all_models_cooling(self) -> None:
        import time
        cb = CircuitBreaker()
        router = UsageBasedRouter(cb, model_request_interval_seconds=10.0)
        m_a = PoolMember(provider_id="node-a", group_name="grp")
        m_b = PoolMember(provider_id="node-b", group_name="grp")

        now = time.monotonic()
        # node-a rested 4 seconds, node-b rested only 1 second
        router._last_dispatched_at["node-a"] = now - 4.0
        router._last_dispatched_at["node-b"] = now - 1.0

        ordered = await router.order_group_candidates([m_a, m_b], strategy="usage-based-routing-v2")
        # node-a rested longer, so it should come before node-b
        self.assertEqual(ordered[0].provider_id, "node-a")
        self.assertEqual(ordered[1].provider_id, "node-b")

    async def test_model_in_both_primary_and_secondary_groups(self) -> None:
        cb = CircuitBreaker()
        router = UsageBasedRouter(cb)
        members = [
            PoolMember(provider_id="heavy-only", group_name="gemini-3-flash"),
            PoolMember(provider_id="shared-node", group_name="gemini-3-flash, light_llm_pool"),
            PoolMember(provider_id="light-only", group_name="light_llm_pool"),
        ]

        # 1. Routing for primary group gemini-3-flash contains shared-node and heavy-only, not light-only
        cands_primary = await router.order_candidates_with_fallback(
            members,
            primary_group="gemini-3-flash",
            strict_group=True,
        )
        pids_primary = [c.provider_id for c in cands_primary]
        self.assertIn("shared-node", pids_primary)
        self.assertIn("heavy-only", pids_primary)
        self.assertNotIn("light-only", pids_primary)

        # 2. Routing for secondary group light_llm_pool contains shared-node and light-only, not heavy-only
        cands_secondary = await router.order_candidates_with_fallback(
            members,
            primary_group="light_llm_pool",
            strict_group=True,
        )
        pids_secondary = [c.provider_id for c in cands_secondary]
        self.assertIn("shared-node", pids_secondary)
        self.assertIn("light-only", pids_secondary)
        self.assertNotIn("heavy-only", pids_secondary)

    async def test_congestion_promotion_promotes_standby_group(self) -> None:
        cb = CircuitBreaker()
        router = UsageBasedRouter(
            cb,
            fallbacks={
                "tier-free": ["tier-paid"],
                "tier-paid": ["tier-agnes"],
            },
        )
        members = [
            PoolMember(provider_id="free-1", group_name="tier-free"),
            PoolMember(provider_id="free-2", group_name="tier-free"),
            PoolMember(provider_id="free-3", group_name="tier-free"),
            PoolMember(provider_id="paid-1", group_name="tier-paid"),
            PoolMember(provider_id="agnes-1", group_name="tier-agnes"),
        ]

        # 1. Normal state: primary is tier-free
        cands_normal = await router.order_candidates_with_fallback(
            members,
            primary_group="tier-free",
            enable_congestion_promotion=True,
        )
        self.assertEqual(cands_normal[0].group_name, "tier-free")

        # 2. Free group becomes congested (e.g. latency exceeds 18s)
        for nid in ("free-1", "free-2", "free-3"):
            st = cb.get_state(nid)
            st.ewma_latency_ms = 20000.0
            st.recent_call_times.extend([(1.0, True), (2.0, True)])

        self.assertTrue(await router.is_group_congested([m for m in members if m.group_name == "tier-free"]))

        # Under congestion of tier-free, tier-paid should be sequentially PROMOTED to the first slot!
        cands_promoted = await router.order_candidates_with_fallback(
            members,
            primary_group="tier-free",
            enable_congestion_promotion=True,
        )
        self.assertEqual(cands_promoted[0].group_name, "tier-paid")
        self.assertEqual(cands_promoted[0].provider_id, "paid-1")

    async def test_congestion_promotion_sequential_standby_promotion_4_tiers(self) -> None:
        """Verify 4-tier chain: gemini-flash -> fallback -> agnes -> ds-official sequentially promotes along fallback rules."""
        cb = CircuitBreaker()
        router = UsageBasedRouter(
            cb,
            fallbacks={
                "gemini-flash": ["gemini-flash-fallback"],
                "gemini-flash-fallback": ["agnes"],
                "agnes": ["ds-official"],
            },
        )
        members = [
            PoolMember(provider_id="gf-1", group_name="gemini-flash"),
            PoolMember(provider_id="gf-2", group_name="gemini-flash"),
            PoolMember(provider_id="gfb-1", group_name="gemini-flash-fallback"),
            PoolMember(provider_id="ag-1", group_name="agnes"),
            PoolMember(provider_id="ds-1", group_name="ds-official"),
        ]

        # 1. Congest only the primary free group
        for nid in ("gf-1", "gf-2"):
            st = cb.get_state(nid)
            st.ewma_latency_ms = 18000.0
            st.recent_call_times.extend([(1.0, True), (2.0, True)])

        # 2. Order candidates under congestion: should sequentially promote next tier 'gemini-flash-fallback'
        cands1 = await router.order_candidates_with_fallback(
            members,
            primary_group="gemini-flash",
            enable_congestion_promotion=True,
        )
        distinct_groups1 = []
        for c in cands1:
            if c.group_name not in distinct_groups1:
                distinct_groups1.append(c.group_name)

        self.assertEqual(
            distinct_groups1,
            ["gemini-flash-fallback", "agnes", "ds-official"],
        )

        # 3. If gemini-flash-fallback is also congested, sequentially promote to 'agnes'
        st_fb = cb.get_state("gfb-1")
        st_fb.ewma_latency_ms = 19000.0
        st_fb.recent_call_times.extend([(1.0, True), (2.0, True)])

        cands2 = await router.order_candidates_with_fallback(
            members,
            primary_group="gemini-flash",
            enable_congestion_promotion=True,
        )
        distinct_groups2 = []
        for c in cands2:
            if c.group_name not in distinct_groups2:
                distinct_groups2.append(c.group_name)

        self.assertEqual(
            distinct_groups2,
            ["agnes", "ds-official"],
        )

    async def test_router_get_and_set_active_group(self) -> None:
        """Verify dynamic routing status reporting, manual active group override, and reset."""
        cb = CircuitBreaker()
        router = UsageBasedRouter(
            cb,
            fallbacks={
                "gemini-3-flash": ["gemini-3-flash-fallback"],
                "gemini-3-flash-fallback": ["agnes"],
                "agnes": ["deepseek-official"],
            },
        )
        members = [
            PoolMember(provider_id="gf-1", group_name="gemini-3-flash"),
            PoolMember(provider_id="ag-1", group_name="agnes"),
            PoolMember(provider_id="ds-1", group_name="deepseek-official"),
        ]

        # 1. Default routing status
        st = await router.get_routing_status(members)
        self.assertEqual(st["default_group"], "gemini-3-flash")
        self.assertEqual(st["active_group"], "gemini-3-flash")
        self.assertFalse(st["is_manual_override"])
        self.assertFalse(st["is_promoted"])

        # 2. Set manual active group to 'agnes'
        await router.set_manual_active_group("agnes")
        st_manual = await router.get_routing_status(members)
        self.assertEqual(st_manual["active_group"], "agnes")
        self.assertTrue(st_manual["is_manual_override"])
        self.assertEqual(st_manual["manual_active_group"], "agnes")
        # active_chain must start with agnes and NOT append upstream gemini-3-flash
        self.assertEqual(st_manual["active_chain"], ["agnes", "deepseek-official"])

        # Candidates order must start with agnes
        cands = await router.order_candidates_with_fallback(members)
        self.assertEqual(cands[0].group_name, "agnes")
        self.assertEqual([c.group_name for c in cands], ["agnes", "deepseek-official"])

        # 3. Reset active group back to auto
        await router.reset_active_group()
        st_reset = await router.get_routing_status(members)
        self.assertEqual(st_reset["active_group"], "gemini-3-flash")
        self.assertFalse(st_reset["is_manual_override"])
        self.assertIsNone(st_reset["manual_active_group"])
        self.assertEqual(
            st_reset["active_chain"],
            ["gemini-3-flash", "gemini-3-flash-fallback", "agnes", "deepseek-official"],
        )

    async def test_reset_active_group_resets_promotion_and_keeps_cooling_nodes(self) -> None:
        """Verify resetting active group clears group promotion while strictly keeping dead nodes locked in circuit."""
        cb = CircuitBreaker()
        router = UsageBasedRouter(
            cb,
            fallbacks={"group-main": ["group-backup"]},
        )
        m_dead = PoolMember(provider_id="node-dead", group_name="group-main")
        m_slow = PoolMember(provider_id="node-slow", group_name="group-main")
        m_backup = PoolMember(provider_id="node-backup", group_name="group-backup")
        members = [m_dead, m_slow, m_backup]

        # 1. node-dead trips open
        await cb.mark_failure("node-dead", "502 Bad Gateway: 无可用渠道")
        self.assertFalse(await cb.is_available("node-dead"))

        # 2. node-slow is healthy but slow (20s EWMA), causing group-main to congest
        st_slow = cb.get_state("node-slow")
        st_slow.ewma_latency_ms = 20000.0
        st_slow.recent_call_times.extend([(1.0, True), (2.0, True)])

        # Verify group is congested and promoted to backup
        status_before = await router.get_routing_status(members)
        self.assertTrue(status_before["is_promoted"])
        self.assertEqual(status_before["active_group"], "group-backup")

        # 3. User clicks 'Reset Active Group to Auto'
        await router.reset_active_group()
        await cb.soft_reset_healthy_qos_metrics()

        # 4. Verify: promotion state is reset to main group!
        status_after = await router.get_routing_status(members)
        self.assertFalse(status_after["is_promoted"])
        self.assertEqual(status_after["active_group"], "group-main")

        # 5. Crucial: dead node MUST remain strictly locked in circuit!
        self.assertFalse(await cb.is_available("node-dead"))
        st_dead = cb.get_state("node-dead")
        self.assertTrue(st_dead.is_open)

        # And node-slow's EWMA is reset to healthy baseline 2500ms
        self.assertEqual(st_slow.ewma_latency_ms, 2500.0)

    async def test_cooling_nodes_never_dispatched_in_candidates(self) -> None:
        """Verify that nodes in circuit open / deep-sleep cooling are NEVER included in candidates."""
        cb = CircuitBreaker()
        router = UsageBasedRouter(
            cb,
            fallbacks={"gemini-flash": ["gemini-flash-fallback"]},
        )
        m_healthy = PoolMember(provider_id="healthy-1", group_name="gemini-flash")
        m_cooling = PoolMember(provider_id="cooling-dead", group_name="gemini-flash")
        m_fallback = PoolMember(provider_id="fallback-1", group_name="gemini-flash-fallback")

        # Trip cooling-dead into open circuit
        await cb.mark_failure("cooling-dead", "502 Bad Gateway: 无可用渠道")
        self.assertFalse(await cb.is_available("cooling-dead"))

        # 1. Group candidates must ONLY contain healthy-1, cooling-dead is completely omitted
        group_cands = await router.order_group_candidates([m_healthy, m_cooling])
        self.assertEqual([m.provider_id for m in group_cands], ["healthy-1"])

        # 2. In full fallback chain, cooling-dead must NOT appear as retry candidate
        all_cands = await router.order_candidates_with_fallback(
            [m_healthy, m_cooling, m_fallback],
            primary_group="gemini-flash",
            max_attempts_per_group=3,
        )
        cand_pids = [c.provider_id for c in all_cands]
        self.assertNotIn("cooling-dead", cand_pids)
        self.assertEqual(cand_pids, ["healthy-1", "fallback-1"])

    async def test_dead_node_never_preempts_busy_healthy_node(self) -> None:
        """Verify that an idle dead node (0% avail, 50s timeout) is NEVER ranked before a busy healthy node."""
        cb = CircuitBreaker()
        router = UsageBasedRouter(cb)

        m_healthy = PoolMember(provider_id="healthy-busy", max_parallel_requests=6, weight=1)
        m_dead = PoolMember(provider_id="dead-idle", max_parallel_requests=6, weight=1)

        st_healthy = cb.get_state("healthy-busy")
        st_healthy.active_requests = 2  # Busy with 2 concurrent group requests
        st_healthy.ewma_latency_ms = 15000.0  # Normal 15s completion time
        st_healthy.recent_call_times.extend([(1.0, True), (2.0, True), (3.0, True)])

        st_dead = cb.get_state("dead-idle")
        st_dead.active_requests = 0  # 0 active requests
        st_dead.ewma_latency_ms = 50000.0  # 50s timeout
        st_dead.recent_call_times.extend([(1.0, False), (2.0, False), (3.0, False)])  # 0% avail

        ordered = await router.order_group_candidates([m_dead, m_healthy], strategy="usage-based-routing-v2")
        self.assertEqual(ordered[0].provider_id, "healthy-busy")
        self.assertEqual(ordered[1].provider_id, "dead-idle")

    async def test_pool_member_effective_timeout(self) -> None:
        """Verify effective timeout and cooldown resolution for default, aggressive mode, and custom overrides."""
        m_def = PoolMember(provider_id="m-def")
        self.assertEqual(m_def.get_effective_timeout(22.0), 22.0)
        self.assertEqual(m_def.get_effective_cooldown(1800.0), 1800.0)

        m_agg = PoolMember(provider_id="m-agg", aggressive_mode=True)
        self.assertEqual(m_agg.get_effective_timeout(22.0), 24.0)
        self.assertEqual(m_agg.get_effective_cooldown(1800.0), 45.0)

        m_custom = PoolMember(provider_id="m-cust", timeout_seconds=6.5, cooldown_seconds=30.0)
        self.assertEqual(m_custom.get_effective_timeout(22.0), 6.5)
        self.assertEqual(m_custom.get_effective_cooldown(1800.0), 30.0)

        # Custom timeout takes precedence over aggressive preset
        m_agg_custom = PoolMember(provider_id="m-both", aggressive_mode=True, timeout_seconds=5.0)
        self.assertEqual(m_agg_custom.get_effective_timeout(22.0), 5.0)

    async def test_aggressive_node_relaxed_cooldown_and_no_deep_sleep(self) -> None:
        """Verify aggressive nodes use relaxed short cooldown (e.g. 45s) and are exempt from deep sleep."""
        cb = CircuitBreaker()
        node = "agg-node-fail"

        # Even with 502/UPSTREAM_DEAD, aggressive node should get short cooldown and NEVER enter deep sleep!
        tripped, category, cd = await cb.mark_failure(
            node,
            "502 Bad Gateway: 无可用渠道",
            custom_cooldown=45.0,
            is_aggressive=True,
        )
        self.assertTrue(tripped)
        st = cb.get_state(node)
        self.assertFalse(st.is_deep_sleep)
        status = (await cb.get_all_status())[node]
        self.assertEqual(status["status"], "OPEN")
        # Cooldown must be short (<= 45.0s), not locking until tomorrow!
        self.assertLessEqual(status["cooldown_remaining_seconds"], 45.0)

    async def test_aggressive_node_tolerates_multiple_failures_before_tripping(self) -> None:
        """Verify aggressive nodes tolerate multiple failures according to aggressive_allowed_fails."""
        cb = CircuitBreaker(aggressive_allowed_fails=3)
        node = "agg-node-tolerate"

        # Attempts 1, 2, 3 should NOT trip
        for attempt in range(1, 4):
            tripped, category, cd = await cb.mark_failure(
                node,
                "Request timed out after 24s",
                custom_cooldown=45.0,
                is_aggressive=True,
            )
            self.assertFalse(tripped, f"Attempt {attempt} should not have tripped")
            self.assertTrue(await cb.is_available(node))

        # Attempt 4 strictly exceeds aggressive_allowed_fails (3) and trips
        tripped4, category4, cd4 = await cb.mark_failure(
            node,
            "Request timed out after 24s",
            custom_cooldown=45.0,
            is_aggressive=True,
        )
        self.assertTrue(tripped4)
        self.assertFalse(await cb.is_available(node))
        st = cb.get_state(node)
        self.assertFalse(st.is_deep_sleep)

    async def test_aggressive_node_priority_boost(self) -> None:
        """Verify that healthy aggressive nodes receive priority score bonus to抢跑 first."""
        from routing import normalize_members
        cb = CircuitBreaker()
        router = UsageBasedRouter(cb)
        raw = [
            {"provider_id": "normal-node", "group_name": "flash", "aggressive_mode": False},
            {"provider_id": "agg-node", "group_name": "flash", "aggressive_mode": True},
        ]
        members = normalize_members(raw)
        # Both nodes are cold/healthy
        ordered = await router.order_group_candidates(members, strategy="usage-based-routing-v2")
        self.assertEqual(ordered[0].provider_id, "agg-node")
        self.assertEqual(ordered[1].provider_id, "normal-node")

    async def test_ratelimit_short_cooldown(self) -> None:
        """Verify 429 errors trip with short cooldown (90s default) instead of 1800s."""
        cb = CircuitBreaker(
            cooldown_time=1800.0,
            ratelimit_cooldown_seconds=90.0,
            allowed_fails_policy={ErrorCategory.RATE_LIMIT.value: 0},
        )
        node = "rl-node"
        tripped, category, cd = await cb.mark_failure(node, "429 Too Many Requests")
        self.assertTrue(tripped)
        self.assertEqual(category, ErrorCategory.RATE_LIMIT)
        status = (await cb.get_all_status())[node]
        # Cooldown remaining should be <= 90s, not 1800s
        self.assertLessEqual(status["cooldown_remaining_seconds"], 90.0)

    async def test_upstream_dead_triggers_deep_sleep_until_0200(self) -> None:
        """Verify 502/UPSTREAM_DEAD triggers deep sleep until next 02:00."""
        cb = CircuitBreaker(daily_reset_hour=2)
        node = "dead-node"
        tripped, category, cd = await cb.mark_failure(node, "502 Bad Gateway: 无可用渠道")
        self.assertTrue(tripped)
        self.assertEqual(category, ErrorCategory.UPSTREAM_DEAD)
        st = cb.get_state(node)
        self.assertTrue(st.is_deep_sleep)
        status = (await cb.get_all_status())[node]
        self.assertEqual(status["status"], "DEEP-SLEEP")

        # When current wall clock exceeds deep_sleep_epoch, is_available wakes it up to HALF-OPEN canary
        st.deep_sleep_epoch = 1.0  # In the past
        is_avail = await cb.is_available(node)
        self.assertTrue(is_avail)
        self.assertTrue(st.half_open)
        self.assertTrue(st.is_canary)
        self.assertFalse(st.is_deep_sleep)


class KeyManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_auth_error_detection(self) -> None:
        self.assertTrue(is_auth_error("401 Unauthorized: Invalid API Key"))
        self.assertTrue(is_auth_error("HTTP 403 Forbidden"))
        self.assertFalse(is_auth_error("429 Too Many Requests"))
        self.assertFalse(is_auth_error("Connection timed out"))

    async def test_key_rotation_and_quarantine(self) -> None:
        km = KeyManager(quarantine_duration=1800.0)
        provider_id = "test-prov-multi"
        keys = ["sk-key1", "sk-key2", "sk-key3"]

        # 1. Round-robin pick works
        k1 = await km.pick_next_key(provider_id, keys)
        k2 = await km.pick_next_key(provider_id, keys)
        k3 = await km.pick_next_key(provider_id, keys)
        self.assertEqual({k1, k2, k3}, set(keys))

        # 2. Key1 fails with 401 Invalid Key -> should be quarantined
        quarantined = await km.mark_key_failure(provider_id, "sk-key1", "401 Unauthorized")
        self.assertTrue(quarantined)

        # 3. Only key2 and key3 should remain valid
        valid_keys = km.get_valid_keys(provider_id, keys)
        self.assertNotIn("sk-key1", valid_keys)
        self.assertEqual(set(valid_keys), {"sk-key2", "sk-key3"})

        # 4. Next picks should only alternate between key2 and key3
        picks = [await km.pick_next_key(provider_id, keys) for _ in range(4)]
        self.assertTrue(all(p in {"sk-key2", "sk-key3"} for p in picks))
        self.assertNotIn("sk-key1", picks)

    async def test_key_fleet_summary(self) -> None:
        km = KeyManager(quarantine_duration=1800.0)
        await km.mark_key_failure("p1", "sk-bad", "401 Unauthorized")
        summary = await km.get_fleet_summary({
            "p1": ["sk-good", "sk-bad"],
            "p2": ["sk-key1"],
        })
        self.assertEqual(summary["total_keys"], 3)
        self.assertEqual(summary["healthy_keys"], 2)
        self.assertEqual(summary["quarantined_keys"], 1)
        self.assertAlmostEqual(summary["health_pct"], 66.7, delta=0.1)
        self.assertEqual(len(summary["nodes"]), 2)

    async def test_vision_unsupported_error_and_image_detection(self) -> None:
        # 1. Error string matching
        err_msg = "Error code: 400 - {'error': {'message': 'Model DeepSeek-V4-Flash does not support image input. Remove the image content or use a vision-capable model.', 'type': 'invalid_request_error'}}"
        self.assertTrue(is_vision_unsupported_error(err_msg))
        self.assertEqual(classify_error(err_msg), ErrorCategory.VISION_UNSUPPORTED)

        # 2. Image detection across diverse parameter shapes
        self.assertTrue(request_contains_image(image_urls=["http://test.com/a.jpg"]))
        self.assertFalse(request_contains_image(image_urls=[]))

        # Dict context containing image_url part
        ctx_with_img = [{"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": "..."}]}]
        self.assertTrue(request_contains_image(contexts=ctx_with_img))

        ctx_text_only = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        self.assertFalse(request_contains_image(contexts=ctx_text_only))

    async def test_get_logs_near_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test_near_metrics.db"
            store = MetricsStore(db_path)

            base_t = 1700000000.0
            await store.record_request(
                group_name="gemini-3-flash",
                node_id="model-target",
                model_name="gemini-3.7-flash",
                is_stream=True,
                ttft_ms=500.0,
                total_latency_ms=1800.0,
                status="SUCCESS",
            )
            # Directly update timestamp in db to base_t
            with store._get_connection() as conn:
                conn.cursor().execute("UPDATE request_metrics SET timestamp = ? WHERE id = 1", (base_t,))
                conn.commit()

            # Query around base_t + 2s
            near_records = await store.get_logs_near_timestamp(target_timestamp=base_t + 2.0, window_seconds=10.0)
            self.assertEqual(len(near_records), 1)
            self.assertEqual(near_records[0]["node_id"], "model-target")
            self.assertAlmostEqual(near_records[0]["time_diff"], 2.0, delta=0.1)

            # Query far outside window
            far_records = await store.get_logs_near_timestamp(target_timestamp=base_t + 100.0, window_seconds=10.0)
            self.assertEqual(len(far_records), 0)

    async def test_mask_model_and_node_rules(self) -> None:
        from vision_utils import mask_model_name, mask_node_id

        # Gemini models masked to hy3 (without suffix)
        self.assertEqual(mask_model_name("piexian-gemini/gemini-3.7-flash", "gemini-3.7-flash"), "hy3")
        self.assertEqual(mask_node_id("piexian-gemini/gemini-3.7-flash", "gemini-3.7-flash"), "hy3")
        self.assertEqual(mask_model_name("some-node", "google/gemini-2.5-flash"), "hy3")

        # GPT models masked to hy4 (without suffix)
        self.assertEqual(mask_model_name("aibz-exp/gpt-5.6-luna", "gpt-5.6-luna"), "hy4")
        self.assertEqual(mask_node_id("aibz-exp/gpt-5.6-luna", "gpt-5.6-luna"), "hy4")
        self.assertEqual(mask_model_name("sub-kkdmx/gpt-5.4-mini", "openai/gpt-5.4-mini"), "hy4")

        # Strip everything before the last slash
        self.assertEqual(mask_node_id("flxapi/deepseek-v4.1-flash", "deepseek-v4.1-flash"), "deepseek-v4.1-flash")
        self.assertEqual(mask_node_id("txvlogvip/agnes-3.0-flash", "agnes-3.0-flash"), "agnes-3.0-flash")
        self.assertEqual(mask_node_id("direct-name", "model"), "direct-name")

        # Other models remain clean
        self.assertEqual(mask_model_name("qwen-node", "qwen-2.5-72b"), "qwen-2.5-72b")


class HedgingTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import sys
        import types
        from enum import Enum

        if "astrbot" not in sys.modules:
            astrbot = types.ModuleType("astrbot")
            astrbot.logger = types.SimpleNamespace(
                info=lambda *a, **k: None,
                warning=lambda *a, **k: None,
                error=lambda *a, **k: None,
            )
            sys.modules["astrbot"] = astrbot

            for name in ("astrbot.core", "astrbot.core.agent", "astrbot.core.provider", "astrbot.core.utils"):
                sys.modules[name] = types.ModuleType(name)

            message = types.ModuleType("astrbot.core.agent.message")
            message.ContentPart = type("ContentPart", (), {})
            message.Message = type("Message", (), {})
            sys.modules["astrbot.core.agent.message"] = message

            tool = types.ModuleType("astrbot.core.agent.tool")
            tool.ToolSet = type("ToolSet", (), {})
            sys.modules["astrbot.core.agent.tool"] = tool

            entities = types.ModuleType("astrbot.core.provider.entities")

            class ProviderType(Enum):
                CHAT_COMPLETION = "chat_completion"

            entities.ProviderType = ProviderType
            entities.LLMResponse = type("LLMResponse", (), {})
            entities.ToolCallsResult = type("ToolCallsResult", (), {})
            sys.modules["astrbot.core.provider.entities"] = entities

            provider_mod = types.ModuleType("astrbot.core.provider.provider")

            class MockProviderBase:
                def __init__(self, provider_config, provider_settings):
                    self.provider_config = provider_config
                    self.provider_settings = provider_settings
                    self.model = ""

                def set_model(self, model):
                    self.model = model

            provider_mod.Provider = MockProviderBase
            sys.modules["astrbot.core.provider.provider"] = provider_mod

            register_mod = types.ModuleType("astrbot.core.provider.register")
            register_mod.provider_cls_map = {}
            register_mod.provider_registry = []
            register_mod.register_provider_adapter = lambda *a, **k: (lambda cls: cls)
            sys.modules["astrbot.core.provider.register"] = register_mod

            err_mod = types.ModuleType("astrbot.core.utils.error_redaction")
            err_mod.safe_error = lambda p, e: f"{p}: {e}" if p else str(e)
            sys.modules["astrbot.core.utils.error_redaction"] = err_mod

    async def test_delayed_hedging_races_and_cancels_slow_node(self) -> None:
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from providers import ProviderLLMPool

        pool_provider = ProviderLLMPool(provider_config={}, provider_settings={})

        # Mock two providers in the SAME group
        member_slow = PoolMember(provider_id="node-slow", group_name="tier1")
        member_fast = PoolMember(provider_id="node-fast", group_name="tier1")

        mock_prov_slow = MagicMock()
        mock_prov_fast = MagicMock()

        slow_cancelled = False

        async def slow_chat(**kwargs):
            nonlocal slow_cancelled
            try:
                await asyncio.sleep(0.6)
                res = MagicMock()
                res.completion_text = "slow response"
                return res
            except asyncio.CancelledError:
                slow_cancelled = True
                raise

        async def fast_chat(**kwargs):
            await asyncio.sleep(0.05)
            res = MagicMock()
            res.completion_text = "fast response"
            return res

        mock_prov_slow.text_chat = AsyncMock(side_effect=slow_chat)
        mock_prov_fast.text_chat = AsyncMock(side_effect=fast_chat)
        mock_prov_slow.get_keys.return_value = []
        mock_prov_fast.get_keys.return_value = []

        # Mock _resolve_candidates to return [member_slow, member_fast]
        pool_provider._resolve_candidates = AsyncMock(
            return_value=(
                [(member_slow, mock_prov_slow), (member_fast, mock_prov_fast)],
                {
                    "failover": True,
                    "request_timeout_seconds": 5,
                    "enable_hedging": True,
                    "hedging_delay_seconds": 0.1,  # After 0.1s, hedge node-fast
                },
            )
        )

        resp = await pool_provider.text_chat("hello")
        self.assertEqual(resp.completion_text, "fast response")
        # Slow node should have been launched, and then cancelled when fast node completed
        self.assertTrue(slow_cancelled)

    async def test_primary_fast_does_not_trigger_hedging(self) -> None:
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from providers import ProviderLLMPool

        pool_provider = ProviderLLMPool(provider_config={}, provider_settings={})

        member_fast = PoolMember(provider_id="node-fast", group_name="tier1")
        member_backup = PoolMember(provider_id="node-backup", group_name="tier1")

        mock_prov_fast = MagicMock()
        mock_prov_backup = MagicMock()

        backup_called = False

        async def fast_chat(**kwargs):
            await asyncio.sleep(0.02)
            res = MagicMock()
            res.completion_text = "primary fast response"
            return res

        async def backup_chat(**kwargs):
            nonlocal backup_called
            backup_called = True
            res = MagicMock()
            res.completion_text = "backup response"
            return res

        mock_prov_fast.text_chat = AsyncMock(side_effect=fast_chat)
        mock_prov_backup.text_chat = AsyncMock(side_effect=backup_chat)
        mock_prov_fast.get_keys.return_value = []
        mock_prov_backup.get_keys.return_value = []

        pool_provider._resolve_candidates = AsyncMock(
            return_value=(
                [(member_fast, mock_prov_fast), (member_backup, mock_prov_backup)],
                {
                    "failover": True,
                    "request_timeout_seconds": 5,
                    "enable_hedging": True,
                    "hedging_delay_seconds": 0.2,  # 0.2s delay > 0.02s fast return
                },
            )
        )

        resp = await pool_provider.text_chat("hello")
        self.assertEqual(resp.completion_text, "primary fast response")
        # Backup should NEVER have been invoked
        self.assertFalse(backup_called)

    async def test_global_deadline_budget_fast_jumps_to_standby_rescue(self) -> None:
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from providers import ProviderLLMPool

        pool_provider = ProviderLLMPool(provider_config={}, provider_settings={})

        m_slow1 = PoolMember(provider_id="node-slow1", group_name="tier-free")
        m_slow2 = PoolMember(provider_id="node-slow2", group_name="tier-paid")
        m_standby = PoolMember(provider_id="node-standby", group_name="tier-agnes")

        mock_prov_slow1 = MagicMock()
        mock_prov_slow2 = MagicMock()
        mock_prov_standby = MagicMock()

        slow2_called = False
        standby_called = False

        async def slow1_chat(**kwargs):
            # Slow 1 fails after 0.25s
            await asyncio.sleep(0.25)
            raise RuntimeError("slow1 timeout/error")

        async def slow2_chat(**kwargs):
            nonlocal slow2_called
            slow2_called = True
            await asyncio.sleep(0.5)
            return MagicMock(completion_text="slow2 response")

        async def standby_chat(**kwargs):
            nonlocal standby_called
            standby_called = True
            res = MagicMock()
            res.completion_text = "standby rescued response"
            return res

        mock_prov_slow1.text_chat = AsyncMock(side_effect=slow1_chat)
        mock_prov_slow2.text_chat = AsyncMock(side_effect=slow2_chat)
        mock_prov_standby.text_chat = AsyncMock(side_effect=standby_chat)

        mock_prov_slow1.get_keys.return_value = []
        mock_prov_slow2.get_keys.return_value = []
        mock_prov_standby.get_keys.return_value = []

        pool_provider._resolve_candidates = AsyncMock(
            return_value=(
                [
                    (m_slow1, mock_prov_slow1),
                    (m_slow2, mock_prov_slow2),
                    (m_standby, mock_prov_standby),
                ],
                {
                    "failover": True,
                    "request_timeout_seconds": 5,
                    "global_timeout_seconds": 7.2,  # Tight budget
                    "enable_hedging": False,
                },
            )
        )

        resp = await pool_provider.text_chat("hello")
        self.assertEqual(resp.completion_text, "standby rescued response")
        # Standby should be invoked, skipping slow2
        self.assertTrue(standby_called)
        self.assertFalse(slow2_called)


if __name__ == "__main__":
    unittest.main()
