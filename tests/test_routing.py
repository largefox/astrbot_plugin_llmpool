from __future__ import annotations

import random
import unittest

from routing import PoolMember, normalize_members, rotate_members, weighted_random_order


class RoutingTests(unittest.TestCase):
    def test_normalize_members_filters_disabled_empty_and_duplicates(self) -> None:
        raw = [
            {"provider_id": "vision-a", "enabled": True, "weight": 3},
            {"provider_id": "vision-a", "enabled": True, "weight": 9},
            {"provider_id": "", "enabled": True},
            {"provider_id": "vision-b", "enabled": False},
            {"provider_id": "vision-c", "weight": 0},
        ]
        self.assertEqual(
            normalize_members(raw),
            [
                PoolMember("vision-a", 3),
                PoolMember("vision-c", 1),
            ],
        )

    def test_round_robin_rotation(self) -> None:
        members = [
            PoolMember("a"),
            PoolMember("b"),
            PoolMember("c"),
        ]
        self.assertEqual(
            [m.provider_id for m in rotate_members(members, 1)],
            ["b", "c", "a"],
        )
        self.assertEqual(
            [m.provider_id for m in rotate_members(members, 4)],
            ["b", "c", "a"],
        )

    def test_weighted_order_is_without_replacement(self) -> None:
        members = [
            PoolMember("a", 5),
            PoolMember("b", 2),
            PoolMember("c", 1),
        ]
        result = weighted_random_order(members, rng=random.Random(7))
        self.assertEqual(len(result), 3)
        self.assertEqual({m.provider_id for m in result}, {"a", "b", "c"})

    def test_pool_member_multi_group_parsing_and_matching(self) -> None:
        member = PoolMember("shared-model", group_name="gemini-3-flash, light_llm_pool")
        self.assertEqual(member.group_names, ("gemini-3-flash", "light_llm_pool"))
        self.assertTrue(member.matches_group("gemini-3-flash"))
        self.assertTrue(member.matches_group("light_llm_pool"))
        self.assertFalse(member.matches_group("unrelated-group"))

    def test_normalize_members_merges_multi_groups_for_same_provider(self) -> None:
        raw = [
            {"provider_id": "shared-prov", "group_name": "gemini-3-flash", "weight": 2},
            {"provider_id": "shared-prov", "group_name": "light_llm_pool"},
        ]
        members = normalize_members(raw)
        self.assertEqual(len(members), 1)
        self.assertEqual(members[0].provider_id, "shared-prov")
        self.assertEqual(members[0].group_names, ("gemini-3-flash", "light_llm_pool"))
        self.assertEqual(members[0].weight, 2)

    def test_normalize_members_minimal_schema_fields(self) -> None:
        """Members saved with only the 3 simplified schema fields should get defaults."""
        raw = [
            {"provider_id": "prov-a", "group_name": "tier1", "enabled": True},
            {"provider_id": "prov-b", "group_name": "tier2", "enabled": True},
        ]
        members = normalize_members(raw)
        self.assertEqual(len(members), 2)
        self.assertEqual(members[0].provider_id, "prov-a")
        self.assertEqual(members[0].weight, 1)
        self.assertEqual(members[0].max_parallel_requests, 6)
        self.assertEqual(members[0].rpm, 90)
        self.assertTrue(members[0].supports_vision)
        self.assertFalse(members[0].aggressive_mode)
        self.assertEqual(members[0].timeout_seconds, 0.0)
        self.assertEqual(members[0].cooldown_seconds, 0.0)


if __name__ == "__main__":
    unittest.main()
