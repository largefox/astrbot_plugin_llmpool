from __future__ import annotations

import asyncio
import unittest

from constants import (
    POOL_MODEL_NAME,
    POOL_PROVIDER_ID,
    POOL_PROVIDER_TYPE,
)
from lifecycle import (
    migrate_caption_references,
    provider_manager_is_initialized,
    reconcile_default_config,
    sanitize_plugin_members,
    sync_pool_runtime,
)


class FakeInstance:
    def __init__(self, provider_id: str, provider_type: str, model: str) -> None:
        self.provider_config = {
            "id": provider_id,
            "type": provider_type,
            "model": model,
        }
        self._model = model

    def get_model(self) -> str:
        return self._model


class FakeManager:
    def __init__(self, initialized: bool) -> None:
        self._mcp_init_task = object() if initialized else None
        self.inst_map = {}
        self.provider_insts = []
        self.providers_config = []
        self.provider_sources_config = []
        self.resource_lock = asyncio.Lock()
        self.terminated = []
        self.loaded = []

    async def terminate_provider(self, provider_id: str) -> None:
        self.terminated.append(provider_id)
        self.inst_map.pop(provider_id, None)

    async def load_provider(self, config: dict) -> None:
        self.loaded.append(dict(config))
        instance = FakeInstance(config["id"], config["type"], config["model"])
        self.inst_map[config["id"]] = instance
        self.provider_insts.append(instance)


class ConfigLifecycleTests(unittest.TestCase):
    def test_reconcile_creates_single_canonical_provider(self) -> None:
        config = {
            "provider": [{"id": "vision-a", "type": "openai_chat_completion"}],
            "provider_sources": [],
            "provider_ltm_settings": {"image_caption_provider_id": ""},
        }
        result = reconcile_default_config(config)
        self.assertIsNone(result.conflict)
        pool = [
            item
            for item in config["provider"]
            if item.get("type") == POOL_PROVIDER_TYPE
        ]
        self.assertEqual(len(pool), 1)
        self.assertEqual(pool[0]["id"], POOL_PROVIDER_ID)
        self.assertEqual(pool[0]["model"], POOL_MODEL_NAME)

    def test_reconcile_removes_generated_duplicate_and_migrates_reference(self) -> None:
        config = {
            "provider": [
                {
                    "id": POOL_PROVIDER_ID,
                    "type": POOL_PROVIDER_TYPE,
                    "model": POOL_PROVIDER_ID,
                },
                {
                    "id": f"{POOL_PROVIDER_ID}/{POOL_PROVIDER_ID}",
                    "provider_source_id": POOL_PROVIDER_ID,
                    "model": POOL_PROVIDER_ID,
                },
                {"id": "vision-a", "type": "openai_chat_completion"},
            ],
            "provider_sources": [
                {"id": POOL_PROVIDER_ID, "type": POOL_PROVIDER_TYPE}
            ],
            "provider_ltm_settings": {
                "image_caption_provider_id": f"{POOL_PROVIDER_ID}/{POOL_PROVIDER_ID}"
            },
        }
        result = reconcile_default_config(config)
        self.assertEqual(
            result.duplicate_provider_ids,
            (f"{POOL_PROVIDER_ID}/{POOL_PROVIDER_ID}",),
        )
        self.assertEqual(config["provider_ltm_settings"]["image_caption_provider_id"], POOL_PROVIDER_ID)
        self.assertFalse(config["provider_sources"])
        self.assertEqual(
            [p["id"] for p in config["provider"] if p.get("type") == POOL_PROVIDER_TYPE],
            [POOL_PROVIDER_ID],
        )

    def test_foreign_canonical_id_is_not_overwritten(self) -> None:
        config = {
            "provider": [
                {"id": POOL_PROVIDER_ID, "type": "foreign_provider"}
            ]
        }
        result = reconcile_default_config(config)
        self.assertIsNotNone(result.conflict)
        self.assertEqual(config["provider"][0]["type"], "foreign_provider")


    def test_foreign_canonical_source_is_not_mutated(self) -> None:
        config = {
            "provider": [
                {"id": "vision-a", "type": "openai_chat_completion"}
            ],
            "provider_sources": [
                {"id": POOL_PROVIDER_ID, "type": "foreign_source"}
            ],
        }
        snapshot = {
            "provider": [dict(item) for item in config["provider"]],
            "provider_sources": [dict(item) for item in config["provider_sources"]],
        }
        result = reconcile_default_config(config)
        self.assertIsNotNone(result.conflict)
        self.assertEqual(config, snapshot)

    def test_reference_migration_handles_provider_model_suffix(self) -> None:
        config = {
            "provider_ltm_settings": {
                "image_caption_provider_id": f"{POOL_PROVIDER_ID}/legacy-model"
            }
        }
        changed = migrate_caption_references(config, {POOL_PROVIDER_ID})
        self.assertEqual(changed, 1)
        self.assertEqual(
            config["provider_ltm_settings"]["image_caption_provider_id"],
            POOL_PROVIDER_ID,
        )

    def test_sanitize_plugin_members_removes_self_references(self) -> None:
        plugin_config = {
            "members": [
                {"provider_id": "vision-a"},
                {"provider_id": POOL_PROVIDER_ID},
                {"provider_id": f"{POOL_PROVIDER_ID}/{POOL_PROVIDER_ID}"},
            ]
        }
        removed = sanitize_plugin_members(plugin_config, {POOL_PROVIDER_ID})
        self.assertEqual(removed, 2)
        self.assertEqual(plugin_config["members"], [{"provider_id": "vision-a"}])

    def test_provider_manager_initialized_probe(self) -> None:
        self.assertFalse(provider_manager_is_initialized(FakeManager(False)))
        self.assertTrue(provider_manager_is_initialized(FakeManager(True)))

    def test_reconcile_creates_secondary_canonical_provider_when_enabled(self) -> None:
        config = {
            "provider": [{"id": "vision-a", "type": "openai_chat_completion"}],
            "provider_sources": [],
            "provider_ltm_settings": {},
        }
        plugin_config = {
            "dedicated_pools": [
                {"id": "light_llm_pool", "group": "light_llm_pool", "model_name": "light-pool-router"},
            ],
        }
        result = reconcile_default_config(config, plugin_config)
        self.assertIsNone(result.conflict)
        self.assertEqual(len(result.canonical_configs), 2)
        pool_providers = [
            item for item in config["provider"] if item.get("type") == POOL_PROVIDER_TYPE
        ]
        self.assertEqual(len(pool_providers), 2)
        ids = {p["id"] for p in pool_providers}
        self.assertEqual(ids, {POOL_PROVIDER_ID, "light_llm_pool"})

    def test_reconcile_removes_secondary_provider_when_disabled(self) -> None:
        config = {
            "provider": [
                {"id": POOL_PROVIDER_ID, "type": POOL_PROVIDER_TYPE, "model": POOL_MODEL_NAME},
                {"id": "light_llm_pool", "type": POOL_PROVIDER_TYPE, "model": "light-pool-router"},
                {"id": "vision-a", "type": "openai_chat_completion"},
            ],
            "provider_sources": [],
        }
        plugin_config = {
            "dedicated_pools": [],
        }
        result = reconcile_default_config(config, plugin_config)
        self.assertIsNone(result.conflict)
        pool_providers = [
            item for item in config["provider"] if item.get("type") == POOL_PROVIDER_TYPE
        ]
        self.assertEqual(len(pool_providers), 1)
        self.assertEqual(pool_providers[0]["id"], POOL_PROVIDER_ID)
        self.assertIn("light_llm_pool", result.duplicate_provider_ids)

    def test_foreign_canonical_id_rejects_secondary_pool_conflict(self) -> None:
        config = {
            "provider": [
                {"id": "light_llm_pool", "type": "foreign_adapter"},
            ],
            "provider_sources": [],
        }
        plugin_config = {
            "dedicated_pools": [
                {"id": "light_llm_pool", "group": "light_llm_pool", "model_name": "light-pool-router"},
            ],
        }
        result = reconcile_default_config(config, plugin_config)
        self.assertIsNotNone(result.conflict)
        self.assertIn("light_llm_pool", result.conflict)


class RuntimeLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_startup_stages_without_double_loading(self) -> None:
        manager = FakeManager(False)
        providers = []
        sources = []
        canonical = {
            "id": POOL_PROVIDER_ID,
            "type": POOL_PROVIDER_TYPE,
            "model": POOL_MODEL_NAME,
        }
        result = await sync_pool_runtime(
            manager,
            providers=providers,
            provider_sources=sources,
            canonical_config=canonical,
            duplicate_provider_ids=(),
        )
        self.assertEqual(result.state, "staged")
        self.assertFalse(manager.loaded)

    async def test_hot_install_loads_canonical_provider(self) -> None:
        manager = FakeManager(True)
        canonical = {
            "id": POOL_PROVIDER_ID,
            "type": POOL_PROVIDER_TYPE,
            "model": POOL_MODEL_NAME,
        }
        result = await sync_pool_runtime(
            manager,
            providers=[canonical],
            provider_sources=[],
            canonical_config=canonical,
            duplicate_provider_ids=(),
        )
        self.assertEqual(result.state, "loaded")
        self.assertIn(POOL_PROVIDER_ID, manager.inst_map)

    async def test_hot_update_terminates_duplicate_and_reloads_legacy_model(self) -> None:
        manager = FakeManager(True)
        duplicate = f"{POOL_PROVIDER_ID}/{POOL_PROVIDER_ID}"
        manager.inst_map[duplicate] = FakeInstance(
            duplicate, POOL_PROVIDER_TYPE, POOL_PROVIDER_ID
        )
        manager.inst_map[POOL_PROVIDER_ID] = FakeInstance(
            POOL_PROVIDER_ID, POOL_PROVIDER_TYPE, POOL_PROVIDER_ID
        )
        canonical = {
            "id": POOL_PROVIDER_ID,
            "type": POOL_PROVIDER_TYPE,
            "model": POOL_MODEL_NAME,
        }
        result = await sync_pool_runtime(
            manager,
            providers=[canonical],
            provider_sources=[],
            canonical_config=canonical,
            duplicate_provider_ids=(duplicate,),
        )
        self.assertEqual(result.state, "reloaded")
        self.assertIn(duplicate, manager.terminated)
        self.assertEqual(
            manager.inst_map[POOL_PROVIDER_ID].get_model(),
            POOL_MODEL_NAME,
        )

    async def test_sync_pool_runtime_loads_both_canonical_providers(self) -> None:
        manager = FakeManager(True)
        canonical_primary = {
            "id": POOL_PROVIDER_ID,
            "type": POOL_PROVIDER_TYPE,
            "model": POOL_MODEL_NAME,
        }
        canonical_secondary = {
            "id": "light_llm_pool",
            "type": POOL_PROVIDER_TYPE,
            "model": "light-pool-router",
        }
        result = await sync_pool_runtime(
            manager,
            providers=[canonical_primary, canonical_secondary],
            provider_sources=[],
            canonical_configs=[canonical_primary, canonical_secondary],
            duplicate_provider_ids=(),
        )
        self.assertEqual(result.state, "loaded")
        self.assertIn(POOL_PROVIDER_ID, manager.inst_map)
        self.assertIn("light_llm_pool", manager.inst_map)
        self.assertEqual(
            manager.inst_map["light_llm_pool"].get_model(),
            "light-pool-router",
        )


if __name__ == "__main__":
    unittest.main()
