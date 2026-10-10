from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ImportContractTests(unittest.TestCase):
    def test_plugin_registers_without_dashboard_source_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "plugin"
            package.mkdir()
            for name in (
                "__init__.py",
                "constants.py",
                "lifecycle.py",
                "circuit_breaker.py",
                "key_manager.py",
                "vision_utils.py",
                "metrics_store.py",
                "routing.py",
                "runtime.py",
                "providers.py",
                "main.py",
            ):
                shutil.copy2(ROOT / name, package / name)

            script = textwrap.dedent(
                f"""
                import importlib
                import sys
                import types
                from enum import Enum

                class Logger:
                    def __getattr__(self, name):
                        return lambda *args, **kwargs: None

                logger = Logger()

                astrbot = types.ModuleType("astrbot")
                astrbot.logger = logger
                sys.modules["astrbot"] = astrbot

                api = types.ModuleType("astrbot.api")
                api.__path__ = []
                class AstrBotConfig(dict):
                    def save_config(self, *args, **kwargs):
                        pass
                class Star:
                    def __init__(self, context):
                        self.context = context
                api.AstrBotConfig = AstrBotConfig
                api.logger = logger
                api.star = types.SimpleNamespace(Star=Star, Context=object)
                sys.modules["astrbot.api"] = api

                web = types.ModuleType("astrbot.api.web")
                web.json_response = lambda data, *a, **k: data
                web.error_response = lambda msg, *a, **k: {{"error": msg}}
                web.request = types.SimpleNamespace(query={{}}, json=lambda: {{}})
                sys.modules["astrbot.api.web"] = web

                event_mod = types.ModuleType("astrbot.api.event")
                event_mod.AstrMessageEvent = type("AstrMessageEvent", (), {{}})
                filter_mod = types.SimpleNamespace(
                    command=lambda *a, **k: lambda func: func,
                    on_llm_request=lambda *a, **k: lambda func: func,
                )
                event_mod.filter = filter_mod
                sys.modules["astrbot.api.event"] = event_mod

                prov_mod = types.ModuleType("astrbot.api.provider")
                prov_mod.ProviderRequest = type("ProviderRequest", (), {{}})
                prov_mod.LLMResponse = type("LLMResponse", (), {{}})
                sys.modules["astrbot.api.provider"] = prov_mod

                for name in (
                    "astrbot.core",
                    "astrbot.core.agent",
                    "astrbot.core.message",
                    "astrbot.core.provider",
                    "astrbot.core.utils",
                ):
                    sys.modules[name] = types.ModuleType(name)

                comp_mod = types.ModuleType("astrbot.core.message.components")
                comp_mod.Reply = type("Reply", (), {{}})
                comp_mod.Plain = type("Plain", (), {{}})
                sys.modules["astrbot.core.message.components"] = comp_mod

                res_mod = types.ModuleType("astrbot.core.message.message_event_result")
                res_mod.MessageChain = type("MessageChain", (), {{}})
                sys.modules["astrbot.core.message.message_event_result"] = res_mod

                astrbot_path = types.ModuleType("astrbot.core.utils.astrbot_path")
                astrbot_path.get_astrbot_plugin_data_path = lambda: tmp
                sys.modules["astrbot.core.utils.astrbot_path"] = astrbot_path

                message = types.ModuleType("astrbot.core.agent.message")
                message.ContentPart = type("ContentPart", (), {{}})
                message.Message = type("Message", (), {{}})
                sys.modules[message.__name__] = message

                tool = types.ModuleType("astrbot.core.agent.tool")
                tool.ToolSet = type("ToolSet", (), {{}})
                sys.modules[tool.__name__] = tool

                entities = types.ModuleType("astrbot.core.provider.entities")
                class ProviderType(Enum):
                    CHAT_COMPLETION = "chat_completion"
                entities.ProviderType = ProviderType
                class LLMResponse:
                    def __init__(self, completion_text=""):
                        self.completion_text = completion_text
                entities.LLMResponse = LLMResponse
                entities.ToolCallsResult = type("ToolCallsResult", (), {{}})
                sys.modules[entities.__name__] = entities

                provider_module = types.ModuleType("astrbot.core.provider.provider")
                class Provider:
                    def __init__(self, provider_config, provider_settings):
                        self.provider_config = provider_config
                        self.provider_settings = provider_settings
                        self.model_name = ""
                    def set_model(self, model):
                        self.model_name = model
                    def get_model(self):
                        return self.model_name
                provider_module.Provider = Provider
                sys.modules[provider_module.__name__] = provider_module

                register = types.ModuleType("astrbot.core.provider.register")
                register.provider_cls_map = {{}}
                register.provider_registry = []
                def register_provider_adapter(
                    type_name,
                    desc,
                    provider_type=None,
                    default_config_tmpl=None,
                    provider_display_name=None,
                ):
                    def decorate(cls):
                        metadata = types.SimpleNamespace(
                            cls_type=cls,
                            type=type_name,
                            default_config_tmpl=default_config_tmpl,
                            provider_display_name=provider_display_name,
                        )
                        register.provider_cls_map[type_name] = metadata
                        register.provider_registry.append(metadata)
                        return cls
                    return decorate
                register.register_provider_adapter = register_provider_adapter
                sys.modules[register.__name__] = register

                redaction = types.ModuleType("astrbot.core.utils.error_redaction")
                redaction.safe_error = lambda prefix, error: prefix + str(error)
                sys.modules[redaction.__name__] = redaction

                sys.path.insert(0, {tmp!r})
                importlib.import_module("plugin.main")

                metadata = register.provider_cls_map["llm_pool"]
                assert metadata.default_config_tmpl is None
                provider = metadata.cls_type(
                    {{
                        "id": "llm-pool",
                        "type": "llm_pool",
                        "model": "legacy",
                    }},
                    {{}},
                )
                assert provider.get_model() == "llm-pool-router"

                class MemberProvider(Provider):
                    def __init__(self, response_text):
                        super().__init__({{}}, {{}})
                        self.response_text = response_text
                        self.calls = 0
                    async def text_chat(self, **kwargs):
                        self.calls += 1
                        return LLMResponse(self.response_text)

                first = MemberProvider("   ")
                second = MemberProvider("a detailed image caption")
                light_prov = MemberProvider("lightweight decision result")
                context = types.SimpleNamespace(
                    get_provider_by_id=lambda provider_id: {{
                        "vision-a": first,
                        "vision-b": second,
                        "light-node": light_prov,
                    }}.get(provider_id)
                )
                plugin_config = {{
                    "members": [
                        {{"provider_id": "vision-a", "group_name": "default"}},
                        {{"provider_id": "vision-b", "group_name": "default"}},
                        {{"provider_id": "light-node", "group_name": "light_llm_pool"}},
                    ],
                    "dedicated_pools": [
                        {{"id": "light_llm_pool", "group": "light_llm_pool", "model_name": "light-pool-router"}},
                    ],
                    "strategy": "ordered_failover",
                    "failover": True,
                    "cooldown_seconds": 0,
                }}
                runtime = importlib.import_module("plugin.runtime")
                runtime.bind_runtime(context, plugin_config)
                import asyncio
                response = asyncio.run(provider.text_chat(image_urls=["https://example.test/image.jpg"]))
                assert response.completion_text == "a detailed image caption"
                assert first.calls == 1
                assert second.calls == 1

                # Test secondary pool provider
                sec_provider = metadata.cls_type(
                    {{
                        "id": "light_llm_pool",
                        "type": "llm_pool",
                        "model": "light-pool-router",
                    }},
                    {{}},
                )
                assert sec_provider.get_model() == "light-pool-router"
                sec_models = asyncio.run(sec_provider.get_models())
                assert "light-pool-router" in sec_models
                assert "light_llm_pool" in sec_models

                # Secondary pool text_chat routes to light_node directly
                sec_resp = asyncio.run(sec_provider.text_chat(prompt="is this message safe?"))
                assert sec_resp.completion_text == "lightweight decision result"
                assert light_prov.calls == 1

                # Calling primary pool with model="light-pool-router" also routes to light_node
                router_resp = asyncio.run(provider.text_chat(prompt="check intent", model="light-pool-router"))
                assert router_resp.completion_text == "lightweight decision result"
                assert light_prov.calls == 2
                """
            )
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                result.returncode,
                0,
                msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )


if __name__ == "__main__":
    unittest.main()
