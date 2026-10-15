"""OpenAI-compatible HTTP API server for LLM Pool.

Allows external applications (NextChat, ChatBox, Open WebUI, Cursor, etc.)
to access the LLM pool, its multi-tier failover, circuit breaker, hedging,
and KPI metrics via standard OpenAI REST API endpoints:
  - POST /v1/chat/completions (supports streaming & non-streaming)
  - GET  /v1/models
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from aiohttp import web

try:
    from .constants import POOL_MODEL_NAME, POOL_PROVIDER_ID
    from .runtime import get_runtime
except ImportError:
    from constants import POOL_MODEL_NAME, POOL_PROVIDER_ID
    from runtime import get_runtime

logger = logging.getLogger("astrbot_plugin_llmpool.api_server")


def _cors_headers() -> dict[str, str]:
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Authorization, Content-Type, Accept, X-Requested-With",
    }


def _add_cors_headers(response: web.StreamResponse) -> None:
    for k, v in _cors_headers().items():
        if k not in response.headers:
            response.headers[k] = v


class OpenAIToolSetAdapter:
    """Adapts OpenAI API tools/functions into AstrBot's ToolSet interface.

    Allows upstream providers (e.g. AstrBot's ProviderOpenAIOfficial) to recognize
    tools, attach them to the LLM payload, and parse returned tool_calls without error.
    """

    def __init__(
        self,
        raw_tools: list[dict[str, Any]] | None = None,
        raw_functions: list[dict[str, Any]] | None = None,
    ) -> None:
        self.tools: list[dict[str, Any]] = []
        if raw_tools and isinstance(raw_tools, list):
            for t in raw_tools:
                if isinstance(t, dict):
                    self.tools.append(t)
        elif raw_functions and isinstance(raw_functions, list):
            # Legacy OpenAI function_calling format
            for f in raw_functions:
                if isinstance(f, dict):
                    self.tools.append({"type": "function", "function": f})

    def get_func_desc_openai_style(
        self, omit_empty_parameter_field: bool = False
    ) -> list[dict[str, Any]]:
        return self.tools


def _parse_openai_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]], str | None, list[str]]:
    """Parse OpenAI format messages into prompt, history contexts, system prompt, and image urls.

    Carefully preserves tool_calls, tool_call_id, and multi-turn tool message chains.
    """
    system_parts: list[str] = []
    contexts: list[dict[str, Any]] = []
    image_urls: list[str] = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user").strip()
        raw_content = msg.get("content")
        tool_calls = msg.get("tool_calls")
        tool_call_id = msg.get("tool_call_id")
        name = msg.get("name")

        text_content = ""
        if isinstance(raw_content, str):
            text_content = raw_content
        elif isinstance(raw_content, list):
            for part in raw_content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    text_content += str(part.get("text") or "")
                elif ptype == "image_url":
                    iu = part.get("image_url", {})
                    url = iu.get("url") if isinstance(iu, dict) else str(iu)
                    if url:
                        image_urls.append(str(url))

        if role == "system":
            if text_content:
                system_parts.append(text_content)
            continue

        # Construct full normalized message dict
        normalized_msg: dict[str, Any] = {"role": role}
        if raw_content is not None or not tool_calls:
            normalized_msg["content"] = text_content if text_content else (raw_content or "")
        else:
            normalized_msg["content"] = None

        if tool_calls and isinstance(tool_calls, list):
            normalized_msg["tool_calls"] = tool_calls
        if tool_call_id:
            normalized_msg["tool_call_id"] = str(tool_call_id)
        if name:
            normalized_msg["name"] = str(name)

        contexts.append(normalized_msg)

    system_prompt = "\n\n".join(system_parts) if system_parts else None

    # Separate last user message as prompt if it is a pure user text message
    last_user_prompt = ""
    history_contexts: list[dict[str, Any]] = []
    if contexts and contexts[-1].get("role") == "user" and not contexts[-1].get("tool_calls"):
        user_c = contexts[-1].get("content")
        last_user_prompt = user_c if isinstance(user_c, str) else ""
        history_contexts = contexts[:-1]
    else:
        # Last message is tool response or assistant message
        history_contexts = contexts

    return last_user_prompt, history_contexts, system_prompt, image_urls


def _extract_tool_calls_from_response(response: Any) -> list[dict[str, Any]] | None:
    """Extract tool calls from AstrBot LLMResponse object or raw_completion."""
    # 1. Try extracting from raw_completion (OpenAI ChatCompletion structure)
    raw_comp = getattr(response, "raw_completion", None)
    if raw_comp and hasattr(raw_comp, "choices") and raw_comp.choices:
        msg = getattr(raw_comp.choices[0], "message", None)
        if msg and getattr(msg, "tool_calls", None):
            res_tcs = []
            for tc in msg.tool_calls:
                tc_dict = {
                    "id": getattr(tc, "id", None) or f"call_{uuid.uuid4().hex[:8]}",
                    "type": getattr(tc, "type", "function"),
                    "function": {
                        "name": getattr(getattr(tc, "function", None), "name", ""),
                        "arguments": getattr(getattr(tc, "function", None), "arguments", "{}"),
                    },
                }
                res_tcs.append(tc_dict)
            if res_tcs:
                return res_tcs

    # 2. Extract from AstrBot LLMResponse fields (tools_call_name, tools_call_args, tools_call_ids)
    names = getattr(response, "tools_call_name", []) or []
    args_ls = getattr(response, "tools_call_args", []) or []
    ids = getattr(response, "tools_call_ids", []) or []

    if names:
        res_tcs = []
        for i, name in enumerate(names):
            call_id = ids[i] if i < len(ids) and ids[i] else f"call_{uuid.uuid4().hex[:8]}"
            arg = args_ls[i] if i < len(args_ls) else {}
            arg_str = json.dumps(arg, ensure_ascii=False) if isinstance(arg, dict) else str(arg or "{}")
            res_tcs.append({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": str(name),
                    "arguments": arg_str,
                },
            })
        return res_tcs

    return None


def _make_chat_completion(
    request_id: str,
    model: str,
    text: str | None,
    created_time: int,
    prompt_len: int = 0,
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    comp_len = len(text or "")
    message_obj: dict[str, Any] = {
        "role": "assistant",
        "content": text if text is not None else (None if tool_calls else ""),
    }
    finish_reason = "tool_calls" if tool_calls else "stop"
    if tool_calls:
        message_obj["tool_calls"] = tool_calls

    return {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": created_time,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message_obj,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": max(1, prompt_len // 4),
            "completion_tokens": max(1, comp_len // 4) if comp_len else 0,
            "total_tokens": max(1, (prompt_len + comp_len) // 4),
        },
    }


def _make_chat_chunk(
    request_id: str,
    model: str,
    delta_text: str | None,
    created_time: int,
    finish_reason: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    if tool_calls:
        delta["tool_calls"] = tool_calls
    elif delta_text:
        delta["content"] = delta_text
    elif finish_reason is None:
        delta["role"] = "assistant"

    return {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


class APIServer:
    """Manages the lifecycle of an aiohttp web server exposing OpenAI endpoints."""

    def __init__(self, context: Any, api_key: str = "") -> None:
        self.context = context
        self.api_key = api_key.strip()
        self.runner: web.AppRunner | None = None
        self.site: web.TCPSite | None = None
        self.host: str = "0.0.0.0"
        self.port: int = 5120

    def _get_provider_instance(self, requested_model: str | None = None) -> Any:
        manager = getattr(self.context, "provider_manager", None)
        inst_map = getattr(manager, "inst_map", {}) if manager else {}

        target_provider_id = POOL_PROVIDER_ID
        if requested_model:
            try:
                try:
                    from .lifecycle import resolve_dedicated_pools
                except ImportError:
                    from lifecycle import resolve_dedicated_pools
                _, plugin_config = get_runtime()
                for p in resolve_dedicated_pools(plugin_config):
                    if p.get("model_name") == requested_model or p.get("id") == requested_model:
                        target_provider_id = p.get("id")
                        break
            except Exception:
                pass

        prov = inst_map.get(target_provider_id)
        if prov is not None:
            return prov

        # Search for any registered pool provider
        if isinstance(inst_map, dict):
            for p in inst_map.values():
                if getattr(p, "_llm_pool_owned", False):
                    return p

        # Fallback instantiate
        try:
            try:
                from .providers import ProviderLLMPool
            except ImportError:
                from providers import ProviderLLMPool
            return ProviderLLMPool(
                provider_config={
                    "id": target_provider_id,
                    "type": "llm_pool",
                    "model": requested_model or POOL_MODEL_NAME,
                }
            )
        except Exception as exc:
            logger.error("Failed to instantiate fallback ProviderLLMPool: %s", exc)
            raise RuntimeError(f"模型池 Provider 不可用: {exc}") from exc

    def _get_available_models(self) -> list[str]:
        models = [POOL_MODEL_NAME, "llm-pool"]
        try:
            try:
                from .lifecycle import resolve_dedicated_pools
            except ImportError:
                from lifecycle import resolve_dedicated_pools
            _, plugin_config = get_runtime()
            for p in resolve_dedicated_pools(plugin_config):
                mname = p.get("model_name")
                if mname and mname not in models:
                    models.append(mname)
        except Exception:
            pass
        return models

    @web.middleware
    async def middleware(self, request: web.Request, handler: Any) -> web.StreamResponse:
        # 1. Handle CORS preflight
        if request.method == "OPTIONS":
            resp = web.Response(status=204)
            _add_cors_headers(resp)
            return resp

        # 2. Authentication check (skip for health probes)
        if self.api_key and request.path not in ("/", "/health", "/v1/health"):
            auth = request.headers.get("Authorization", "").strip()
            bearer_prefix = "Bearer "
            token = auth[len(bearer_prefix):].strip() if auth.startswith(bearer_prefix) else auth
            if token != self.api_key:
                err_body = {
                    "error": {
                        "message": "Incorrect or missing API key in Authorization header.",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_api_key",
                    }
                }
                resp = web.json_response(err_body, status=401)
                _add_cors_headers(resp)
                return resp

        try:
            response = await handler(request)
        except web.HTTPException as ex:
            _add_cors_headers(ex)
            raise
        except Exception as exc:
            logger.exception("Error handling %s: %s", request.path, exc)
            err_body = {
                "error": {
                    "message": str(exc),
                    "type": "api_error",
                    "param": None,
                    "code": "internal_error",
                }
            }
            resp = web.json_response(err_body, status=500)
            _add_cors_headers(resp)
            return resp

        _add_cors_headers(response)
        return response

    async def handle_models(self, request: web.Request) -> web.Response:
        models = self._get_available_models()
        data = [
            {
                "id": m,
                "object": "model",
                "created": 1700000000,
                "owned_by": "llm-pool",
                "permission": [],
                "root": m,
                "parent": None,
            }
            for m in models
        ]
        return web.json_response({"object": "list", "data": data})

    async def handle_chat_completions(self, request: web.Request) -> web.StreamResponse:
        try:
            body = await request.json()
        except Exception:
            return web.json_response(
                {
                    "error": {
                        "message": "Invalid JSON payload in request body",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "bad_json",
                    }
                },
                status=400,
            )

        if not isinstance(body, dict):
            return web.json_response(
                {
                    "error": {
                        "message": "Request body must be a JSON object",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "bad_request",
                    }
                },
                status=400,
            )

        messages = body.get("messages", [])
        if not isinstance(messages, list) or not messages:
            return web.json_response(
                {
                    "error": {
                        "message": "'messages' array is required and must not be empty",
                        "type": "invalid_request_error",
                        "param": "messages",
                        "code": "missing_required_parameter",
                    }
                },
                status=400,
            )

        requested_model = str(body.get("model") or POOL_MODEL_NAME).strip()
        stream = bool(body.get("stream", False))
        req_id = uuid.uuid4().hex[:12]
        created_time = int(time.time())

        # Extract tools & tool_choice
        tools_payload = body.get("tools")
        functions_payload = body.get("functions")
        tool_choice = body.get("tool_choice", "auto")
        func_tool = (
            OpenAIToolSetAdapter(raw_tools=tools_payload, raw_functions=functions_payload)
            if (tools_payload or functions_payload)
            else None
        )

        prompt, history_contexts, system_prompt, image_urls = _parse_openai_messages(messages)
        provider = self._get_provider_instance(requested_model)

        # 1. Non-streaming completion
        if not stream:
            try:
                response = await provider.text_chat(
                    prompt=prompt,
                    contexts=history_contexts,
                    system_prompt=system_prompt,
                    image_urls=image_urls or None,
                    model=requested_model,
                    func_tool=func_tool,
                    tool_choice=tool_choice,
                )
                comp_text = None
                for attr in ("completion_text", "text"):
                    val = getattr(response, attr, None)
                    if isinstance(val, str) and val:
                        comp_text = val
                        break
                tool_calls = _extract_tool_calls_from_response(response)
            except Exception as exc:
                logger.error("text_chat failed: %s", exc)
                return web.json_response(
                    {
                        "error": {
                            "message": str(exc),
                            "type": "api_error",
                            "param": None,
                            "code": "provider_error",
                        }
                    },
                    status=502,
                )

            data = _make_chat_completion(
                request_id=req_id,
                model=requested_model,
                text=comp_text if comp_text or not tool_calls else None,
                created_time=created_time,
                prompt_len=len(prompt),
                tool_calls=tool_calls,
            )
            return web.json_response(data)

        # 2. Streaming completion (SSE)
        sse_resp = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
        _add_cors_headers(sse_resp)
        await sse_resp.prepare(request)

        # Send initial chunk
        init_chunk = _make_chat_chunk(req_id, requested_model, None, created_time)
        await sse_resp.write(f"data: {json.dumps(init_chunk, ensure_ascii=False)}\n\n".encode("utf-8"))

        stream_tool_calls = None
        try:
            gen = provider.text_chat_stream(
                prompt=prompt,
                contexts=history_contexts,
                system_prompt=system_prompt,
                image_urls=image_urls or None,
                model=requested_model,
                func_tool=func_tool,
                tool_choice=tool_choice,
            )
            async for chunk in gen:
                # Check for tool_calls in stream chunks
                chunk_tool_calls = _extract_tool_calls_from_response(chunk)
                if chunk_tool_calls:
                    stream_tool_calls = chunk_tool_calls
                    c_data = _make_chat_chunk(
                        req_id,
                        requested_model,
                        None,
                        created_time,
                        tool_calls=chunk_tool_calls,
                    )
                    await sse_resp.write(f"data: {json.dumps(c_data, ensure_ascii=False)}\n\n".encode("utf-8"))

                chunk_text = None
                for attr in ("completion_text", "text"):
                    val = getattr(chunk, attr, None)
                    if isinstance(val, str) and val:
                        chunk_text = val
                        break
                if chunk_text is None and isinstance(chunk, str) and chunk:
                    chunk_text = chunk

                if chunk_text:
                    c_data = _make_chat_chunk(req_id, requested_model, chunk_text, created_time)
                    await sse_resp.write(f"data: {json.dumps(c_data, ensure_ascii=False)}\n\n".encode("utf-8"))

            # Send terminal chunk
            finish_reason = "tool_calls" if stream_tool_calls else "stop"
            stop_chunk = _make_chat_chunk(req_id, requested_model, None, created_time, finish_reason=finish_reason)
            await sse_resp.write(f"data: {json.dumps(stop_chunk, ensure_ascii=False)}\n\n".encode("utf-8"))
            await sse_resp.write(b"data: [DONE]\n\n")
        except Exception as exc:
            logger.error("text_chat_stream failed: %s", exc)
            err_chunk = {
                "error": {
                    "message": str(exc),
                    "type": "api_error",
                    "code": "stream_error",
                }
            }
            try:
                await sse_resp.write(f"data: {json.dumps(err_chunk, ensure_ascii=False)}\n\n".encode("utf-8"))
            except Exception:
                pass
        finally:
            await sse_resp.write_eof()

        return sse_resp

    async def handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({
            "status": "ok",
            "service": "llm-pool-openai-api",
            "models": self._get_available_models(),
        })

    def create_app(self) -> web.Application:
        app = web.Application(middlewares=[self.middleware])
        # Register routes
        app.router.add_get("/", self.handle_health)
        app.router.add_get("/health", self.handle_health)
        app.router.add_get("/v1/health", self.handle_health)
        app.router.add_get("/v1/models", self.handle_models)
        app.router.add_get("/models", self.handle_models)
        app.router.add_post("/v1/chat/completions", self.handle_chat_completions)
        app.router.add_post("/chat/completions", self.handle_chat_completions)

        # CORS preflight options routes
        for path in ("/", "/health", "/v1/health", "/v1/models", "/models", "/v1/chat/completions", "/chat/completions"):
            app.router.add_options(path, self._handle_options)

        return app

    async def _handle_options(self, request: web.Request) -> web.Response:
        resp = web.Response(status=204)
        _add_cors_headers(resp)
        return resp

    async def start(self, host: str = "0.0.0.0", port: int = 5120) -> None:
        await self.stop()
        self.host = host
        self.port = port
        app = self.create_app()
        self.runner = web.AppRunner(app, access_log=None)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, host=self.host, port=self.port)
        await self.site.start()
        logger.info("LLM Pool OpenAI-compatible API server started at http://%s:%d/v1", self.host, self.port)

    async def stop(self) -> None:
        if self.site:
            try:
                await self.site.stop()
            except Exception as exc:
                logger.debug("Error stopping TCPSite: %s", exc)
            self.site = None

        if self.runner:
            try:
                await self.runner.cleanup()
            except Exception as exc:
                logger.debug("Error cleaning up AppRunner: %s", exc)
            self.runner = None
