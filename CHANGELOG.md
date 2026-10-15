# Changelog

## 1.0.4

- **完整支持外部程序工具调用 (Tool / Function Calling)**：
  - 新增 `OpenAIToolSetAdapter` 适配器，将客户端请求的 `tools` 与 `functions` 规范映射为 AstrBot `ToolSet` 协议，解决 AstrBot 内部因 `tools is None` 忽略上游模型返回工具调用的问题；
  - 完整保留 `messages` 历史上下文中 `assistant` 的 `tool_calls` 与 `tool` 角色的 `tool_call_id`，修复多轮工具调用时上游报错 400 Bad Request 的问题；
  - 规范响应格式：当模型决定调用工具时，正确返回标准的 `tool_calls` 结构并设置 `finish_reason: "tool_calls"`（同时兼容非流式与流式 SSE 分块）；
  - 实时抓包模块同步支持工具调用捕获与函数参数高亮展示。
- 新增**实时内容抓包 (Prompt Inspector)**：支持在可视化页面中查看每一次提交给 LLM 的完整 System Prompt、多轮上下文、当前提示词、多模态附件及 LLM 完整返回内容。
- 新增对外 OpenAI 兼容 HTTP 服务（默认端口 5120），支持将模型池路由与高可用梯队暴露给外部程序（NextChat/Cursor/Open WebUI 等）使用，支持流式 SSE、非流式与 Bearer 密钥认证
- 优化 Provider 发现与导入：支持从 AstrBot 静态配置中自动识别可用对话模型，自动过滤非聊天类型（TTS/STT/Embeddings 等）
- 增强前端模型池配置管理：支持手动输入自定义 Provider ID，已配置成员始终在下拉列表中保留
- 修复专用节点池模板字段丢失问题（保存时补全 `__template_key: "pool"`）
- 优化配置卡片展开与表格展示样式，提升大屏幕及嵌套排版体验
- 更新 metadata 描述与版本号

## 1.0.2

- 修复 `providers.py` logger 导入：`from astrbot` → `from astrbot.api`
