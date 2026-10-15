# Changelog

## 1.0.4

- 新增**实时内容抓包 (Prompt Inspector)**：支持在可视化页面中查看每一次提交给 LLM 的完整 System Prompt、多轮上下文、当前提示词、多模态附件及 LLM 完整返回内容。
  - **极致省资源设计**：采用活跃租约心跳机制，仅在用户打开捕获页面时向内存环形队列（最近 100 条）暂存记录，页面未打开或切走时完全休眠、不捕获长文本，0 额外内存与磁盘占用。
  - 支持一键复制各部分文本、格式化多轮气泡展示、以及完整 Raw JSON 数据导出。
- 新增对外 OpenAI 兼容 HTTP 服务（默认端口 5120），支持将模型池路由与高可用梯队暴露给外部程序（NextChat/Cursor/Open WebUI 等）使用，支持流式 SSE、非流式与 Bearer 密钥认证
- 优化 Provider 发现与导入：支持从 AstrBot 静态配置中自动识别可用对话模型，自动过滤非聊天类型（TTS/STT/Embeddings 等）
- 增强前端模型池配置管理：支持手动输入自定义 Provider ID，已配置成员始终在下拉列表中保留
- 修复专用节点池模板字段丢失问题（保存时补全 `__template_key: "pool"`）
- 优化配置卡片展开与表格展示样式，提升大屏幕及嵌套排版体验
- 更新 metadata 描述与版本号

## 1.0.2

- 修复 `providers.py` logger 导入：`from astrbot` → `from astrbot.api`
