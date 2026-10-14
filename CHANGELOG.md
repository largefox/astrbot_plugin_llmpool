# Changelog

## 1.0.4

- 优化 Provider 发现与导入：支持从 AstrBot 静态配置中自动识别可用对话模型，自动过滤非聊天类型（TTS/STT/Embeddings 等）
- 增强前端模型池配置管理：支持手动输入自定义 Provider ID，已配置成员始终在下拉列表中保留
- 修复专用节点池模板字段丢失问题（保存时补全 `__template_key: "pool"`）
- 优化配置卡片展开与表格展示样式，提升大屏幕及嵌套排版体验
- 更新 metadata 描述与版本号

## 1.0.2

- 修复 `providers.py` logger 导入：`from astrbot` → `from astrbot.api`
