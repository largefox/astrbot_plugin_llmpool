# Changelog

## 0.2.0

- 修复热安装后 `image-caption-pool` 只写入配置、没有进入运行时的问题
- 将模型池 Provider 改为插件托管，不再暴露可误建重复模型卡的来源模板
- 自动迁移并清理 `image-caption-pool/image-caption-pool` 等旧重复项
- 将虚拟模型名改为 `caption-router`
- 自动迁移群聊图片转述 Provider 引用
- 自动删除模型池成员中的自引用
- 增加生命周期单元测试、配置格式测试和 Python 编译检查
- 固定 GitHub Actions 到完整提交 SHA
- 增加 MIT License 与敏感文件忽略规则

## 0.1.0

- 首个可运行版本
- 支持轮询、加权随机、顺序故障转移、超时和失败冷却
