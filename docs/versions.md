# 版本、资产和校验

## 固定版本

| 入口 | 用途 |
|---|---|
| `v0.1.0` → `904f54b707d9bc16677bb2bc0eb5d7b02083e33a` | 原始公开包装，静态离线 Demo，历史训练／评测源码 |
| `v0.2.0` | 严格恢复、公开 GRU 报告和 CPU CI；科研六组结果与旧权重未替换 |
| `v0.3.0` | 独立GRU推理／目标／评测、24个推理权重和4个逐题数字文件；不含完整GRU恢复包 |
| `main` / Pages | 当前文档与动态保存结果回放；以后可能继续更新 |

```bash
git clone https://github.com/Zane12518/latent-multitrajectory.git
cd latent-multitrajectory
git switch --detach v0.1.0
# 阅读或复现历史固定包；如需新恢复检查与文档：
git switch --detach v0.2.0
```

原 v0.1 Release 的三份现存文件保留：`latent-multitrajectory-assets.tar.gz`、`latent-multitrajectory-v0.1.0.zip`、`demo-replay.html`。旧资产清单中的 SHA-256 字段与旧源码校验工具属于历史快照；后来 main 改为字节数＋CRC32。曾单独提供的 `SHA256SUMS` 已不在 v0.1 Release，不用一个失效下载链接替代它，也不重新上传或改写旧资产。

当前 main 的 `scripts/verify_assets.py` 仍只校验旧包字节数和 CRC32。它用于发现传输损坏，不是密码学身份或来源认证。新训练状态的数据内容指纹用于严格恢复，和 Release 归档的传输校验是不同用途。

## v0.2 新归档

新 Release 仅提供本版源码 ZIP、脱敏证据 ZIP 及这两份**新文件**的 `SHA256SUMS`。不重复打包旧权重／数据，不改变其许可。清单不含私有审计目录、原科研仓库或内部报告路径。

```bash
curl -fLO https://github.com/Zane12518/latent-multitrajectory/releases/download/v0.2.0/latent-multitrajectory-v0.2.0.zip
curl -fLO https://github.com/Zane12518/latent-multitrajectory/releases/download/v0.2.0/latent-multitrajectory-evidence-v0.2.0.zip
curl -fLO https://github.com/Zane12518/latent-multitrajectory/releases/download/v0.2.0/SHA256SUMS
sha256sum -c SHA256SUMS
# macOS 等价命令：shasum -a 256 -c SHA256SUMS
```

SHA-256 仅用于这两份新归档的完整性检查，不是来源认证或权利证明。本轮不为旧大文件重做 SHA 扫描。下载地址以 [v0.2.0 Release](https://github.com/Zane12518/latent-multitrajectory/releases/tag/v0.2.0) 为准。

原 v0.1 静态 HTML 与 Pages 当前动画都只是实验回放。历史静态文件不会被当前动画覆盖。

## v0.3 固定版本

```bash
git switch --detach v0.3.0
python -m pip install -e '.[test]'
```

该版本将包元数据升级到0.3.0，避免与早期候选沿用0.2.0混淆。下载[源码ZIP](https://github.com/Zane12518/latent-multitrajectory/releases/download/v0.3.0/latent-multitrajectory-v0.3.0.zip)、[24权重包](https://github.com/Zane12518/latent-multitrajectory/releases/download/v0.3.0/latent-multitrajectory-gru-inference-v0.3.0.tar.gz)、[4统计文件包](https://github.com/Zane12518/latent-multitrajectory/releases/download/v0.3.0/latent-multitrajectory-gru-statistics-v0.3.0.tar.gz)和[SHA256SUMS](https://github.com/Zane12518/latent-multitrajectory/releases/download/v0.3.0/SHA256SUMS)，使用`sha256sum -c SHA256SUMS`或macOS的`shasum -a 256 -c SHA256SUMS`核对。

新包的SHA-256仅用于完整性；旧包、旧标签和Demo文件不改。GRU完整恢复未验证，推理权重不能作为GRAM Session或完整训练恢复输入。完整下载与运行步骤见[GRU复现](gru/README.md)。
