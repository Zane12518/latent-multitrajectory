# 历史社区 GRAM / Mixer / query-only：私有候选

本模块来自社区 `ad3002/gram` 实现和项目扩展，不是官方 `ahn-ml/gram-release`，也不是项目自建 GRU。原 MIT 版权见 [LICENSE](LICENSE)，来源与修改范围见 [NOTICE](NOTICE)。权重、数据和报告不自动适用代码 MIT。

模型核心、Mixer、query-only、decoder 保持快照算法；只把 dataset dispatch 限制到 Maze/N-Queens。该模块独立安装，避免与当前 GRAM 核心代码混淆。

## 安装及一题两候选接口

```bash
python3.12 -m venv .venv-history
. .venv-history/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install ./legacy/community_mixer
python legacy/community_mixer/run_smoke.py --case maze12 --base assets/maze12_base.pt --mixer assets/maze12_mixer_s35.pt --data inputs/maze_12x12.tsv --output outputs/maze12.json
python legacy/community_mixer/run_smoke.py --case nq8 --base assets/nq8_base.pt --mixer assets/nq8_mixer_s2.pt --data inputs/nqueens_8x8.tsv --output outputs/nq8.json
python legacy/community_mixer/run_smoke.py --case query_only --base assets/nq8_base.pt --mixer assets/nq8_mixer_s2.pt --data inputs/nqueens_8x8.tsv --output outputs/query_only.json
```

`assets/` 和 `inputs/` 是待审批文件的放置位置，当前没有公开下载链接。macOS 不使用 Linux CPU wheel 索引。安装入口已用上轮已验收干净依赖环境的**隔离副本**验证，本轮从官方源重新下载依赖因超时/持续缓慢未完成，不能写成重新联网安装全通过。

训练标签不进入先验生成；TSV 的 `all_completions_compact` 只用于离线评分。Maze 使用最短路径约束解码 max_paths64，NQ 使用 beam64；这些是工具/约束读出，不是当前主线原生 argmax。query-only smoke 的 seed2 Mixer 复用旧接口见证，并非原端到端合同的 seed1 或 Maze budget-elastic 权重；不据此复现历史 E2E 成绩。

本次只运行一题、两候选的 CPU FP32 接口，不复现历史 BF16、完整测试或吞吐。旧 query-only 16格总门 **NO_GO** 保留；调用减少不能换算成相同比例的时延下降。历史完整结果与发布状态见 [审查说明](../../docs/history_candidate/README.md)。
