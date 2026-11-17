# 隐空间多轨迹推理 · Latent Multi-trajectory Reasoning

**在固定候选预算下，学习生成更多不同的有效解，同时衡量质量与实际选答。**

[在线动态回放](https://zane12518.github.io/latent-multitrajectory/demo/) · [快速运行](#快速运行) · [成果与证据](docs/evidence_index.md) · [检查点与版本](docs/versions.md) · [CPU CI](https://github.com/Zane12518/latent-multitrajectory/actions/workflows/cpu.yml)

## 成果与公开状态

| 研究线 | 主结果 | 当前公开内容 |
|---|---|---|
| **GRAM 核心 · 项目 Maze10** | 六组固定终点 Multi／Set；4/6 覆盖同时区间为正，2/6 通过全部三项质量保护 | [完整结果](docs/results.md)、代码、配置、12 个推理权重、6 个续训起点、数据、保存统计和 Demo |
| **自建 GRU · SATLIB / Maze10** | 各 3 个质量起点 × 2 条续训流；Set−Q-multi 的六组覆盖区间均为正，三项质量保护通过 | **新增公开报告与全部逐组统计**：[报告](docs/gru_results.md)、[12 组表](docs/evidence/gru/paired_results.md)；GRU 代码与权重尚未发布 |
| **有限参考 · 自建 GRU** | 完整质量监督下，最多 8 个分配参考仍有增量；有限总监督起点能力不足 | [正例、负例及监督成本](docs/limited_references.md)；不是“整个训练只见 8 个解” |
| **历史覆盖与计算复用** | Maze12、N-Queens8、Frozen Base＋Mixer、incremental query-only | [成果索引与缺口](docs/history.md)；未作为当前可运行复现包发布 |

GRU 的六组描述性平均增量为 SATLIB **+0.394944** 个满足赋值／题、Maze10 **+1.380972** 条最短路径／题。它们与本仓库可运行的 GRAM 适配是不同宿主，不能互相替代证据。公开报告、公开源码、公开权重与独立重训复现的状态分别列于[证据索引](docs/evidence_index.md)。

**本候选分支新增**：[独立 GRU 推理／目标／逐题评测模块](docs/gru/README.md)、[24 个权重身份清单](docs/gru/assets.json)、[数据准备](docs/gru/data.md)与[历史资产清点](docs/gru/history_inventory.md)。代码和新资产尚未公开发布，权重不在 Git 中；v0.1/v0.2、在线 Demo 和六组 GRAM 结果保持不变。[验证与未完成项](docs/gru/validation.md) · [拟发布范围](docs/gru/publication.md)。

## 方法与证据边界

```text
题目 + 随机先验 → 共享 GRAM 递归核心 → K32 候选 → 固定路径置信选答
                          ↑
训练：原生监督 / KL / ACT + DAG 质量 + Multi 或 Set 辅助
评分：离线检查合法性、最短性和路径 identity，不参与选答
```

Multi 对相同完整参考均匀独立监督；Set 用轨迹—参考分数、温度 0.5、12 次 Sinkhorn 归一化和停止梯度的分配权重，组织集合级训练。两臂共用质量项与 0.1 辅助系数；有限迭代不保证行边际精确均衡。详见[方法](docs/method.md)。

**集合分配发生于训练期；推理没有跨轨迹通信、验证器筛选、修复或重采样。** 当前可运行宿主为官方 GRAM 递归核心的项目 Maze10 适配，不是官方预训练 Maze 模型或官方榜单。测试已参与开发；固定模型的地图区间不覆盖训练起点或方法选择的不确定性。全部六组结果与未确认质量门均保留。

## 快速运行

以下为 **v0.2.0 / 当前 main** 的 CPU 路径。历史固定包使用 `v0.1.0`，两者区别见[版本说明](docs/versions.md)。

```bash
git clone https://github.com/Zane12518/latent-multitrajectory.git
cd latent-multitrajectory
git switch --detach v0.2.0
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test]'
python -m pytest -q
python examples/replay.py --out outputs/replay.html
```

上面的 PyTorch CPU 安装源适用于 Linux；macOS 使用 `python -m pip install torch==2.7.1`。实测环境和覆盖范围见[验证记录](docs/validation_v0.2.md)。

打开 `outputs/replay.html`：4 张地图、6 组配对、K=1/4/8/16/32，可播放、暂停、逐候选推进。这是**保存实验结果的动态回放，不是实时推理**；动画也不是递归内部状态或真实计算时间线。默认地图 6800 的 1511 Multi / Set 均为 U=2，合法候选为 15/32、17/32，并非只展示 Set 获胜样例。合法性标记来自离线 checker。

### 小规模真实训练与恢复

```bash
python training/train.py --config configs/cpu_smoke.json --data data/toy.json --arm set --device cpu --out outputs/toy_train
python evaluation/evaluate.py --config configs/cpu_smoke.json --data data/toy.json --checkpoint outputs/toy_train/step00002.pt --device cpu --repeats 1 --limit 2 --out outputs/toy_eval
python training/train.py --config configs/cpu_smoke.json --data data/toy.json --resume outputs/toy_train/step00002.pt --arm set --device cpu --updates 1 --out outputs/toy_resume
```

输出目录必须不存在。4 张合成图和两步训练仅验证工程链路，不代表学会任务；最小示例 U32=0。新检查点默认核对完整输入／参考内容、配置、Adam、RNG、题序及 carry；Multi／Set 分叉必须显式声明。旧 v1 文件需警示性兼容读取，不能补造它们缺失的身份信息。见[严格恢复与分叉](docs/resume.md)。

## 使用已公开权重与数据

[下载 v0.1.0 资产包](https://github.com/Zane12518/latent-multitrajectory/releases/download/v0.1.0/latent-multitrajectory-assets.tar.gz)，在仓库根目录解压得到 `release_assets/`。此历史包保持不变，包含推理权重、续训状态、完整项目 Maze10 数据和保存指标；[清单](docs/checkpoints.json)区分权重与完整状态。

```bash
curl -fL https://github.com/Zane12518/latent-multitrajectory/releases/download/v0.1.0/latent-multitrajectory-assets.tar.gz -o latent-multitrajectory-assets.tar.gz
tar -xzf latent-multitrajectory-assets.tar.gz
python scripts/verify_assets.py release_assets
python evaluation/reproduce_table.py --metrics release_assets/statistics/metrics.npz --out outputs/table_readback.json
python evaluation/evaluate.py --config configs/cpu_smoke.json --data data/demo_inputs.json --checkpoint release_assets/weights/c1301_1511_set.pt --device cpu --repeats 1 --limit 2 --out outputs/checkpoint_cpu
```

CPU 两图历史权重示例：U32=1.0，合法率=0.234375，H=0.5，路径选答=0.5。它不是完整测试成绩，也不要求与 CUDA 随机流逐值相同。完整训练／评测和历史 GPU 核验范围见[复现说明](docs/reproduce.md)。本轮没有重训科研端点。

当前旧资产检查使用字节数与 CRC32；新版本归档的校验说明与原 v0.1 的差异见[版本与校验](docs/versions.md)。校验和不是来源认证。

## 目录与许可

```text
configs/          配方和 CPU 工程样例
src/latent_multi/ GRAM 适配、集合目标、采样、指标、恢复检查
training/         训练与显式恢复／分叉
evaluation/       原生 K32 评分、配对统计、历史数字回读
data/             合成样例与保存候选回放
examples/         离线动态 Demo
tests/            正确性、恢复及公开证据一致性测试
docs/             成果报告、逐组统计、证据索引、资产与版本边界
.github/workflows/公开 CPU CI
third_party/      上游许可与修改说明
```

项目新增代码：[MIT](LICENSE)。GRAM 第三方代码保留原始 [LICENSE / NOTICE](third_party/README.md)。**数据、报告／统计、推理权重和续训状态仅维持现有许可状态，不自动适用代码 MIT，也未另授下游修改／再分发许可。** 详见[资产条款](ASSET_TERMS.md)和[权利清单](docs/rights.md)。SATLIB 原始数据不随本次发布。

目录组织参考 [cq-pretrain](https://github.com/ADIOCLASSMATE/cq-pretrain)，未复制其代码、模型架构或生产训练结构。反馈请使用仓库 Issues。
