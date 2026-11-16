# 隐空间多轨迹推理 · Fixed-budget Multi-trajectory Reasoning

**当前公开复现包：v0.1.0，GRAM–Maze10 主线。** 项目新增源码采用 MIT；该主线的检查点、数据与完整续训状态随 Release 提供，具体范围见 [资产说明](ASSET_TERMS.md)。其他研究线的成果与公开状态分别列于下表。

同一道题、固定生成预算，能否在保持答案质量时得到更多不同有效解？项目围绕共享递归、多轨迹覆盖、完整参考集合分配与计算复用开展研究。当前可直接运行的发布包提供官方GRAM递归核心的**项目Maze10适配**、强Multi对照和可复核评测；下面同时列出其他已形成的研究成果，区分结果摘要与已公开复现资产。

## 项目成果总览

| 研究线 | 宿主与任务 | 主要进展 | 公开状态 |
|---|---|---|---|
| GRAM 核心的集合分配 | 官方 GRAM 递归核心的项目 Maze10 适配 | 六组固定终点 Multi／Set 比较；4/6 组覆盖同时区间为正，2/6 组通过全部三项质量保护 | **已公开可复现**：代码、配置、数据、检查点、评测与动态结果回放 |
| GRU 强多参考对照 | 自建递归 GRU；SATLIB UF20-91、项目 Maze10 | 两任务各六组 Set−Q-multi 覆盖同时区间均为正，三项质量保护均通过 | **结果已核实，公开复现包待整理**：本页公开摘要，代码与权重尚未随本仓库发布 |
| 有限分配参考 | 自建递归 GRU；SATLIB、Maze10 | 保留完整质量监督，仅限制分配参考为最多 8 个；两任务各一组匹配比较取得覆盖增量并通过三项保护 | **结果已核实，资产待整理**；有限总监督与有限获取成本另列，不混为同一结论 |
| 历史覆盖与计算复用 | Frozen Base＋Mixer、Maze12、N-Queens8、incremental query-only | 覆盖目标、有效吞吐与粒子状态复用的历史报告和实验资产 | **成果索引，公开复现包待逐项整理**；未包含在 v0.1.0 资产包中 |

### 自建 GRU：集合分配相对强多参考训练的增量

共同质量项、训练参考、初始化与新增训练预算匹配；Q-multi 与 Set 均增加 0.1 辅助项，主要差别是均匀独立多参考监督与轨迹—完整参考软分配。每任务为 **3 个质量起点 × 2 条续训流**，正式评价固定 K32、受训 16 轮出口，每题平均 3 个评价随机流。

| 任务与完整测试人口 | 匹配 Q-multi U32 | Set U32 | 描述性平均 ΔU／题 | 覆盖与质量 |
|---|---:|---:|---:|---|
| SATLIB UF20-91：布尔约束满足，1,000 题 | 2.583667 | 2.978611 | +0.394944 | 6/6 覆盖同时区间为正；6/6 三项保护通过 |
| 项目 Maze10：最短路径规划，1,200 图 | 6.098565 | 7.479537 | +1.380972 | 6/6 覆盖同时区间为正；6/6 三项保护通过 |

SAT 的 U 是不同满足赋值数，Maze 的 U 是不同合法最短路径数。三项保护分别对应候选合法率、首条质量和固定无工具选答，相对匹配 Q-multi 的同时区间满足预登记 −1 个百分点非劣界。上述均值是六组条件的描述性汇总；地图／题目区间条件于固定模型，不能代替跨训练起点的不确定性。

这组证据支持**当前配置下集合分配相对匹配多参考监督的额外价值**。它来自自建 GRU，不是官方 GRAM 成绩，也不证明推理期通信或共享 loop 的独立优势；相对其他普通质量控制仍保留部分质量保护未确认的条件。测试人口已参与方法开发。不同宿主采用各自的质量训练与固定选答规则，绝对分数不直接作为架构优劣对照。

结果来源：`GRAM-ASSIGN-VALUE-20261111-r1` 的完整测试报告与独立终审，以及 `GRAM-ASSIGN-FOLLOWUP-20261112-r1` 的后续核验。内部已有实验矩阵、源码和检查点索引；公开许可、依赖整理、模型绑定及干净环境复现仍待逐项完成，原始报告和复现资产尚未随本仓库发布。

### 有限参考：减少分配参考，不等于减少全部监督

| 设置 | 任务 | Set−Q-multi ΔU／题 | 三项质量保护 |
|---|---|---:|---|
| 完整质量监督＋最多 8 个分配参考 | SATLIB UF20-91，1,000 题 | +0.432333 | 通过 |
| 完整质量监督＋最多 8 个分配参考 | 项目 Maze10，1,200 图 | +0.687778 | 通过 |

两项各为一组匹配配对，来源为 `GRAM-ASSIGN-FOLLOWUP-20261112-r1`。共同质量项仍访问完整合法解／最短路径 DAG，只有辅助分配分支使用较小参考集合；该结果不代表整个训练只使用 8 条解，也不等于节约了最初完整参考池的获取成本。

另有不先完整枚举的有限获取与有限总监督试验，其质量起点尚未建立可用生成能力，未形成稳定集合增量。这些结果与失败边界一并保留，公开资产待整理。

### 历史成果：覆盖、有效吞吐与计算复用

| 成果 | 研究内容 | 复现整理重点 |
|---|---|---|
| Maze12 目标分量对照 | Frozen Base＋Mixer 下的覆盖目标与重复控制；保留 41 个训练种子的历史报告 | 对齐宿主、目标分量、匹配对照及各自预算 |
| N-Queens8 覆盖与有效吞吐 | 正式多副本实验中的不同有效解覆盖与 unique-valid/s | 同时列出合法率、墙钟与有效解收益 |
| incremental query-only | 复用已有粒子状态并追加查询，减少粒子前向调用 | 将前向调用减少与端到端加速分别计量 |

这些条目已有内部报告和资产线索，当前仅作为项目成果索引；代码、权重、数据与许可尚待逐项整理后发布。历史 Frozen Base＋Mixer 结果不写成 Base 端到端联合训练收益，也不与后续 Set 集合分配视为同一机制。

## 已公开主线：GRAM–Maze10

以下方法、六组结果、运行命令与下载入口均对应当前已公开的 GRAM–Maze10 发布包。

```text
题目 + 随机先验 → 共享GRAM递归核心 → K32候选 → 固定路径置信选答
                         ↑
训练侧：原生监督/KL/ACT + DAG质量 + Multi 或 Set辅助
评价侧：独立检查合法性、最短性和路径identity（不参与选答）
```

Set用同题的轨迹—完整参考分数，经过温度0.5、12次Sinkhorn归一化形成停止梯度的软分配；Multi用同一参考池均匀独立监督。两者均增加0.1辅助项。
**集合分配发生在训练期；推理没有跨轨迹通信、修复或验证器筛选。** 先前自建GRU与Frozen Base+Mixer是不同研究线，未混入本结果。

### GRAM–Maze10 结果与范围

完整项目Maze10测试1,200图，K32、16外层先验递归、3评价随机流。固定终点，不按测试选点。U为不同合法最短路径数。

| 质量起点 / 续训流 | 强Multi U | Set U | ΔU | 三项质量保护 |
|---|---:|---:|---:|---|
| 1301 / 1404 | 3.0203 | 3.2903 | +0.2700 | 通过 |
| 1301 / 1405 | 3.1392 | 3.1458 | +0.0067 | 未确认 |
| 1301 / 1511 | 2.8969 | 3.3942 | +0.4972 | 通过 |
| 1501 / 1502 | 2.1664 | 2.3083 | +0.1419 | 未确认 |
| 1501 / 1503 | 2.1953 | 2.3319 | +0.1367 | 未确认 |
| 1501 / 1512 | 2.2856 | 2.3258 | +0.0403 | 未确认 |

4/6组覆盖同时区间为正，2/6组同时通过合法率、首条最短率与路径选答各−1个百分点保护。完整绝对值、区间、负结果和计算口径见[结果](docs/results.md)。
结论是**特定起点和续训流下的收益**，不是普遍稳定成功。两个质量初始化与六条续训流分层报告；三个评价随机流不是训练重复。
测试已用于方法开发，属于测试知情开发证据。不是官方预训练Maze成绩或官方Maze榜单。

## 安装与快速开始

CPU验证环境：Linux x86-64、Python 3.12、PyTorch 2.7.1+cpu。GPU数值核对：H100 80GB、Python3.10、PyTorch2.9.0+cu128、BF16；模型546,882参数。精确依赖见[实测CPU版本](requirements-cpu-tested.txt)。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test]'
python -m pytest -q
python examples/replay.py --out outputs/replay.html
```

打开`outputs/replay.html`即可离线动态回放全部12个端点。可切换4张地图、6组配对与K=1/4/8/16/32，播放／暂停、逐条推进或拖动进度，指标随已完成候选累积。页面明确标为**实验结果回放，不是实时模型推理**；动画不是递归状态或实际计算时间线。
默认地图6800上，1511 Multi/Set均有U=2，合法候选分别15/32和17/32；没有挑选一个“Set必胜”的样例。可在页面切换地图，或用`--map-index 1/2/3`设置初始地图。
所有路径/合法性标识来自离线评价checker；不是部署时验证器选答。

## 真实小规模训练、推理与恢复

以下命令已在干净CPU环境运行。4张合成图只是工程样例；两步训练不代表学会任务，实测最小推理U=0，保留这一输出。

```bash
python training/train.py --config configs/cpu_smoke.json --data data/toy.json --arm set --device cpu --out outputs/toy_train
python evaluation/evaluate.py --config configs/cpu_smoke.json --data data/toy.json --checkpoint outputs/toy_train/step00002.pt --device cpu --repeats 1 --limit 2 --out outputs/toy_eval
python training/train.py --config configs/cpu_smoke.json --data data/toy.json --resume outputs/toy_train/step00002.pt --arm set --device cpu --updates 1 --out outputs/toy_resume
```

所有输出目录必须是新目录；不会覆盖旧产物。恢复包含模型、Adam、RNG、题序、游标、参考访问次数及ACT carry；CPU/CUDA RNG不能冒充同一完整恢复。

## 检查点与完整复现

发布资产：12个推理权重（每个约2.1MiB）、6个完整配对续训起点（每个约6.7MiB）、完整数据和紧凑逐题指标。权重不是第三方预训练基座；Maze参数来自项目新初始化训练。
[下载资产包](https://github.com/Zane12518/latent-multitrajectory/releases/download/v0.1.0/latent-multitrajectory-assets.tar.gz)，解压到仓库根目录。二进制资产不进入Git历史；清单、SHA256、CRC32与字节数见[checkpoint清单](docs/checkpoints.json)、[asset manifest](docs/asset_manifest.json)。SHA256用于内容完整性核验，校验值本身不构成独立来源认证。

资产包解压到`release_assets/`后：

```bash
python scripts/verify_assets.py release_assets
python evaluation/reproduce_table.py --metrics release_assets/statistics/metrics.npz --out outputs/table_readback.json
```

也可在CPU上加载真实训练权重，进行两图推理：

```bash
python evaluation/evaluate.py --config configs/cpu_smoke.json --data data/demo_inputs.json --checkpoint release_assets/weights/c1301_1511_set.pt --device cpu --repeats 1 --limit 2 --out outputs/checkpoint_cpu
```

已测输出：`count=2, U32=1.0, validity=0.234375, H=0.5, selected=0.5`。这是FP32小批运行示例，不是完整测试分数，也不与CUDA随机流逐值等同。

完整GPU续训/评测命令见[复现说明](docs/reproduce.md)。本轮验证了12模型首批输出与历史逐值一致、一次真实更新的完整恢复，以及所有1,200图统计重算；**没有为开源重训1000步或重生成已成功的完整预测**。

## 目录

```text
configs/          历史配方和CPU工程样例
src/latent_multi/ 模型适配、参考分配、采样、指标、选答
training/         训练与完整恢复入口
evaluation/       原生K32生成、配对统计、历史结果复算
scripts/          合成数据和资产校验
data/             小样例、明确标注的结果回放
tests/            正确性、更新与恢复测试
examples/         离线可视化Demo
docs/             方法、完整对照、数据、检查点和验证范围
third_party/      上游版权与修改说明
```

## 权利、引用与联系

官方GRAM核心采用其MIT许可，见[NOTICE](NOTICE)与[第三方说明](third_party/README.md)。项目新增源码采用[MIT](LICENSE)，非代码资产见[资产说明](ASSET_TERMS.md)。
未附作者、联系方式或论文引用。问题反馈请使用本仓库 Issues。目录组织参考[cq-pretrain](https://github.com/ADIOCLASSMATE/cq-pretrain)，没有复制其代码或架构。

## 演示与下载

[动态实验回放](https://zane12518.github.io/latent-multitrajectory/demo/)提供地图切换与逐候选动画，无需后端或模型下载。当前源码可生成相同的离线交互页面；[v0.1.0 离线 Demo](https://github.com/Zane12518/latent-multitrajectory/releases/download/v0.1.0/demo-replay.html)保留原静态版，不覆盖历史发布资产。两者都是保存结果展示，不是在线实时推理。

[Release 文件与 SHA256 校验](https://github.com/Zane12518/latent-multitrajectory/releases/tag/v0.1.0)。
