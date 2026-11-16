# 隐空间多轨迹推理 · Fixed-budget Multi-trajectory Reasoning

**v0.1.0 研究发布。** 项目新增源码采用 MIT；检查点、数据与完整续训状态随 Release 提供，具体范围见 [资产说明](ASSET_TERMS.md)。

同一道题、固定K32预算，能否在保持答案质量时得到更多不同有效解？本仓库提供官方GRAM递归核心的**项目Maze10适配**、训练期完整参考分配、强Multi对照和可复核评测。

```text
题目 + 随机先验 → 共享GRAM递归核心 → K32候选 → 固定路径置信选答
                         ↑
训练侧：原生监督/KL/ACT + DAG质量 + Multi 或 Set辅助
评价侧：独立检查合法性、最短性和路径identity（不参与选答）
```

Set用同题的轨迹—完整参考分数，经过温度0.5、12次Sinkhorn归一化形成停止梯度的软分配；Multi用同一参考池均匀独立监督。两者均增加0.1辅助项。
**集合分配发生在训练期；推理没有跨轨迹通信、修复或验证器筛选。** 先前自建GRU与Frozen Base+Mixer是不同研究线，未混入本结果。

## 结果与范围

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
