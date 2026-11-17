# 自建 GRU：推理与保存统计复现

v0.3提供独立GRU推理、目标函数和评测代码，以及24个推理权重和4个逐题统计文件。v0.1/v0.2和现有Demo保持原样。[发布范围与许可](publication.md) · [24个模型身份](assets.json) · [下载包与校验](release_assets.json)。

## 模型与方法

| 对象 | SATLIB UF20-91 | 项目 Maze10 |
|---|---|---|
| 宿主 | 项目 SAT20 GRU，3,982,786 参数 | 项目共享 GRU Planner，121,412 参数 |
| 输入 | 至多 91 个三文字子句，20 个布尔变量 | 10×10 地图，起点／终点／墙／可通行格 |
| 输出 | 每变量二分类 logits | 每格四方向 logits；从起点回放到终点 |
| 随机性 | 初态与条件共同注入 0.25 高斯噪声，每轮再注入 0.05 噪声 | 初态与条件共同注入 0.15 高斯噪声，每轮再注入 0.03 噪声 |
| 训练／评价深度 | 训练交替 8/16，主评价 16 | 训练交替 8/16，主评价 16 |
| 正式测试人口 | 官方 UF20-91 全部 1,000 题 | 项目自建完整 1,200 图，不是官方 Maze 榜单 |
| 质量起点／续训流 | 74、98、912，各 A/B | 77、96、1117，各 A/B |

这不是官方 GRAM。模型的随机初态、逐轮噪声、GRU 状态更新和截断方式均按留存实现保留，不能借用 GRAM 的先验／后验、ACT 或 KL 描述。没有新增推理期跨轨迹通信。

两任务均为共同质量项 + 0.1 辅助。Multi 对相同完整参考均匀独立监督；Set 在同题 K32×M 评分上使用温度 0.5、12 次 Sinkhorn 和停止梯度权重。SAT 评分为变量 log 概率和；Maze 为完整最短路径上的动作 log 概率和，辅助除以答案／路径长度。它们是给定随机状态下的可微概率代理，不是先验采样加 argmax 的实际答案概率。

SAT 共同质量项访问已完整枚举的训练合法解池；Maze 使用原 cell 最优动作损失与完整最短路 DAG 质量。参考去重后每次最多 32 个，按题目访问次数轮换。有限分配参考不等于有限总监督，见[已公开有限参考结果](../limited_references.md)。

核心 `models.py` / `objective.py` 与 portable_v4 字节相同；`native_metrics.py` 仅改为包内相对导入。新接口增加输入检查、权重绑定、明确的子集评分和逐题输出，不改旧目标或科学结果。

## 安装与无权重 smoke

先获取固定版本，再在源码根目录执行：

```bash
git clone https://github.com/Zane12518/latent-multitrajectory.git
cd latent-multitrajectory
git switch --detach v0.3.0
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test]'
python -m pytest -q tests/test_gru_portable.py
python examples/gru_smoke.py --out outputs/gru_smoke
```

CPU 安装源适用于 Linux；macOS 用普通 PyPI 的 `torch==2.7.1`。示例使用随机未训练模型与两个人工题，各生成 32 个候选；结果只说明接口可运行，不是 benchmark 成绩。输出中有输入、候选和评分。输出路径已存在时拒绝覆盖。

## 下载资产与真实推理

[v0.3 Release](https://github.com/Zane12518/latent-multitrajectory/releases/tag/v0.3.0)提供源码、推理权重和逐题统计三个包。下面下载全部三包并校验；macOS将`sha256sum`替换为`shasum -a 256`。校验用于传输完整性，不是来源认证或下游许可。

```bash
for file in latent-multitrajectory-v0.3.0.zip latent-multitrajectory-gru-inference-v0.3.0.tar.gz latent-multitrajectory-gru-statistics-v0.3.0.tar.gz SHA256SUMS; do
  curl --fail --location "https://github.com/Zane12518/latent-multitrajectory/releases/download/v0.3.0/$file" -o "$file"
done
sha256sum -c SHA256SUMS
tar -xzf latent-multitrajectory-gru-inference-v0.3.0.tar.gz
tar -xzf latent-multitrajectory-gru-statistics-v0.3.0.tar.gz
```

解压得到`candidate_weights/`和`candidate_statistics/`。不要在已有未备份目录上重复解压。24个权重均为模型张量，**不是完整续训状态**。两图小批真实推理：

```bash
python -m latent_multi.gru.predict --config configs/gru/maze10.json --manifest docs/gru/assets.json --weights candidate_weights/main_maze77_a_multi.pt --rows data/demo_inputs.json --limit 2 --output outputs/gru_multi.json
python -m latent_multi.gru.evaluate --task maze --rows data/demo_inputs.json --predictions outputs/gru_multi.json --allow-subset --output outputs/gru_multi_metrics.json
python -m latent_multi.gru.predict --config configs/gru/maze10.json --manifest docs/gru/assets.json --weights candidate_weights/main_maze77_a_set.pt --rows data/demo_inputs.json --limit 2 --output outputs/gru_set.json
python -m latent_multi.gru.evaluate --task maze --rows data/demo_inputs.json --predictions outputs/gru_set.json --allow-subset --output outputs/gru_set_metrics.json
python -m latent_multi.gru.summarize --metrics outputs/gru_set_metrics.json --baseline outputs/gru_multi_metrics.json --output outputs/gru_pair.json
```

SAT 用 `configs/gru/satlib20.json`、对应 `main_sat*_*.pt` 和[自行转换的 UF20-91 输入](data.md)。不传 `--limit` 会生成全部输入；正式全人口评分不传 `--allow-subset`。推理读取题目，不读取 `solution_bits`、训练路径或最优动作标签；评分单独使用 checker。原固定选答是 token 最大 log 概率平均，Maze 排除墙和终点，并不是 GRAM 分支的新路径置信规则。

每个评价 RNG 独立生成并评分；`summarize --metrics` 可接三份文件，先按题平均随机流。匹配控制用相同三份 RNG 的 `--baseline`，保存新增／丢失／保留的有限候选池集合差。汇总器输出描述均值，不新计算同时区间或把评价流当训练重复。已有正式区间保留在[原始逐组表](../evidence/gru/paired_results.md)。

## 保存数字数组复算

上面的统计包解压后，每任务的 `per_rng_metrics.npz`、`per_rng_solution_changes.npz` 分别位于 `candidate_statistics/sat/` 和 `candidate_statistics/maze/`：

```bash
python -m latent_multi.gru.reproduce_statistics --task sat --statistics candidate_statistics/sat --tables docs/evidence/gru --output outputs/sat_statistics.json
python -m latent_multi.gru.reproduce_statistics --task maze --statistics candidate_statistics/maze --tables docs/evidence/gru --output outputs/maze_statistics.json
python -m pytest -q tests/test_gru_statistics.py
```

每任务实际通过 672 行 K 曲线、600 行集合变化，最大数值差为 0。命令核对题目／随机流／K、供给—重复恒等式、集合计数和公开 CSV；损坏计数与非有限表格会报错。它不读取原始预测 identity、不重新生成候选、不重抽区间，也不把汇总 CSV 的完整性检查解释为全量历史预测复现。

本版包版本为 `0.3.0`；正式 `v0.2.0` tag 不包含GRU模块。应记录tag/commit和包来源。发布复用的第二轮验证使用新建venv、构建wheel后非editable安装，见[历史复核记录](review_r2.md)；该记录中的未发布状态对应审查当时，不是本版资产状态。

## 验证层级与恢复

[本次验证](validation.md)区分：新 CPU 推理、与原函数的逐值比较、保存数字数组重算，以及复用的旧验收。每次前向一题；原历史评分微批和 CUDA 随机流并未重放，所以不要求 CPU 输出逐候选等于历史预测。

本模块提供推理、训练目标和评测，**未提供完整 GRU 续训 runner**。24 个 `.pt` 仅有模型张量，没有 Adam、RNG、题序、访问游标。不要把它们传给 GRAM 的 `Session.restore()`。原完整 checkpoint 与 CUDA 连续／恢复对照仍待可访问挂载及独立验收；CPU 参数读回不是 CUDA 续训证明。

项目新增接口代码沿用项目 MIT；历史代码的原 MIT 版权单独保留于 [NOTICE](../../NOTICE)。权重、数据、统计和旧源码发布范围见[发布范围与权利清单](publication.md)，不自动套用代码许可证。
