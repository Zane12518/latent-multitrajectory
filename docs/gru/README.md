# 自建 GRU：候选复现模块

状态：**已在独立候选分支接入并完成 CPU 验证，尚未公开发布。** v0.1 / v0.2 的代码、资产和结果保持原样。发布范围见[候选清单](publication.md)，逐模型身份见[24 个权重清单](assets.json)。

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

在候选源码根目录执行（发布前不要误用仅含 v0.2 的 main）：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test]'
python -m pytest -q tests/test_gru_portable.py
python examples/gru_smoke.py --out outputs/gru_smoke
```

CPU 安装源适用于 Linux；macOS 用普通 PyPI 的 `torch==2.7.1`。示例使用随机未训练模型与两个人工题，各生成 32 个候选；结果只说明接口可运行，不是 benchmark 成绩。输出中有输入、候选和评分。输出路径已存在时拒绝覆盖。

## 使用候选权重

24 个文件仍在私有候选区，**没有公开下载链接**。确认发布范围后再加入 Release；不要把本说明当作它们已经公开。获得合规副本后，将文件放入 `candidate_weights/`，例如：

```bash
python -m latent_multi.gru.predict --config configs/gru/maze10.json --manifest docs/gru/assets.json --weights candidate_weights/main_maze77_a_multi.pt --rows data/demo_inputs.json --limit 2 --output outputs/gru_multi.json
python -m latent_multi.gru.evaluate --task maze --rows data/demo_inputs.json --predictions outputs/gru_multi.json --allow-subset --output outputs/gru_multi_metrics.json
python -m latent_multi.gru.predict --config configs/gru/maze10.json --manifest docs/gru/assets.json --weights candidate_weights/main_maze77_a_set.pt --rows data/demo_inputs.json --limit 2 --output outputs/gru_set.json
python -m latent_multi.gru.evaluate --task maze --rows data/demo_inputs.json --predictions outputs/gru_set.json --allow-subset --output outputs/gru_set_metrics.json
python -m latent_multi.gru.summarize --metrics outputs/gru_set_metrics.json --baseline outputs/gru_multi_metrics.json --output outputs/gru_pair.json
```

SAT 用 `configs/gru/satlib20.json`、对应 `main_sat*_*.pt` 和[自行转换的 UF20-91 输入](data.md)。不传 `--limit` 会生成全部输入；正式全人口评分不传 `--allow-subset`。推理读取题目，不读取 `solution_bits`、训练路径或最优动作标签；评分单独使用 checker。原固定选答是 token 最大 log 概率平均，Maze 排除墙和终点，并不是 GRAM 分支的新路径置信规则。

每个评价 RNG 独立生成并评分；`summarize --metrics` 可接三份文件，先按题平均随机流。匹配控制用相同三份 RNG 的 `--baseline`，保存新增／丢失／保留的有限候选池集合差。汇总器输出描述均值，不新计算同时区间或把评价流当训练重复。已有正式区间保留在[原始逐组表](../evidence/gru/paired_results.md)。

## 验证层级与恢复

[本次验证](validation.md)区分：新 CPU 推理、与原函数的逐值比较、保存数字数组重算，以及复用的旧验收。每次前向一题；原历史评分微批和 CUDA 随机流并未重放，所以不要求 CPU 输出逐候选等于历史预测。

本模块提供推理、训练目标和评测，**未提供完整 GRU 续训 runner**。24 个 `.pt` 仅有模型张量，没有 Adam、RNG、题序、访问游标。不要把它们传给 GRAM 的 `Session.restore()`。原完整 checkpoint 与 CUDA 连续／恢复对照仍待可访问挂载及独立验收；CPU 参数读回不是 CUDA 续训证明。

项目新增接口代码沿用项目 MIT；历史代码的原 MIT 版权单独保留于 [NOTICE](../../NOTICE)。权重、数据、统计和旧源码发布范围见[权利与待确认清单](publication.md)，不自动套用代码许可证。
