# GRU 数据来源、转换与监督边界

## SATLIB

测试为 [SATLIB UF20-91](https://www.cs.ubc.ca/~hoos/SATLIB/benchm.html)：20 变量、91 子句、1,000 个可满足实例，DIMACS CNF。训练 **3,072**、selection **128** 是项目自生成，不能称官方划分。来源与引用见 [SATLIB 首页](https://www.cs.ubc.ca/~hoos/SATLIB/)：Hoos and Stützle, *SATLIB: An Online Resource for Research on SAT*, SAT 2000, pp. 283–292。

本次未从官方说明确认可将原始实例再分发的许可证。仓库不带 CNF、完整 SAT 数据或训练解池，使用者从官方来源自行获取：

```bash
mkdir -p inputs
curl --fail --location https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf20-91.tar.gz -o inputs/uf20-91.tar.gz
python -m latent_multi.gru.prepare_satlib --archive inputs/uf20-91.tar.gz --output inputs/uf20-91.json
```

这条转换已真实运行：1,000/1,000 题，原始包 323,695 字节、CRC32 `7b4625e9`。转换按源文件名排序，保存 `source_filename`，不枚举满足赋值、不生成训练标签。CRC32 用于检查字节一致性，不是来源认证或许可证明；如果官方更新文件，不静默假定仍为同一版本。

输出 ID 为本次转换局部序号。原项目保存预测的整数 ID 不能仅凭行数直接绑定，必须同时核对源文件名和完整子句内容；当前完整原人口在未挂载存储中，这一步仍未新验收。新生成可使用转换文件自身的 ID，不能将其冒充历史评价随机流重放。

解析检查：完整 `p cnf 20 91`、每句三个不同变量、无零文字／永真子句、变量范围、1000 个唯一文件名；不会解压 tar 中的路径或执行内容。损坏、重复和非完整人口会报错。

项目训练／selection 生成器、参考产生及全量划分尚未整理为公开可复现包。历史训练解池先完整枚举，再供质量项和分配参考选择；最多取 32 或 8 个参考不减少已发生的枚举成本。有限获取的求解预算是另一条实验，不能与参考子集设置混称。

## Maze10

项目派生迷宫，不是官方 Maze 数据集。输入 `x` 长 100：1 可通行、2 墙、3 起点、4 终点；动作 0 上、1 下、2 左、3 右。路径遇越界、墙、重复格判非法。合法路径 identity 为完整经过格点序列；目标还要求长度等于最短距离。离线评分从公开地图重新求最短距离以核对 `optimal`，不参与选答。

训练参考为最短路径集合，不是全部合法路径全集。当前候选模块使用已有公开 `data/demo_inputs.json` 的两个地图完成真实推理；它不等于原 GRU 完整 1,200 图已重新生成或其原始人口绑定全部重验。

v0.1 已公开的项目 Maze10 资产保持原样。将其用于历史 GRU 全量复现前，仍须与 GRU 原人口逐 ID、逐输入和最短性绑定；不能凭任务名相同直接认定。新包不另附完整 Maze 人口或训练参考。

## 预测与评分文件

- 推理 JSON：`task/K/depth/seed`、权重内容标识、逐题 `id/candidates/selector_scores`。不嵌入内部路径或平台信息。
- 评分 JSON：每题 K=1/2/4/8/16/32，U、U_all、V/D、目标 V/D、首条、H、实际选答与解集合。
- 历史 tensor 预测：只用 `weights_only=True` 加载，检查 logits 形状、有限值和保存 argmax；不是任意 pickle 兼容读取。
- 全量评分默认要求输入和预测 ID 完整一致；小批检查必须显式 `--allow-subset`，输出实际题数。

数据、报告统计与权重维持现有非代码许可状态；公开代码 MIT 不自动覆盖它们。
