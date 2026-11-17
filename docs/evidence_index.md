# 成果、证据与可复现状态

“有报告”“可下载权重”“保存统计重算”和“干净环境从头重训”是不同状态。

| 研究线 | 内部报告 | 公开报告／统计 | 公开运行代码 | 公开权重／数据 | 本次独立验证 |
|---|---|---|---|---|---|
| GRAM 核心项目 Maze10 | 有 | [六组结果](results.md)、[数字](results.json) | 本仓库 | [v0.1.0](https://github.com/Zane12518/latent-multitrajectory/releases/tag/v0.1.0) | CPU 安装、测试、短训练恢复、权重小批、保存统计回读；非完整重训 |
| GRU SATLIB／Maze10 | 有，终审已定位 | [12 配对](evidence/gru/paired_results.md)、全部控制与 K 曲线如下 | 未发布此宿主 | 未发布此宿主 | 汇总导出／表间一致性；未重新生成逐题输出 |
| GRU 有限参考 | 有 | [报告](limited_references.md)、[全条件数字](evidence/gru/limited_reference.json) | 未发布 | 未发布 | 报告字段回读；没有新增模型调用 |
| Maze12／NQ8／query-only | 有记录与资产线索 | [历史索引](history.md)，非完整公开实证包 | 待核验 | 待核验 | 尚无本次干净环境复现 |

## 可直接读取的 GRU 证据

| 文件 | 范围 |
|---|---|
| [SAT summary](evidence/gru/sat_summary.json)、[Maze summary](evidence/gru/maze_summary.json) | 全部已评分模型绝对值、六组主比较、缺失条件 |
| [SAT comparisons](evidence/gru/sat_all_paired_comparisons.csv)、[Maze comparisons](evidence/gru/maze_all_paired_comparisons.csv) | 所有可评价比较，不只正例；同时／点值区间与保护 |
| [SAT K curves](evidence/gru/sat_all_models_K_curves.csv)、[Maze K curves](evidence/gru/maze_all_models_K_curves.csv) | 全部模型与 K 前缀；不是等时延 |
| [SAT sets](evidence/gru/sat_legal_solution_changes.csv)、[Maze sets](evidence/gru/maze_legal_solution_changes.csv) | 新增／丢失／保留的聚合结果 |
| [SAT family](evidence/gru/sat_paired_family.json)、[Maze family](evidence/gru/maze_paired_family.json) | 比较族、2000 次抽样种子、同时区间系数和缺失 |
| [强控制](evidence/gru/strong_controls_followup.json)、[selection](evidence/gru/selection_followup.json) | 后续普通控制选择依据；与原固定比较族分开 |
| [状态清单](evidence/gru/status_inventory.json)、[审计范围](evidence/gru/audit_scope.json) | 验收、缺失与停止边界；公开导出来源 |

原报告编号和文件名保留用于追踪，但私有路径、组织路由、凭据、平台 Job 明细及内部知识库 URL 未发布。当前 GRU 聚合表不含逐地图预测，无法仅靠它们独立重建 bootstrap；表内一致性检查不能冒称重新验证所有原始预测。

## 运行与验证入口

- [严格恢复／分叉](resume.md)，[完整复现命令](reproduce.md)，[v0.1 与 v0.2](versions.md)。
- [本次验证范围](validation_v0.2.md)；[历史包装回执](validation.json)原样保留，不当作本次新跑结果。
- [权利清单](rights.md)、[发布范围](release_scope.md)。当前只新增代码与报告统计，未重新分发 GRU／历史权重或 SATLIB 原始数据。

## 待发布 GRU 接入

候选分支新增 [GRU 模块与命令](gru/README.md)、[逐模型绑定](gru/assets.json)、[验证层级](gru/validation.md)和[历史资产缺口](gru/history_inventory.md)。公开状态仍按上表；候选分支的代码接入、私有权重小批运行不自动变成已公开资产。
本次读取两任务各 28 个模型的逐题数字数组，重算 1,344 行 K 曲线及 1,200 行集合变化聚合，与现有公开表逐值相同。它不是原始 logits/解 identity 全量重验，也未重抽原同时区间。完整原预测暂不可访问，保留此前已通过的验收，不补造本次通过状态。

第二轮核验及逐项状态见 [GRU 审查 r2](gru/review_r2.md)／[机器回执](gru/review_r2.json)。原 14 项测试已在新建、非 editable 安装环境复核；24 个权重各两图／两候选、深度 8 与 16 的 48 个原函数对照均一致。四个私有数字数组已通过新复算命令；公开统计与候选私有文件分开。CUDA 恢复、完整历史候选重新生成、从头重训与新 GitHub CI 均未完成。
