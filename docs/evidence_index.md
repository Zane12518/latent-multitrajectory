# 成果、证据与可复现状态

“有报告”“可下载权重”“保存统计重算”和“干净环境从头重训”是不同状态。

| 研究线 | 内部报告 | 公开报告／统计 | 公开运行代码 | 公开权重／数据 | 已记录验证（按版本区分） |
|---|---|---|---|---|---|
| GRAM 核心项目 Maze10 | 有 | [六组结果](results.md)、[数字](results.json) | 本仓库 | [v0.1.0](https://github.com/Zane12518/latent-multitrajectory/releases/tag/v0.1.0) | CPU 安装、测试、短训练恢复、权重小批、保存统计回读；非完整重训 |
| GRU SATLIB／Maze10 | 有，终审已定位 | [12 配对](evidence/gru/paired_results.md)、全部控制与 K 曲线如下 | [独立GRU模块](gru/README.md) | [v0.3的24权重与4逐题文件](gru/publication.md)；不附原始数据 | v0.3：17测试、24加载、48小批对照、保存数组复算；v0.4：[H200短程恢复](gru/continuation_candidate.md)；均非完整重训 |
| GRU 有限参考 | 有 | [报告](limited_references.md)、[全条件数字](evidence/gru/limited_reference.json) | 未发布 | 未发布 | 报告字段回读；没有新增模型调用 |
| Maze12／NQ8／query-only | 有，原记录保留 | [3份完整报告](history_candidate/README.md) | [独立社区适配](../legacy/community_mixer/README.md) | v0.4：5权重、2一题样例；非全量历史包 | 4个真实CPU接口；完整历史实验未重跑 |

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

- [严格恢复／分叉](resume.md)，[完整复现命令](reproduce.md)，[版本区别](versions.md)。
- [v0.2当时的验证范围](validation_v0.2.md)与[v0.1历史包装回执](validation.json)原样保留，不作为后续版本重新执行的记录。
- GRU：[v0.3发布前验证](gru/validation.md)、[v0.4 H200短程恢复说明](gru/continuation_candidate.md)及[机器回执](gru/resume_validation_candidate.json)。H200记录验证同一环境下两步连续训练与保存后重启的一致性，不是完整历史训练复现。
- [权利清单](rights.md)、[发布范围](release_scope.md)。v0.3当时新增已批准的GRU代码、24推理权重和4逐题数值文件；该版本不分发SATLIB原始数据或历史Mixer权重。v0.4的历史代表性资产另列于下方。

## v0.3 GRU 接入与历史验证

v0.3发布时提供 [GRU 模块与命令](gru/README.md)、[逐模型绑定](gru/assets.json)、[资产下载清单](gru/release_assets.json)和[历史资产缺口](gru/history_inventory.md)。v0.3仅公开逐项获批的24个推理文件和4个数字文件；完整训练状态和其他历史权重不在该版本中。
v0.3整理阶段读取两任务各 28 个模型的逐题数字数组，重算 1,344 行 K 曲线及 1,200 行集合变化聚合，与现有公开表逐值相同。它不是原始 logits/解 identity 全量重验，也未重抽原同时区间。该轮审查时完整原预测不可访问，因此沿用此前已通过的验收；这不是后续版本重新访问或验收原预测的记录。

v0.3发布前第二轮核验及逐项状态见 [GRU 审查 r2](gru/review_r2.md)／[机器回执](gru/review_r2.json)。原 14 项测试已在新建、非 editable 安装环境复核；24 个权重各两图／两候选、深度 8 与 16 的 48 个原函数对照均一致。该轮四个数字数组通过复算命令，v0.3将其作为独立附件提供。截至v0.3当时，CUDA恢复、完整历史候选重新生成和从头重训均未完成。v0.4随后新增的H200短程恢复证据见下方，不回写旧审查记录，也不等同完整历史训练复现。各版本CI以对应提交的GitHub Actions记录为准。

## v0.4 历史资产与恢复

- [127项审阅范围、5项精简导出及历史正负结果](history_candidate/README.md)：v0.4仅公开已批准的代表性范围。
- [GRU续训与H200两步恢复对照](gru/continuation_candidate.md)：模型/Adam/RNG/题序/后续logits一致；不是完整历史重训。
- 历史v0.3审阅中“未恢复”的记录保持其当时范围；上述新证据独立列示。版本CI以[Actions](https://github.com/Zane12518/latent-multitrajectory/actions/workflows/cpu.yml)实际记录为准；v0.4公开下载与校验入口见[版本说明](versions.md#v04-历史接口与恢复)。
