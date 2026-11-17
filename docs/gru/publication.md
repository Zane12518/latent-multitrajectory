# v0.3 GRU 发布范围与权利

权利持有人已确认有权公开本次精确清单，并批准发布：30个候选源码／配置／测试／说明增量及必要版本文档更新、24个GRU推理权重、4个逐题统计文件。v0.1/v0.2、六组GRAM结果和现有Demo保持不变。

| 对象 | 来源与公开范围 | 许可与验证 |
|---|---|---|
| 项目新增接口、配置、测试 | 独立 `src/latent_multi/gru/`、配置、评测与smoke | 项目代码MIT；与官方GRAM分开 |
| 历史提取模型、目标、指标 | 已审阅portable_v4及原函数归档 | 原MIT、版权和NOTICE保留；不冒称官方GRAM |
| [24个推理权重](assets.json) | 项目自建GRU：SATLIB UF20-91与Maze10各12个权重，源合同和端点逐项列明 | 已确认公开权限；tensor-only，24加载和48小批对照通过；无Adam/RNG/题序 |
| [4个逐题文件](release_assets.json) | 两任务各metrics与solution_changes；数值与计数，不含原题面 | 已确认公开权限；1344行K曲线及1200行集合变化复算一致 |
| SATLIB原始数据、项目训练／selection人口 | 不附入本包 | [官方获取与转换](data.md)；3072训练和128selection为项目生成 |
| 原完整GRU断点、原始全量logits | 不发布 | 当前完整原件访问和CUDA恢复仍未验证 |
| Maze12/NQ8/query-only历史源码与127权重 | 只保留成果索引与缺口 | 不纳入本次发布，旧负结果保留 |

项目新增代码MIT不自动覆盖权重、数据或统计。非代码资产**维持现有许可状态，不另授下游修改、训练再利用或再分发许可**，见[资产条款](../../ASSET_TERMS.md)。历史提取代码的[完整MIT](../../third_party/HISTORICAL-GRAM-LICENSE.txt)与[官方GRAM许可](../../third_party/GRAM-LICENSE.txt)分别保留。

下载见[v0.3 Release](https://github.com/Zane12518/latent-multitrajectory/releases/tag/v0.3.0)。[资产清单](release_assets.json)列出归档大小与SHA-256，Release的`SHA256SUMS`覆盖源码、权重和统计三包；逐文件CRC32沿用已验收绑定。校验值不是来源认证或许可证明。

历史[审查回执](review_r2.md)保留原状，包括当时的私有状态、失败及CUDA缺项。本次确认改变的是公开范围，不把旧验证改写为新的科研实验或完整训练恢复证明。无内部路径、平台日志、SATLIB原实例或历史权重被加入Git。
