# GRU 完整状态续训：候选入口与本次验证

状态：v0.4公开续训入口及短程验证记录；原完整断点和完整训练数据仍未发布。v0.3 的 24 个文件仍是纯推理权重，不能用来冒充完整续训状态。

## 已完成

找回 SAT74/A/Multi step5000 与 Maze77/A/Multi step7000 原完整状态，以及各自训练人口（3072、6736 题）。原文件保持不变；验证使用独立副本。

新增 `python -m latent_multi.gru.train` 续训入口复用已发布 GRU 模型与目标：K32，四个两题 microbatch，8/16 轮交替，AdamW、clip=1、共同质量项与0.1辅助、32参考及原参考轮换。Maze 保留逐位置最优动作概率质量与完整最短路径 DAG 质量；不是均匀动作 CE。没有持久跨批 carry，状态在每次 forward 初始化。

H200、FP32、PyTorch 2.7.0a0+ecf3bae40a.nv25.02、CUDA12.8：每任务比较连续两步和一步后保存、启动新进程再一步，全部模型张量、Adam、CPU/CUDA/Python/NumPy RNG、题序、游标、参考访问次数、更新日志及下一次先验 logits 逐位一致。两任务总实际执行8次更新，未重跑完整实验。

缺优化器、数据内容身份变化、配置变化、隐式 Multi/Set 切换均拒绝；显式 `--fork-arm` 可分叉并记录 lineage。详见[机器回执](resume_validation_candidate.json)。

## 使用入口（完整状态和训练数据尚未随包公开）

在仓库根目录安装后，以下命令使用用户自己有权访问的完整状态及训练 JSON：

```bash
python -m pip install -e .
python -m latent_multi.gru.train --task maze --rows inputs/maze_train.json --checkpoint inputs/maze_multi.pt --trusted-legacy --updates 1 --output outputs/gru_first --device cuda
python -m latent_multi.gru.train --task maze --rows inputs/maze_train.json --checkpoint outputs/gru_first/full_state.pt --updates 1 --output outputs/gru_resumed --device cuda
```

SAT 使用 `--task sat` 和相应的 SAT 训练 JSON / 完整状态。输出目录必须新建；不能用 v0.3 推理 state_dict 代替 `--checkpoint`。

`--trusted-legacy` 仅用于自己可信的原始训练 pickle。旧文件没有完整科学配置和数据内容身份；第一次接入是**明确绑定旧状态的兼容续训**，不能证明旧文件中不存在的元数据。之后保存的新格式核对完整输入/参考内容和冻结配置。模型、优化器、RNG等真实存在的旧状态已恢复；没有伪造历史环境或精确重演历史完整轨迹。

本入口覆盖已登记 main 阶段的 mass/Multi/Set 续训；不包含从零构建历史强质量前缀的全部训练流水线。旧环境与当前 CUDA 环境不同，短程一致性不等于跨软硬件逐位复现。本版GitHub CI仅覆盖CPU工程检查，实际状态见[工作流](https://github.com/Zane12518/latent-multitrajectory/actions/workflows/cpu.yml)。CUDA证据来自上述H200短程对照，旧版本和旧验证回执不回写。
