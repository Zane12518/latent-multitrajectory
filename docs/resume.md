# 严格恢复、分叉与历史兼容

v0.2 默认写 `latent-multitrajectory-resume-v2`。推理权重不含完整状态，不能作为训练恢复文件。

## 默认 exact

恢复前检查完整配置（模型、训练日程、microbatch、辅助系数、配置预算及评价设置）、训练记录顺序与全部输入／参考内容、模型张量、Adam 参数绑定／矩／步号、Python/NumPy/Torch 与独立 prior RNG、题序游标、参考访问次数、ACT carry 及其输入身份。训练中的 arm 也必须一致。

数据用规范 JSON 的 BLAKE2b 内容指纹检测变更：相同 ID 但内容不同会拒绝。指纹不是签名或来源认证。原研究数据、旧 checkpoint 和旧统计均不改写。读取使用 `weights_only=True`；仍应只加载可信来源文件。

CPU/CUDA 后端与所记录软件版本必须匹配，CUDA RNG 设备数必须匹配。即使通过这些检查，也不主张跨 GPU 型号、驱动和平台的训练轨迹必然逐位相同。当前逐值恢复回归测试在 CPU 执行；本轮未新增 GPU 恢复实验。

`--updates N` 是本次进程执行的更新块长度，写入 summary；不重置全局 step 或学习率日程。继续执行更多块会增加累计训练预算，即使每一步状态延续正确，也不能声称仍是旧固定预算科研端点。配置中的计划块长度变更必须显式登记。

## 合法的 Multi／Set 分叉

从同一完整状态分别分叉，继承优化器、carry 与随机状态；不复制对方后续轨迹。

```bash
python training/train.py --config configs/cpu_smoke.json --data data/toy.json --resume outputs/toy_train/step00002.pt --resume-mode fork --arm multi --updates 1 --out outputs/toy_fork_multi
python training/train.py --config configs/cpu_smoke.json --data data/toy.json --resume outputs/toy_train/step00002.pt --resume-mode fork --arm set --updates 1 --out outputs/toy_fork_set
```

fork 的白名单只有训练臂 mass/multi/set，以及显式 `--stream` 的新随机流。其余配置和数据仍严格一致。产物记录 parent step/stream、原臂、新臂及随机流变更。改变续训流不是精确续跑，也不是新质量初始化。

## 显式新实验

只允许逐项声明 `training.schedule`、`training.aux_coef`、`training.microbatches`、`training.updates` 的真实差异；名称必须与差异完全吻合。模型结构、训练输入或参考内容变更不支持伪装成完整恢复，需另行设计权重初始化实验。

例如新配置只将辅助系数改为 .05：

```bash
python training/train.py --config configs/my_aux005.json --data data/toy.json --resume outputs/toy_train/step00002.pt --resume-mode new-experiment --allow-config-change training.aux_coef --arm set --updates 1 --out outputs/new_aux005
```

`configs/my_aux005.json` 是用户自行创建的实验配置，不随仓库提供；该命令是接口说明，不列为已运行快速示例。每个差异以 before/after 写入 lineage，不冒称原实验精确恢复。优化器超参数仍不允许静默改动。

## v0.1 历史状态

旧公开 v1 文件有模型、Adam、RNG、游标、carry 和配置，但缺少 v2 的完整内容指纹、实现协议与完整来源链。默认拒绝将其当作已验证 exact。显式 `--resume-mode legacy` 才允许读取，同时发出警告，并把“历史内容身份未验证”写入后续状态。

历史 CUDA 起点不能在 CPU 上恢复其 CUDA 随机状态；CPU 可加载推理权重，不是同一训练轨迹。历史 GPU 续训命令见[复现说明](reproduce.md)，本轮未重新训练。

历史文件保留原样，不补写伪造身份。v2 后代可以验证当前检查点以后的状态延续，不能追溯证明旧 v1 前史。
