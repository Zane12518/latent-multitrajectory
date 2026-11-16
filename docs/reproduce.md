# 复现与验证范围

## 1. 历史固定终点

`configs/maze10_gram.json`绑定当前0.1辅助配方。6个起点文件已含续训流RNG，不再传`--stream`重复重置。
`resume/source1301_flow2026111511.pt`等文件包含9000步模型、Adam一/二阶矩与步号、Python/NumPy/Torch CPU/CUDA RNG、独立先验RNG、题序与游标、reference_visits及ACT carry。
这是完整状态导出，未将EMA权重冒充恢复。推理权重不含优化器，不能传给`--resume`。

在与记录一致的GPU环境、下载完整资产后运行（本次仅小批验证，未重新跑1000步）：

```bash
python training/train.py --config configs/maze10_gram.json --data release_assets/data/train.json --resume release_assets/resume/source1301_flow2026111511.pt --arm multi --device cuda --out outputs/flow1511_multi
python training/train.py --config configs/maze10_gram.json --data release_assets/data/train.json --resume release_assets/resume/source1301_flow2026111511.pt --arm set --device cuda --out outputs/flow1511_set
python evaluation/evaluate.py --config configs/maze10_gram.json --data release_assets/data/test.json --checkpoint release_assets/weights/c1301_1511_multi.pt --device cuda --out outputs/eval_multi
python evaluation/evaluate.py --config configs/maze10_gram.json --data release_assets/data/test.json --checkpoint release_assets/weights/c1301_1511_set.pt --device cuda --out outputs/eval_set
python evaluation/paired.py --multi outputs/eval_multi/per_map.json --set outputs/eval_set/per_map.json --out outputs/paired.json
```

单对统计脚本的比较族为1对×4指标，不能把该区间与历史每source 3对×4指标的同时区间直接替换。
完整六组既有比较请运行`evaluation/reproduce_table.py`，复用相同20,000次地图重采样、种子与比较族。

## 2. 从头质量训练

质量前缀入口使用同模型、`--arm mass`、prefix日程；初始seed及9000步预算需在独立配置中记录。
当前候选重点是从可恢复强起点配对续训，未对新环境从头重训9000步；不要声称这一流程已重新复现。
从头跑新seed也不是对旧source的精确恢复。

## 3. 精度和随机性

正式历史推理BF16、Torch2.9.0+cu128、H100；CPU smoke为FP32/Torch2.7.1+cpu，不能要求它与CUDA高斯流逐值相同。
逻辑键绑定数据版本、地图序号、评价重复、生成流和32个高斯节点；相同端点对共享底噪。唯一键检查不证明严格IID。
K=1/4/8/16/32为同K32候选前缀，不是为每个K独立重新生成，更不是等墙钟比较。

## 4. 已测与未测

- 干净CPU安装、15项测试、2步真实训练、下一步恢复与推理：通过。
- 12个公开候选权重，每个固定首2图×K32×第1评价重复：H100输出与历史logits/argmax逐值相同。
- 完整9000步起点的一次更新和重放、完整状态读回、先验梯度非零：通过；工程更新不作为科研端点。
- 12模型×1200图×3评价流×5个K前缀的保存指标和六组同时区间：重算通过。
- 全量训练/全量重新生成：未重复执行；复用历史验收。跨GPU型号、跨软件版本的训练等价未主张。
- 首次GPU包装验收因未加载历史Pydantic依赖而失败；补齐依赖后的单次重试通过。失败记录保留在私有交付审计。
- GitHub匿名访问、公共下载与下载后安装：发布后另外核验，不将其混为历史模型训练验证。

测试不会证明方法普遍有效；完整科学结论与未确认条件见结果表。
