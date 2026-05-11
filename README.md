# DrugCLIP — 高通量虚拟筛选优化智能体基线

第四届世界科学智能大赛 — AI4S 智能体 CNS 挑战赛 · 任务 1

基于 UniMol Transformer + 对比学习的蛋白质口袋—配体分子检索模型，配合 LLM 驱动的自主优化 Agent，
系统性提升虚拟筛选的早期富集表现（EF1%）。

> GitHub: [ATIpiu/DrugCLIP-BaseLine](https://github.com/ATIpiu/DrugCLIP-BaseLine)

## 赛题概述

蛋白结构预测（AlphaFold 等）使"全基因组尺度"药物发现成为可能。传统 docking 在海量化合物库面前算力不可承受。
Science 论文 *Deep contrastive learning enables genome-wide virtual screening* 提出 **DrugCLIP**：
将口袋与分子映射到同一向量空间，通过向量相似度实现超快检索，相比 docking 速度提升数个数量级。

本赛题进一步挑战：**构建科学智能体（Agent）自动执行"数据准备→训练/微调→评测→超参/策略迭代→结果汇报"的闭环，
系统性优化虚拟筛选的 EF1% 指标。**

### 评测数据

| 数据集 | 任务数 | 说明 |
|--------|--------|------|
| DUD-E | 102 | 每任务 1 个 receptor + 共晶配体 |
| LIT-PCBA | 15 | 每任务可能有多个 receptor 结构 |
| **总计** | **117** | 总配体数 2,092,260 |

### 评测指标

**Mean EF1%**（Enrichment Factor @ top 1%）— 衡量模型在排名前 1% 的候选里富集到活性分子的倍数。

---

## 方法

### 模型架构

```
┌─────────────┐    ┌─────────────┐
│ Ligand      │    │ Protein     │
│ SMILES      │    │ Pocket PDB  │
└──────┬──────┘    └──────┬──────┘
       ↓                  ↓
  3D Conformer         PDB Parser
       ↓                  ↓
  Tokenization      Tokenization
  + Distances       + Distances
  + Edge Types      + Edge Types
       ↓                  ↓
  UniMol Encoder    UniMol Encoder
  (Transformer     (Transformer
   w/ GBF bias)     w/ GBF bias)
       ↓                  ↓
  NonLinearHead     NonLinearHead
   Projection        Projection
       ↓                  ↓
  ┌─────────────────────────────┐
  │  InfoNCE + Triplet Margin   │
  │  L2-normalized embeddings   │
  └─────────────────────────────┘
```

- **编码器**：UniMol Transformer（基于 3D 坐标的 GBF 距离编码 + 可学习边类型偏置）
- **对比学习**：InfoNCE（batch 内负采样）+ Triplet Margin 联合损失
- **检索方式**：pocket 向量与 ligand 向量做内积，按相似度排序
- **预训练权重**：支持加载原始 DrugCLIP (unicore) checkpoint，自动匹配架构并冻结部分层微调

---

## 环境配置

```powershell
conda create -n drugclip python=3.11 -y
conda activate drugclip

pip install -r requirements.txt



验证：

```powershell
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## 数据准备

### PDBbind 训练数据

```
data/pbpp-2020/
├── 187l/
│   ├── 187l_pocket.pdb       # 蛋白质口袋
│   ├── 187l_ligand.sdf       # 配体 SDF（或 mol2）
│   └── 187l_ligand.mol2
└── ...
```

下载：[PDBbind-2020 on HuggingFace](https://huggingface.co/datasets/photonmz/pdbbindpp-2020/tree/main)

```powershell
cd data
hf download THU-ATOM/PDBbind --repo-type dataset --local-dir .\THU-ATOM_PDBbind
```

### 预训练模型

下载原始 DrugCLIP checkpoint（unicore 格式，15层/512维/64头）：

https://drive.google.com/drive/folders/1zW1MGpgunynFxTKXC2Q4RgWxZmg6CInV

放置到 `train/model/Base/checkpoint_best.pt`。

### Benchmark 数据

由主办方提供：

```
data/benchmark/benchmark/
├── manifest.jsonl            # 117 任务索引
├── tasks/
│   ├── dude_aa2ar/
│   │   ├── task.json
│   │   ├── ligands.csv
│   │   ├── receptors/
│   │   └── refs/
│   └── ...
```

---

## 快速开始

### 快速调试（30 秒）

```powershell
python -m train.run --mode train --train-data data/fasttest --epochs 1 --batch-size 8
```

### 微调预训练模型（推荐）

从原始 DrugCLIP checkpoint 出发，冻结前 12 层 encoder + embedding + GBF，只微调最后 3 层 + projection head：

```powershell
python -m train.run --mode train \
    --pretrained train/model/Base/checkpoint_best.pt \
    --freeze-encoder-layers 0,1,2,3,4,5,6,7,8,9,10,11 \
    --freeze-embeddings --freeze-gbf \
    --new-lr 1e-4 \
    --train-data data/pbpp-2020 \
    --epochs 20 --batch-size 32
```

微调模式会自动：
- 检测 checkpoint 架构（层数/维度/heads/原子类型数）
- 构建匹配模型（NonLinearHead projection + GBF projection）
- 加载全部 encoder + projection + GBF 权重
- 加载 checkpoint 对应的 atom 字典进行数据 tokenization
- 使用 `[BOS]` token pooling（匹配预训练方式）

### 全量训练 + 提交生成

```powershell
python -m train.run --mode full \
    --pretrained train/model/Base/checkpoint_best.pt \
    --freeze-encoder-layers 0,1,2,3,4,5,6,7,8,9,10,11 \
    --freeze-embeddings --freeze-gbf \
    --new-lr 1e-4 \
    --train-data data/pbpp-2020 \
    --benchmark-dir data/benchmark/benchmark \
    --epochs 20 --batch-size 32
```

### 仅推理

```powershell
python -m train.run --mode inference \
    --ckpt output/models/drugclip/checkpoints/best.pt \
    --benchmark-dir data/benchmark/benchmark
```

推理引擎自动从 `config.json` 读取完整模型架构和 atom 字典配置。

### Agent 自主优化

```powershell
python -m train.run --mode agent --train-data data/pbpp-2020
```

Agent 支持微调模式调参：可自动调整冻结层数、学习率、batch size 等。

---

## 关键参数

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 20 | 训练轮数 |
| `--batch-size` | 32 | 批大小 |
| `--lr` | 3e-4 | 学习率 |
| `--encoder-layers` | 8 | Transformer 层数（微调时自动检测） |
| `--encoder-dim` | 384 | 隐藏维度（微调时自动检测） |
| `--max-samples` | 0 | 限制加载样本数（0 = 全部） |
| `--device` | cuda | cuda / cpu |
| `--seed` | 42 | 随机种子 |

### 微调参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pretrained` | None | 预训练 checkpoint 路径（unicore 或 odyssey 格式） |
| `--freeze-encoder-layers` | "" | 逗号分隔的冻结层索引，如 `"0,1,2"` |
| `--freeze-embeddings` | False | 冻结 token embedding |
| `--freeze-gbf` | False | 冻结 GBF 层（means/stds/mul/bias + projection） |
| `--freeze-project` | False | 冻结 projection head |
| `--new-lr` | None | 微调学习率（覆盖 `--lr`） |
| `--mol-atom-types` | None | 覆盖 mol 原子类型数（自动检测） |
| `--pocket-atom-types` | None | 覆盖 pocket 原子类型数（自动检测） |
| `--no-bos-pool` | False | 使用 mean pooling 代替 `[BOS]` pooling |

### 微调示例

```powershell
# 只解冻 1 层（layer 14），5 epoch 快速验证
python -m train.run --mode train --pretrained train/model/Base/checkpoint_best.pt \
    --freeze-encoder-layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13 \
    --freeze-embeddings --freeze-gbf --new-lr 1e-4 \
    --train-data data/pbpp-2020 --max-samples 530 --epochs 5

# 解冻后 3 层，全量数据，100 epoch
python -m train.run --mode train --pretrained train/model/Base/checkpoint_best.pt \
    --freeze-encoder-layers 0,1,2,3,4,5,6,7,8,9,10,11 \
    --freeze-embeddings --freeze-gbf --new-lr 1e-4 \
    --train-data data/pbpp-2020 --epochs 100 --batch-size 32
```

---

## 训练指标

每个 epoch 输出：

```
Epoch  20 | Loss: 1.1879 | NCE: 1.1409 | Trip: 0.4695 | Temp: 0.0663 | Time: 43.0s | Val EF1: 0.166 AUROC: 0.853 MRR: 0.118 R@1: 0.046 R@5: 0.166
```

| 指标 | 含义 | 方向 |
|------|------|------|
| Loss | 总损失 = InfoNCE + 0.1 × Triplet | ↓ |
| NCE | InfoNCE 对比损失 | ↓ |
| Trip | Triplet Margin 损失 | ↓ |
| Temp | 可学习温度参数 | — |
| Val EF1 | 前 1% 富集因子 | ↑ |
| Val AUROC | 检索排序 AUROC | ↑ |
| Val MRR | 平均倒数排名 | ↑ |
| R@1 / R@5 | Top-1 / Top-5 召回率 | ↑ |

---

## 微调效果

| 数据量 | 解冻层 | Epochs | Val EF1 | Val AUROC |
|--------|--------|--------|---------|-----------|
| 10% (464) | 1 层 | 5 | 0.021 | 0.612 |
| 50% (2336) | 1 层 | 10 | 0.075 | 0.767 |
| 100% (4239) | 3 层 | 20 | **0.166** | **0.853** |

---

## 项目结构

```
DrugCLIP/
├── train/
│   ├── run.py                 # 入口：--mode train|inference|full|agent
│   ├── config.py              # 统一配置（EncoderConfig + ModelConfig + TrainConfig）
│   ├── logger.py              # OdysseyLogger → output/result.log
│   ├── inference.py           # Benchmark 推理引擎 → result.csv + result.zip
│   ├── model/
│   │   ├── drugclip.py        # DrugCLIP + NonLinearHead projection
│   │   ├── unimol_encoder.py  # UniMol Transformer (token + GBF + pair bias) + NonLinearHead
│   │   ├── transformer.py     # TransformerEncoderWithPair
│   │   └── registry.py        # 模型注册表
│   ├── core/
│   │   ├── trainer.py         # Trainer (预训练加载 + 层冻结 + 微调)
│   │   └── loss.py            # InfoNCE + TripletMargin
│   ├── data/
│   │   ├── dataset.py         # CachedPDBbindDataset (支持自定义 atom dict)
│   │   ├── utils.py           # 3D 构象、tokenization、PDB/MOL2 解析、load_atom_dict
│   │   └── benchmark.py       # Benchmark manifest 加载
│   └── tools/
│       ├── train_tool.py      # Agent 训练工具
│       ├── eval_tool.py       # Agent 评估工具
│       └── loss_tool.py       # Agent loss 分析工具
├── agents/
│   ├── llm_agent.py           # LLM tool calling 主控
│   ├── main_agent.py          # 规则 REPL + 自动化流水线
│   ├── tuning_agent.py        # LLM 驱动调参（支持微调模式）
│   ├── model_agent.py         # 模型架构自动设计
│   └── base_agent.py          # Agent 基类
├── output/
│   └── models/drugclip/checkpoints/
└── data/
    ├── pbpp-2020/             # PDBbind 训练集
    ├── fasttest/              # 快速调试集
    └── benchmark/             # 117 任务评测集
```

---

## 常见问题

### Q：CUDA Out of Memory？

```powershell
python -m train.run --mode train --batch-size 16 --train-data data/pbpp-2020
```

### Q：Agent 模式 LLM 调用失败？

1. 确认 `.env` 中 `LLM_API_KEY` 已正确填写
2. 不使用 Agent 模式也可以直接跑训练 + 推理：`--mode full`

### Q：微调后 EF1 很低？

1. 确认 `--pretrained` 路径正确
2. 检查日志中 `Atom dicts loaded: mol=30 types, pocket=9 types`（表示字典加载成功）
3. 确认 `Missing keys: 4`（只有 extra heads 缺失，其他全部加载）
4. 尝试解冻更多层（减少 `--freeze-encoder-layers`）
5. 增加 `--epochs` 和 `--max-samples`

### Q：只想用 10% 数据快速迭代？

```powershell
python -m train.run --mode train --pretrained train/model/Base/checkpoint_best.pt \
    --freeze-encoder-layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13 \
    --freeze-embeddings --freeze-gbf --new-lr 1e-4 \
    --train-data data/pbpp-2020 --max-samples 530 --epochs 5
```
