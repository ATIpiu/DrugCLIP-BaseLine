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

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

验证：

```powershell
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## 数据准备

### PDBbind 训练数据

本项目使用 **THU-ATOM PDBbind**（2808 个蛋白-配体复合物，含 ESMFold 对齐结构），已上传至 ModelScope：

```
data/THU-ATOM_PDBbind/
├── 187l/
│   ├── 187l_ligand.sdf                          # 配体 SDF
│   ├── 187l_ligand.mol2                         # 配体 MOL2
│   ├── 187l_protein_processed_fix.pdb           # 蛋白全结构（用于口袋提取）
│   └── 187l_protein_esmfold_aligned_tr_fix.pdb  # ESMFold 对齐结构
└── ...（共 2808 个复合物）
```

**ModelScope 下载：**

```powershell
conda activate drugclip
modelscope download ATIpiu/THU-ATOM_PDBbind_For_AI4S THU-ATOM_PDBbind.zip --repo-type dataset --local-dir data/
# 解压
cd data && unzip THU-ATOM_PDBbind.zip && cd ..
```

或 Python SDK：

```python
from modelscope.hub.api import HubApi
api = HubApi()
api.download_file(
    model_id='ATIpiu/THU-ATOM_PDBbind_For_AI4S',
    file_path='THU-ATOM_PDBbind.zip',
    local_dir='data/',
    repo_type='dataset',
)
# 解压后目录为 data/THU-ATOM_PDBbind/
```

数据集页面：https://www.modelscope.cn/datasets/ATIpiu/THU-ATOM_PDBbind_For_AI4S

### 预训练模型

原始 DrugCLIP checkpoint（unicore 格式，15层/512维/64头），已上传至 ModelScope：

**ModelScope 下载（推荐）：**

```powershell
# 方式一：CLI（单文件，约 1.1 GB）
conda activate drugclip
modelscope login --token YOUR_TOKEN
modelscope download ATIpiu/DrugCLIP-Base checkpoint_best.pt --local-dir train/model/Base

# 方式二：Python SDK
conda activate drugclip
python -c "
from modelscope.hub.api import HubApi
api = HubApi()
api.login('YOUR_TOKEN')
from modelscope import snapshot_download
snapshot_download('ATIpiu/DrugCLIP-Base', local_dir='train/model/Base')
"

# 方式三：Git LFS
git lfs install
git clone https://www.modelscope.cn/ATIpiu/DrugCLIP-Base.git train/model/Base
```

模型页面：https://www.modelscope.cn/models/ATIpiu/DrugCLIP-Base

下载后确认文件位于 `train/model/Base/checkpoint_best.pt`。

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

### 第一步：完整流程验证（fasttest，约 3 分钟跑通）

> 用内置的 10 个复合物样本，从训练到出推理结果全跑一遍，确认环境正常。

**1. 微调训练（~10 秒）**

```powershell
python -m train.run --mode train \
    --train-data data/fasttest \
    --pretrained train/model/Base/checkpoint_best.pt \
    --freeze-encoder-layers 0,1,2,3,4,5,6,7,8,9,10,11 \
    --freeze-embeddings --freeze-gbf \
    --new-lr 1e-4 \
    --epochs 3 --batch-size 4
```

正常输出（关键日志）：

```
Loaded pretrained: train/model/Base/checkpoint_best.pt
  Missing keys: 4  Unexpected keys: 0        ← 只有 extra head 缺失，正常
Atom dicts loaded: mol=30 types, pocket=9 types   ← 字典加载成功
Freeze: 75,730,378 frozen / 19,575,297 trainable params
Epoch   3 | Loss: 1.74 | NCE: 1.69 | ...
Training Complete | Best Loss: 1.74
```

输出 checkpoint：`output/models/drugclip/checkpoints/best.pt`

---

**2. 预缓存 benchmark 分子（~1 分钟，快速测试模式）**

> benchmark 共 2,092,260 条配体，全量预缓存需数小时。用 `--max-smiles` 先跑小批验证流程。

```powershell
python scripts/build_mol_cache.py --workers 4 --max-smiles 1000
```

正常输出：

```
Atom dict : data/dict_mol.txt  (30 types)
共 1,000 个唯一 SMILES
已缓存: 0 / 1,000
开始 RDKit 3D 构象生成（4 线程）...
  [  1,000/1,000] 成功 998 失败 2 | 44 mol/s | ETA 0.0 min
完成！写入 998 条，DB 大小 xx MB
```

---

**3. 推理（~1 秒，单任务验证）**

> 先用最小任务（811 个配体）验证，避免等待大任务。

```powershell
python -m train.run --mode inference \
    --ckpt output/models/drugclip/checkpoints/best.pt \
    --task-id litpcba_TP53
```

正常输出：

```
Atom dicts: mol=30 types, pocket=9 types
Auto-adapting arch: mol(dim=512,...,gbf_k=128) pocket(vocab=10,gbf_k=128)
Loaded checkpoint: ... (epoch 3)
Total tasks: 1 (limited)
  Preparing 811 ligands via RDKit ...     ← 第一次跑 RDKit（~10s）
  litpcba_TP53: 811 ligands, 10.2s
result.csv: output/result.csv (811 entries)
Submission: output/result.zip            ← 可提交的压缩包
```

第二次运行会直接命中 Level-2 embedding 缓存，**0.6s** 完成：

```powershell
python -m train.run --mode inference \
    --ckpt output/models/drugclip/checkpoints/best.pt \
    --task-id litpcba_TP53
# → [emb cache] 811/811 ligands fully cached
# → litpcba_TP53: 811 ligands, 0.6s
```

---

### 正式训练（完整数据）

从原始 DrugCLIP checkpoint 出发，冻结前 12 层 encoder + embedding + GBF，只微调最后 3 层 + projection head：

```powershell
python -m train.run --mode train \
    --pretrained train/model/Base/checkpoint_best.pt \
    --freeze-encoder-layers 0,1,2,3,4,5,6,7,8,9,10,11 \
    --freeze-embeddings --freeze-gbf \
    --new-lr 1e-4 \
    --train-data data/THU-ATOM_PDBbind \
    --epochs 20 --batch-size 32
```

微调模式会自动：
- 检测 checkpoint 架构（层数/维度/heads/原子类型数）
- 加载全部 encoder + GBF + projection 权重
- 使用 `data/dict_mol.txt`（30 类型）和 `data/dict_pkt.txt`（9 类型）进行 tokenization

### 全量训练 + 提交生成

```powershell
python -m train.run --mode full \
    --pretrained train/model/Base/checkpoint_best.pt \
    --freeze-encoder-layers 0,1,2,3,4,5,6,7,8,9,10,11 \
    --freeze-embeddings --freeze-gbf \
    --new-lr 1e-4 \
    --train-data data/THU-ATOM_PDBbind \
    --benchmark-dir data/benchmark/benchmark \
    --epochs 20 --batch-size 32
```

### 仅推理（全量 benchmark）

```powershell
# 全量 117 任务（建议先跑完整预缓存，否则耗时数小时）
python -m train.run --mode inference \
    --ckpt output/models/drugclip/checkpoints/best.pt

# 指定单任务（快速验证）
python -m train.run --mode inference \
    --ckpt output/models/drugclip/checkpoints/best.pt \
    --task-id litpcba_TP53

# 限制前 N 个任务
python -m train.run --mode inference \
    --ckpt output/models/drugclip/checkpoints/best.pt \
    --max-tasks 5
```

### 推理加速：预缓存全量分子库

benchmark 共有 **2,092,260 条**配体，全量预缓存后推理无需 RDKit，速度提升 10-100x：

```powershell
# 全量预缓存（约 3~6 小时，只需一次，之后永久生效）
python scripts/build_mol_cache.py --workers 4

# 快速测试（只缓存前 N 个，验证流程）
python scripts/build_mol_cache.py --workers 4 --max-smiles 1000

# 中断后续跑：自动跳过已缓存条目
python scripts/build_mol_cache.py --workers 4
```

> **为什么推理慢？** 未预缓存时每个 SMILES 需实时跑 RDKit 3D 构象生成（约 44 mol/s）。
> benchmark 最大任务有 8 万个配体，全跑 RDKit 需 30+ 分钟。**先跑预缓存是关键。**

缓存层级说明：

| 层级 | 内容 | 位置 | 生命周期 |
|------|------|------|----------|
| Level-1 Token 缓存 | SMILES → tokens / distances / edge_types | `output/mol_cache/tokens.db` | 永久，与 checkpoint 无关 |
| Level-2 Embedding 缓存 | SMILES → 嵌入向量 | `output/mol_cache/emb/{ckpt_hash}/{task_id}.pkl` | 按 checkpoint 隔离 |

命中顺序：**Level-2（跳过 RDKit + 模型前向）→ Level-1（跳过 RDKit）→ 实时 RDKit**

### Agent 自主优化

```powershell
python -m train.run --mode agent --train-data data/THU-ATOM_PDBbind
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
    --train-data data/THU-ATOM_PDBbind --max-samples 530 --epochs 5

# 解冻后 3 层，全量数据，100 epoch
python -m train.run --mode train --pretrained train/model/Base/checkpoint_best.pt \
    --freeze-encoder-layers 0,1,2,3,4,5,6,7,8,9,10,11 \
    --freeze-embeddings --freeze-gbf --new-lr 1e-4 \
    --train-data data/THU-ATOM_PDBbind --epochs 100 --batch-size 32
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
│   ├── inference.py           # Benchmark 推理引擎（含两级磁盘缓存）→ result.csv + result.zip
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
├── scripts/
│   ├── build_mol_cache.py     # 预缓存全量 benchmark SMILES tokenization（与 ckpt 无关）
│   ├── fill_random.py         # 填充随机分数（调试用）
│   └── pack_submission.py     # 打包提交文件
├── output/
│   ├── models/drugclip/checkpoints/   # 训练输出的 checkpoint
│   └── mol_cache/
│       ├── tokens.db                  # Level-1：SMILES → tokenized data（SQLite）
│       └── emb/{ckpt_hash}/           # Level-2：SMILES → 嵌入向量（per-task pkl）
└── data/
    ├── THU-ATOM_PDBbind/      # PDBbind 训练集（2808 复合物，ModelScope 下载）
    ├── fasttest/              # 快速调试集
    └── benchmark/             # 117 任务评测集
```

---

## 常见问题

### Q：CUDA Out of Memory？

```powershell
python -m train.run --mode train --batch-size 16 --train-data data/THU-ATOM_PDBbind
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
    --train-data data/THU-ATOM_PDBbind --max-samples 530 --epochs 5
```

### Q：推理太慢怎么办？

先跑一次预缓存脚本（只需一次，之后永久生效）：

```powershell
python scripts/build_mol_cache.py --workers 4
```

缓存完成后，再次推理时：
- **Level-1 命中**：跳过 RDKit 构象生成（最慢的 CPU 步骤）
- **Level-2 命中**：额外跳过模型前向传播，单任务耗时从分钟级降至秒级

### Q：换了新 checkpoint，Level-2 缓存还有效吗？

Level-1（tokens.db）始终有效。Level-2 嵌入缓存按 checkpoint 文件大小 + 修改时间自动隔离，
换 checkpoint 后首次推理会重新编码并建立新 checkpoint 的 Level-2 缓存。
