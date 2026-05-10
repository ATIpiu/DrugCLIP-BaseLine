# DrugCLIP — 高通量虚拟筛选优化智能体基线

第四届世界科学智能大赛 — AI4S 智能体 CNS 挑战赛 · 任务 1

基于 UniMol Transformer + 对比学习的蛋白质口袋—配体分子检索模型，配合 LLM 驱动的自主优化 Agent，
系统性提升虚拟筛选的早期富集表现（EF1%）。

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
相比 AUROC 等全局指标更贴近"只测试 top 少量化合物"的真实虚拟筛选场景。

### 提交格式

```
result.zip
├── result.csv    # task_id,ligand_id,score
└── result.log    # Agent 自主优化日志
```

要求：
- 每个 `(task_id, ligand_id)` 必须且只能出现一次
- `score` 越高排序越靠前
- `result.log` 需证明 Agent 确实进行了自主优化（非人工手调一次提交）

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
   w/ Pair Bias)    w/ Pair Bias)
       ↓                  ↓
   Projection       Projection
   Head             Head
       ↓                  ↓
  ┌─────────────────────────────┐
  │  Contrastive Loss           │
  │  InfoNCE + Triplet Margin   │
  │  (cosine similarity space)  │
  └─────────────────────────────┘
```

- **编码器**：UniMol Transformer（基于 3D 坐标的距离/边类型编码 + GBF 位置嵌入）
- **对比学习**：InfoNCE（batch 内负采样）+ Triplet Margin 联合损失
- **检索方式**：pocket 向量与 ligand 向量做内积，按相似度排序

### Agent 优化闭环

```
┌──────────────────────────────────────────────┐
│           DrugCLIP Agent Pipeline            │
│                                              │
│  ┌──────────┐    ┌──────────┐    ┌─────────┐ │
│  │ Phase 1  │ →  │ Phase 2  │ →  │Phase 3  │ │
│  │ 数据验证  │    │ 优化循环  │    │提交生成  │ │
│  └──────────┘    └──────────┘    └─────────┘ │
│                       │                       │
│          ┌────────────┼────────────┐          │
│          ↓            ↓            ↓          │
│    TuningAgent  train_tool   ModelAgent       │
│    (调参决策)    (训练+验证)  (改架构)        │
│          │            │            │          │
│          └────────────┼────────────┘          │
│                       ↓                       │
│              eval_tool + loss_tool             │
│              (评估 + badcase分析)              │
└──────────────────────────────────────────────┘
```

Agent 自动完成：分析 loss 趋势 → 提出优化假设 → 修改配置/模型 → 重新训练 → 对比改善 → 迭代收敛

---

## 环境配置

### 1. 创建环境

```powershell
conda create -n drugclip python=3.11 -y
conda activate drugclip
```

### 2. 安装 PyTorch

```powershell
# CUDA 12.x（推荐）
pip install torch --index-url https://download.pytorch.org/whl/cu128

# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

验证：

```powershell
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 3. 安装依赖

```powershell
pip install numpy rdkit scipy tqdm openai
```

### 4. 配置 LLM API Key（Agent 模式需要）

```powershell
copy .env.example .env
# 编辑 .env，填入 LLM_API_KEY
```

---

## 数据准备

### PDBbind 训练数据

数据集下载：[PDBbind-2020 on HuggingFace](https://huggingface.co/datasets/photonmz/pdbbindpp-2020/tree/main)（推荐使用更新或扩充的数据集替代）

```
data/pbpp-2020/
├── 187l/
│   ├── 187l_pocket.pdb       # 蛋白质口袋
│   ├── 187l_ligand.sdf       # 配体 SDF（或 mol2）
│   └── 187l_ligand.mol2
├── 1aaq/
│   └── ...
└── ...
```

### Benchmark 数据

由主办方提供，目录结构：

```
data/benchmark/benchmark/
├── manifest.jsonl            # 117 任务索引
├── tasks/
│   ├── dude_aa2ar/
│   │   ├── task.json
│   │   ├── ligands.csv
│   │   ├── receptors/
│   │   │   └── receptor.pdb
│   │   └── refs/
│   │       └── crystal_ligand.mol2
│   └── ...
```

---

## 快速开始

### Step 1：快速调试（30 秒）

```powershell
python -m train.run --mode train --train-data data/fasttest --epochs 1 --batch-size 8
```

期望输出：

```
Epoch   1 | Loss: 2.1334 | NCE: 2.0791 | Trip: 0.5426 | Temp: 0.2001 | Time: 3.2s
Training Complete | Best Loss: 2.1334
```

### Step 2：小规模训练（验证链路）

```powershell
python -m train.run --mode train --train-data data/pbpp-2020 --epochs 5 --batch-size 32 --max-samples 100
```

期望：Loss 从 ~3.5 下降，Val EF1 从 ~0.02 上升。

### Step 3：全量训练 + 提交生成

```powershell
python -m train.run --mode full                     \
  --train-data data/pbpp-2020                       \
  --benchmark-dir data/benchmark/benchmark           \
  --epochs 20 --batch-size 32
```

生成 `output/result.csv` 和 `output/result.zip`。

### Step 4：启动 Agent 自主优化

```powershell
# 快速调试模式
python -m train.run --mode agent --train-data data/fasttest

# 全量优化
python -m train.run --mode agent --train-data data/pbpp-2020
```

Agent 启动后进入交互 REPL，输入 `auto` 启动一键优化流水线，或输入 `help` 查看所有命令。

---

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 20 | 训练轮数 |
| `--batch-size` | 32 | 批大小（显存不足时改小） |
| `--lr` | 3e-4 | 学习率 |
| `--encoder-layers` | 8 | Transformer 层数 |
| `--encoder-dim` | 384 | 隐藏维度 |
| `--max-samples` | 0 | 限制加载样本数（0 = 全部） |
| `--device` | cuda | cuda / cpu |
| `--seed` | 42 | 随机种子 |

### 仅推理

```powershell
python -m train.run --mode inference \
  --ckpt output/models/drugclip/checkpoints/best.pt \
  --benchmark-dir data/benchmark/benchmark
```

---

## 训练指标

每个 epoch 输出：

```
Epoch   5 | Loss: 1.2345 | NCE: 1.1800 | Trip: 0.5450 | Temp: 0.2100 | Time: 45.2s | Val EF1: 0.150 AUROC: 0.720 MRR: 0.120 R@1: 0.050 R@5: 0.120
```

| 指标 | 含义 | 方向 |
|------|------|------|
| Loss | 总损失 = InfoNCE + 0.1 × Triplet | ↓ |
| NCE | 对比学习 InfoNCE 损失 | ↓ |
| Trip | Triplet Margin 损失 | ↓ |
| Temp | 可学习温度参数 | — |
| Val EF1 | 前 1% 富集因子 | ↑ |
| Val AUROC | 检索排序 AUROC（0.5 = 随机） | ↑ |
| Val MRR | 平均倒数排名 | ↑ |
| R@1 / R@5 | Top-1 / Top-5 召回率 | ↑ |

---

## 项目结构

```
DrugCLIP/
├── train/
│   ├── run.py                 # 入口：--mode train|inference|full|agent
│   ├── config.py              # 统一配置（EncoderConfig + TrainConfig + DataConfig）
│   ├── logger.py              # OdysseyLogger → output/result.log
│   ├── inference.py           # Benchmark 推理引擎 → result.csv + result.zip
│   ├── model/
│   │   ├── drugclip.py        # DrugCLIP：双 UniMol + 投影头
│   │   ├── unimol_encoder.py  # UniMol Transformer（token + GBF + pair bias）
│   │   ├── transformer.py     # TransformerEncoderWithPair
│   │   └── registry.py        # 模型注册表（Agent 可注册自定义模型）
│   ├── core/
│   │   ├── trainer.py         # Trainer（epoch 级验证 + 检索指标）
│   │   └── loss.py            # InfoNCE + TripletMargin
│   ├── data/
│   │   ├── dataset.py         # CachedPDBbindDataset + collate_fn
│   │   ├── utils.py           # 3D 构象生成、tokenization、PDB/MOL2 解析
│   │   └── benchmark.py       # Benchmark manifest 加载
│   └── tools/
│       ├── train_tool.py      # Agent 可调用训练工具
│       ├── eval_tool.py       # Agent 可调用评估工具
│       └── loss_tool.py       # Agent 可调用 loss 分析工具
├── agents/
│   ├── llm_agent.py           # DeepSeek/OpenAI tool calling 主控
│   ├── main_agent.py          # 规则 REPL + 自动化流水线
│   ├── tuning_agent.py        # LLM 驱动调参循环
│   ├── model_agent.py         # 模型架构自动设计
│   └── base_agent.py          # Agent 基类
├── output/
│   ├── result.log             # 比赛提交日志（Agent 优化叙事）
│   ├── result.csv / result.zip
│   └── models/drugclip/checkpoints/
└── data/
    ├── pbpp-2020/             # PDBbind 全量训练集
    ├── fasttest/              # 快速调试集
    └── benchmark/benchmark/   # 117 任务评测集
```

---

## Agent 开发

### 通信协议

所有 Agent 和 Tool 统一返回：

```python
{"status": "ok|error", "data": {...}, "summary": "one-line", "error": None}
```

### 添加自定义 Tool

1. 在 `train/tools/` 创建文件，函数签名：`tool_name(config, ...) -> dict`
2. 在 `train/tools/__init__.py` 导出
3. 在 `agents/llm_agent.py` 的 `TOOLS` 列表添加函数定义
4. 在 `LLMMainAgent._tools` 注册处理函数

### 添加自定义 Agent

1. 继承 `agents/base_agent.py` 的 `BaseAgent`
2. 实现 `run(context: dict) -> dict`
3. 在 `agents/llm_agent.py` 或 `agents/main_agent.py` 注册

### 模型架构修改约束

数据预处理固定（tokens + distances + edge_types），模型需保持 I/O 兼容：

| 可修改 | 不可修改 |
|--------|----------|
| `encoder_layers` | `gbf_k` |
| `encoder_embed_dim` | `ATOM_DICT` |
| `encoder_ffn_embed_dim` | `max_seq_len` |
| `attention_heads` | |
| `dropout` | |
| `project_dim` | |
| `temperature` | |

---

## 提分策略

以下是针对提升 Mean EF1% 的优化方向，按预期收益排序：

### 1. 对比学习增强（预期收益：高）

- **负采样策略优化**：当前 batch 内随机负采样区分度有限。可引入 hard negative mining（选相似度中等的样本作为负例）、cross-batch memory bank（MoCo 风格）存储更多负样本
- **多视角对比**：同一 pocket 的不同构象 / 同一 ligand 的不同 protonation 状态作为正例增强
- **加长训练**：对比学习收敛慢，epochs 从 20 → 100+ 通常带来稳定提升

### 2. 数据层面（预期收益：中—高）

- **构象采样增强**：训练时对 ligand 做多构象采样（当前只取最低能量构象），暴露模型于构象多样性
- **活性悬崖过滤**：筛选 PDBbind 中结合亲和力分布合理的样本，去除极端值
- **口袋裁剪策略**：pocket 定义范围影响编码质量。扩大/缩小 pocket 截断半径，找到最优窗口
- **数据扩充**：结合 ChEMBL / BindingDB 补充训练样本（需注意分布一致性）

### 3. 模型层面（预期收益：中）

- **编码器非对称设计**：pocket 和 ligand 物理尺寸不同，可使用不同层数/维度的 encoder（如 pocket 用更深网络）
- **预训练权重初始化**：用 UniMol 在大规模分子库上的预训练权重初始化 encoder
- **Attention 改进**：pocket-ligand cross-attention 或使用 pair-aware attention bias 的调优
- **Projection Head 加深**：当前单层线性投影 → 多层 MLP + BatchNorm，增强表示质量

### 4. 训练策略（预期收益：中）

- **学习率调度**：warmup + cosine decay 替代固定衰减，充分探索后再精准收敛
- **混合精度 + 大 batch**：启用 AMP + 梯度累积，增大有效 batch size → 更多负样本 → 更好对比信号
- **两阶段训练**：先冻结 encoder 只训 projection head → 再全量微调
- **Label Smoothing / Margin Tuning**：调整 triplet margin 和 InfoNCE temperature

### 5. 推理/排序策略（预期收益：中）

- **多模型集成**：不同 seed / 不同构象的模型分别打分后取均值
- **口袋多结构聚合**：LIT-PCBA 中每个靶点有多个 receptor 结构，对各结构分别打分后 max/mean 聚合
- **重排序（Re-ranking）**：引入基于药效团匹配或物理相互作用过滤的二次排序
- **构象系综打分**：对同一 ligand 生成多构象，取 max score，降低单一构象偏差

### 6. 评估/验证策略（预期收益：辅助）

- **留出法验证**：从 PDBbind 按 scaffold split（而非随机 split）划分验证集，更真实反映泛化能力
- **多指标监控**：同时关注 EF1%、EF5%、AUROC、BEDROC，避免单一指标过拟合
- **Badcase 驱动迭代**：自动分析 badcase 中 AUROC 最低的任务，针对性优化

### 7. Agent 优化闭环质量（预期收益：辅助）

- **LLM 驱动的假设-验证**：让 Agent 读取 loss 曲线 + badcase 分布 → 提出具体假设 → 执行对照实验 → 判断改善 → 记录决策链
- **多分支并行探索**：Agent 管理多组超参/架构变体并行训练，自动淘汰劣化方向
- **科学叙事日志**：确保 `result.log` 清晰体现每次优化的 hypothesis → rationale → action → outcome 四步闭环

- ~~老东西交出的焚决：可以直接使用 claude -p 通过命令行的环境变量设置对应 apikey url 等信息，直接开启最高权限，作为子 agent（骗你的，主Agent也可以是它，自己设计一下skills，设计一下工具）。AI 时代了，谁还传统手搓啊~~
---

## 常见问题

### Q：训练 Loss 不下降？

1. 确认 `pip install torch>=2.0.0`
2. 确认看到 `Loaded XXX valid samples` 且数量 > 0
3. warmup 阶段（默认前 5 epoch）LR 从 0 逐渐上升，Loss 下降慢是正常的
4. 在 `train/config.py` 中调整 `warmup_epochs`

### Q：CUDA Out of Memory？

```powershell
python -m train.run --mode train --train-data data/pbpp-2020 --batch-size 16
```

### Q：Agent 模式 LLM 调用失败？

1. 确认 `.env` 中 `LLM_API_KEY` 已正确填写
2. 确认 `LLM_BASE_URL` 可访问
3. 不使用 Agent 模式也可以直接跑训练 + 推理：`--mode full`

### Q：Benchmark 推理找不到 manifest.jsonl？

```powershell
python -m train.run --mode inference \
  --benchmark-dir data/benchmark/benchmark \
  --ckpt output/models/drugclip/checkpoints/best.pt
```

### Q：只想用 10% 数据快速迭代？

```powershell
python -m train.run --mode train --train-data data/pbpp-2020 --max-samples 530 --epochs 5
```
