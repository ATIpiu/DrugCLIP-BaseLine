# DrugCLIP Odyssey v0.1

Drug-target interaction 虚拟筛选基线，基于对比学习对齐蛋白质口袋与配体分子。

## 目录结构

```
train/
├── config.py            # 配置 (ModelConfig, TrainConfig, Config)
├── model/               # 模型架构
│   ├── encoders.py      # PocketEncoder (3D voxel CNN) + FingerprintEncoder (MLP)
│   └── drugclip.py      # DrugCLIP: 对比学习对齐
├── data/                # 数据
│   ├── utils.py         # 指纹计算, PDB/MOL2/SDF 解析, pocket 提取
│   ├── dataset.py       # PDBbindDataset, DataLoader 工厂
│   └── benchmark.py     # benchmark 任务加载
├── core/                # 训练核心
│   ├── trainer.py       # Trainer (AdamW + warmup + cosine scheduler)
│   ├── loss.py          # InfoNCE + TripletMargin → CombinedLoss
│   └── eval.py          # VirtualScreeningEvaluator (EF1%, AUROC, BEDROC)
├── inference.py         # 117-task benchmark 推理
├── logger.py            # result.log 生成
├── run.py               # 入口点
└── prepare_data.py      # PDBbind 数据预处理
```

## 快速开始

```bash
# 安装依赖
pip install torch numpy rdkit

# 训练 + 推理（完整流程）
python -m train.run --mode full \
    --train-data data/pbpp-2020 \
    --benchmark-dir data/benchmark/benchmark \
    --output-dir output \
    --epochs 100 --batch-size 64

# 仅训练
python -m train.run --mode train --train-data data/pbpp-2020 --epochs 50

# 仅推理
python -m train.run --mode inference --ckpt output/checkpoints/best.pt
```

## 数据格式

训练数据使用 PDBbind 格式：
```
data/pbpp-2020/
    {pdb_id}/
        {pdb_id}_pocket.pdb    # 口袋原子坐标
        {pdb_id}_ligand.sdf    # 活性配体
        {pdb_id}_ligand.mol2   # 备选格式
```

## 验证指标

虚拟筛选检索指标：EF1%（top 1% 富集因子）、EF5%、AUROC、Top-K recall、MRR、BEDROC、RIE。
