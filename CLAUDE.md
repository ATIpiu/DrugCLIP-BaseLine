# DrugCLIP Odyssey

Drug-target interaction 虚拟筛选，基于 UniMol Transformer + 对比学习对齐蛋白质口袋与配体分子。

## 项目结构

```
DrugCLIP/
├── .env.example                  # 环境变量模板（复制为 .env 后填写）
├── CLAUDE.md                     # 本文件
├── agents/                       # Agent 框架
│   ├── llm_agent.py              # LLM 自然语言 Agent (DeepSeek tool calling)
│   ├── main_agent.py             # 规则 REPL + 自动化流水线
│   ├── tuning_agent.py           # LLM 驱动调参循环 (train→eval→LLM→adjust→repeat)
│   ├── model_agent.py            # 模型架构修改 + 参考文档阅读
│   ├── benchmark_agent.py        # Benchmark 提交
│   ├── data_agent.py             # 数据验证
│   ├── base_agent.py             # Agent 基类
│   ├── orch_logger.py            # 全局编排日志 (agent.log, JSON-lines)
│   └── agent_logger.py           # 子Agent 独立日志 (agent_logs/<name>/<session>.log)
├── train/                        # 训练代码
│   ├── tools/                    # Agent 可调用的工具
│   │   ├── train_tool.py         # 训练 → 返回 loss/checkpoint
│   │   ├── eval_tool.py          # 评估 → metrics + badcase
│   │   └── loss_tool.py          # Loss 分析 → trend + suggestions
│   ├── model/                    # UniMol Transformer, GBF, DrugCLIP
│   ├── data/                     # 3D 构象, tokenization, PDB 解析, dataset
│   ├── core/                     # Trainer, loss, evaluator
│   ├── inference.py              # Benchmark 推理
│   ├── config.py                 # EncoderConfig, ModelConfig, TrainConfig
│   ├── logger.py                 # OdysseyLogger (比赛用 result.log)
│   └── run.py                    # 入口: --mode train|inference|full|agent
├── data/
│   ├── pbpp-2020/                # 完整 PDBbind (5316 samples)
│   ├── fasttest/                 # 快速测试集 (320 samples)
│   └── benchmark/                # 117 任务 benchmark
└── output/
    ├── result.log                # 比赛提交日志 (Agent 优化叙事)
    ├── agent.log                 # 全局编排日志 (JSON-lines)
    ├── agent_logs/               # 各子Agent 详细日志
    │   ├── main/<session>.log
    │   ├── tuning/<session>.log
    │   ├── model/<session>.log
    │   └── benchmark/<session>.log
    └── checkpoints/
```

## 环境配置

```bash
cp .env.example .env
# 编辑 .env，填入 LLM_API_KEY
```

支持的环境变量：`LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`
兼容：`ANTHROPIC_AUTH_TOKEN`, `OPENAI_API_KEY`

## 常用命令

```bash
# 训练
python -m train.run --mode train --train-data data/pbpp-2020 --epochs 50

# 使用快速测试集训练（调试用）
python -m train.run --mode train --train-data data/fasttest --epochs 5 --batch-size 16

# 推理 + 生成提交
python -m train.run --mode inference --ckpt output/checkpoints/best.pt

# LLM Agent 交互模式
python -m train.run --mode agent --train-data data/pbpp-2020

# Agent 快速调试（使用测试集）
python -m train.run --mode agent --train-data data/fasttest
```

## Agent 开发

### 通信协议

所有 Agent 和 Tool 统一返回：
```python
{"status": "ok|error", "data": {...}, "summary": "one-line", "error": None}
```

### 添加新 Agent

1. 继承 `agents/base_agent.py` 的 `BaseAgent`
2. 实现 `run(context: dict) -> dict`
3. 在 `agents/llm_agent.py` 的 `TOOLS` 列表中添加函数定义
4. 在 `LLMMainAgent._tools` 注册处理函数

### 添加新 Tool

1. 在 `train/tools/` 中创建文件
2. 函数签名: `tool_name(config, ...) -> dict`（统一返回格式）
3. 在 `train/tools/__init__.py` 中导出

### 模型修改注意事项

数据走预处理（tokens + distances + edge_types），模型需保持 I/O 兼容：
- **可改**: `encoder_layers`, `encoder_embed_dim`, `encoder_ffn_embed_dim`, `attention_heads`, `dropout`, `project_dim`, `temperature`
- **不可改**: `gbf_k`, `ATOM_DICT`, `max_seq_len`

## 关键文件

- `agents/llm_agent.py:TOOLS` — LLM 可调用的函数定义（修改时同步更新）
- `train/config.py:Config` — 主配置（修改默认值时更新 logger.log_config）
- `train/core/trainer.py:epoch_metrics` — 结构化训练指标（loss_tool 依赖）
- `train/data/dataset.py:CachedPDBbindDataset` — 数据预加载 + 3D 构象生成
- `train/inference.py:_score_single_task` — Benchmark 推理核心
