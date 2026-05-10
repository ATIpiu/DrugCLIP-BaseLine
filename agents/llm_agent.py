"""LLM-powered Main Agent — natural language → tool calling → execution.

Uses DeepSeek/OpenAI tool-calling API for intent understanding.
"""

import json
import os
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

# Load .env if present (python-dotenv not required — manual parsing)
def _load_dotenv():
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
_load_dotenv()

from openai import OpenAI


# ── Tool definitions (OpenAI/DeepSeek tool-calling format) ────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "train_model",
            "description": "开始训练 DrugCLIP 模型。使用当前配置进行对比学习训练。",
            "parameters": {
                "type": "object",
                "properties": {
                    "epochs": {"type": "integer", "description": "训练轮数"},
                    "lr": {"type": "number", "description": "学习率，如 3e-4"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_loss",
            "description": "分析最近一次训练的 loss 曲线，返回趋势判断和优化建议。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_evaluation",
            "description": "运行验证集评估，返回 AUROC/EF1%/MRR 等虚拟筛选指标和 badcase 分析。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tune_hyperparams",
            "description": "启动 LLM 驱动的自动调参循环。每轮迭代：训练→评估→LLM分析→调整参数→再训练→对比改善。",
            "parameters": {
                "type": "object",
                "properties": {
                    "num_iterations": {"type": "integer", "description": "调参轮数，默认 3。每轮需训练两次（调整前后各一次）"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "modify_architecture",
            "description": "设计并生成新的模型架构文件。调用 ModelAgent（OpenAI规划 + Claude Code代码生成），自动注册到模型注册表。",
            "parameters": {
                "type": "object",
                "properties": {
                    "request": {"type": "string", "description": "对新模型的描述，如 '设计一个用Cross-Attention替代对比学习的模型'"},
                    "reference_doc": {"type": "string", "description": "参考文档路径（可选）"},
                },
                "required": ["request"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_benchmark",
            "description": "使用训练好的模型跑 117 个 benchmark 任务推理，生成 result.csv 和 result.zip。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_data",
            "description": "检查训练数据目录格式和状态。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_status",
            "description": "查看当前配置、训练历史、checkpoint 路径、参考文档等完整状态。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_config",
            "description": "修改单个配置项。",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "配置路径，如 train.lr, train.batch_size, model.mol.encoder_layers"},
                    "value": {"type": "string", "description": "新值"},
                },
                "required": ["key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_reference",
            "description": "加载参考文档供模型架构修改时参考。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文档路径"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "auto_optimize",
            "description": "一键自动优化：训练→评估→调参→再训练，迭代多轮直到收敛，最后生成提交。",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_iterations": {"type": "integer", "description": "最大迭代轮数，默认 5"},
                },
            },
        },
    },
]

SYSTEM_PROMPT = """你是 DrugCLIP Odyssey 的主控 Agent，负责药物虚拟筛选模型的训练和优化。

## 核心能力

1. **训练模型** (train_model)：使用对比学习训练 DrugCLIP，自动进行 per-epoch 验证（EF1%/AUROC/MRR）
2. **分析 Loss** (analyze_loss)：分析训练 loss 曲线，判断收敛状态，给出调参建议
3. **运行评估** (run_evaluation)：在验证集上评估 AUROC/EF1%/MRR + badcase 分析
4. **自动调参** (tune_hyperparams)：LLM 驱动调参循环，每轮 train→eval→分析→调整→再train
5. **修改架构** (modify_architecture)：设计新模型架构，自动生成代码并注册到模型注册表
6. **Benchmark** (run_benchmark)：117 任务推理 → 生成 result.csv + result.zip
7. **数据检查** (check_data)：检查训练数据目录格式与统计
8. **配置管理** (update_config / show_status)：查看/修改模型和训练配置
9. **自动优化** (auto_optimize)：一键端到端优化流水线（data→train→eval→tune→benchmark）
10. **加载参考** (load_reference)：加载外部文档辅助架构设计

## 工作原则

1. 理解用户意图后立即调用工具，不要空谈方案
2. 多步骤任务逐步执行：先训练→再评估→再分析→再优化
3. 每次工具返回后，根据实际结果决定下一步（不预设路径）
4. 训练失败时：先 check_data 确认数据，再 show_status 确认配置
5. 评估失败时：确认已训练过（有 checkpoint），再重试
6. 调参前应确保至少完成过一次训练 + 评估
7. 用简洁中文回复：做了什么 → 什么结果 → 下一步建议
8. 用户请求模糊时，列出具体选项让用户选择（如 "数据在哪？请给路径"）

## 关键指标解读

- **Val EF1%**：前 1% 富集因子，越高越好（主比赛指标）
- **Val AUROC**：检索排序质量，0.5=随机，1.0=完美
- **Val MRR**：平均倒数排名，越高越好
- **Train Loss**：总损失 = InfoNCE + 0.1 × Triplet，应持续下降
- 如果 Val EF1 < 0.05 且不涨 → 模型未学到有效表示，检查数据和配置
- 如果 Train Loss 下降但 Val EF1 不涨 → 过拟合，增大 dropout 或减小模型
- 如果 Train Loss 不降 → 学习率过高，降低 lr"""


# ── LLM Main Agent ────────────────────────────────────────────────

class LLMMainAgent:
    """Natural language DrugCLIP agent powered by DeepSeek/OpenAI tool calling."""

    def __init__(
        self,
        config,
        logger=None,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
    ):
        self.config = config
        self.logger = logger

        # API config — priority: explicit arg > LLM_* env > ANTHROPIC_* env > OPENAI_* env
        api_key = (api_key
                   or os.environ.get("LLM_API_KEY")
                   or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                   or os.environ.get("OPENAI_API_KEY"))
        base_url = (base_url
                    or os.environ.get("LLM_BASE_URL")
                    or "https://api.deepseek.com")
        model = (model
                 or os.environ.get("LLM_MODEL")
                 or os.environ.get("ANTHROPIC_DEFAULT_SONNET_MODEL")
                 or os.environ.get("ANTHROPIC_MODEL")
                 or "deepseek-v4-pro")
        model = re.sub(r"\[\d+m\]", "", model).strip()

        self.client = OpenAI(api_key=api_key, base_url=base_url) if api_key else None

        # Orchestration logger + per-agent logger
        from .orch_logger import OrchLogger
        from .agent_logger import AgentLogger
        self.orch = OrchLogger(config.data.output_dir)
        self.agent_log = AgentLogger(config.data.output_dir, "main")
        self.model = model

        self.context = {"config": config, "checkpoint": None, "history": [], "reference_docs": {}}
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Tool registry
        self._tools = {
            "train_model": self._train,
            "analyze_loss": self._loss,
            "run_evaluation": self._eval,
            "tune_hyperparams": self._tune,
            "modify_architecture": self._model,
            "run_benchmark": self._benchmark,
            "check_data": self._data,
            "show_status": self._status,
            "update_config": self._set_config,
            "load_reference": self._load_ref,
            "auto_optimize": self._auto,
        }

    # ── Main loop ──────────────────────────────────────────────────

    def run(self):
        """Interactive REPL with streaming LLM + tool calling."""
        if not self.client:
            print("Error: No API key. Set ANTHROPIC_AUTH_TOKEN or OPENAI_API_KEY.")
            return

        self._print_banner()
        self._print_quick_help()

        while True:
            try:
                user_input = input("\n>>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n退出。")
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q", "退出"):
                self._log("Goodbye.")
                break
            if user_input.lower() in ("help", "h", "?", "帮助"):
                self._print_quick_help()
                continue
            if user_input.lower() in ("status", "st", "状态"):
                self._print_status()
                continue

            self.messages.append({"role": "user", "content": user_input})
            self.orch.user_message(user_input)

            while True:
                try:
                    resp, tool_calls, reasoning = self._stream_call()
                except KeyboardInterrupt:
                    print("\n  已中断 (LLM stream)")
                    self.orch.error("main", "KeyboardInterrupt during LLM stream")
                    break
                except Exception as e:
                    self._log(f"  API 错误: {e}")
                    self.orch.error("main", str(e))
                    break

                # Log full LLM response
                self.orch.llm_response(resp, tool_calls, reasoning)

                # LLM called a tool
                if tool_calls:
                    try:
                        for tc in tool_calls:
                            name = tc["name"]
                            args = tc["arguments"]
                            self.orch.llm_tool_call(name, args)

                            arg_str = json.dumps(args, ensure_ascii=False)
                            if len(arg_str) > 80:
                                arg_str = arg_str[:77] + "..."
                            print(f"  🔧 {name}({arg_str})")

                            result = self._execute(name, args)

                            result_str = json.dumps(result, ensure_ascii=False, default=str)
                            lines = result_str.split("\n")
                            if len(lines) > 10:
                                for line in lines[:10]:
                                    print(f"     {line[:120]}")
                                print(f"     ... ({len(lines) - 10} more lines folded)")
                            else:
                                for line in lines:
                                    print(f"     {line[:120]}")

                            assistant_msg = {
                                "role": "assistant",
                                "content": resp or "",
                                "tool_calls": [{
                                    "id": tc["id"], "type": "function",
                                    "function": {"name": name, "arguments": json.dumps(args)},
                                }],
                            }
                            if reasoning:
                                assistant_msg["reasoning_content"] = reasoning
                            self.messages.append(assistant_msg)
                            self.messages.append({
                                "role": "tool", "tool_call_id": tc["id"],
                                "content": result_str,
                            })
                    except KeyboardInterrupt:
                        print("\n  已中断 (tool)")
                        self.orch.error("main", "KeyboardInterrupt during tool execution")
                        break

                    # Suggest next step after tool execution
                    hint = self._suggest_next(name, result)
                    if hint:
                        print(f"\n  💡 {hint}")
                    continue

                # Text-only response (no more tool calls)
                resp_msg = {"role": "assistant", "content": resp}
                if reasoning:
                    resp_msg["reasoning_content"] = reasoning
                self.messages.append(resp_msg)
                print()  # blank line after response
                break

    # ── Stream call ────────────────────────────────────────────────

    def _stream_call(self) -> tuple:
        from .llm_utils import stream_llm
        return stream_llm(self.client, self.model, self.messages, tools=TOOLS)

    # ── Tool executor ──────────────────────────────────────────────

    def _execute(self, name: str, args: dict) -> dict:
        handler = self._tools.get(name)
        if not handler:
            return {"status": "error", "summary": f"Unknown tool: {name}"}
        self.orch.tool_call(name, args)
        try:
            result = handler(args)
            self.orch.tool_result(name, result)
            return result
        except KeyboardInterrupt:
            self.orch.error(name, "KeyboardInterrupt during tool execution")
            raise
        except Exception as e:
            err = {"status": "error", "summary": str(e), "traceback": traceback.format_exc()}
            self.orch.error(name, traceback.format_exc())
            return err

    # ── Tool handlers ──────────────────────────────────────────────

    def _train(self, args: dict) -> dict:
        if args.get("epochs"):
            self.config.train.epochs = int(args["epochs"])
        if args.get("lr"):
            self.config.train.lr = float(args["lr"])
        from train.tools.train_tool import train_tool
        result = train_tool(self.config)
        if result["status"] == "ok":
            self.context["checkpoint"] = result["data"]["checkpoint_path"]
            self.context["train_results"] = [result]
            from train.tools.loss_tool import loss_tool
            per_epoch = result["data"].get("per_epoch_metrics", [])
            self.context["loss_data"] = loss_tool(per_epoch)
        return result

    def _loss(self, args: dict) -> dict:
        tr = self.context.get("train_results", [])
        if not tr:
            return {"status": "error", "summary": "尚未训练，请先执行 train_model"}
        from train.tools.loss_tool import loss_tool
        result = loss_tool(tr[-1]["data"].get("per_epoch_metrics", []))
        self.context["loss_data"] = result
        return result

    def _eval(self, args: dict) -> dict:
        ckpt = self.context.get("checkpoint")
        if not ckpt:
            return {"status": "error", "summary": "没有 checkpoint，请先训练"}
        from train.tools.eval_tool import eval_tool
        result = eval_tool(self.config, ckpt)
        self.context["eval_results"] = [result]
        return result

    def _tune(self, args: dict) -> dict:
        """Run tuning loop synchronously."""
        num = int(args.get("num_iterations", 3))
        self.orch.agent_call("MainAgent", "TuningAgent", "tune", {"num_iterations": num})
        from agents.tuning_agent import TuningAgent
        agent = TuningAgent(orch_logger=self.orch, api_client=self.client,
                            model=self.model, output_dir=self.config.data.output_dir)
        result = agent.run(self.context, num_iterations=num)
        self.orch.agent_result("TuningAgent", result)
        return result

    def _model(self, args: dict) -> dict:
        """Run ModelAgent synchronously."""
        if args.get("reference_doc"):
            self._load_ref({"path": args["reference_doc"]})
        user_request = args.get("request", "Design an improved drug-target interaction model")
        self.context["user_request"] = user_request
        self.orch.agent_call("MainAgent", "ModelAgent", "generate_model", {"request": user_request})
        from agents.model_agent import ModelAgent
        agent = ModelAgent(
            output_dir=self.config.data.output_dir,
            openai_client=self.client,
            openai_model=self.model,
        )
        result = agent.run(self.context)
        self.orch.agent_result("ModelAgent", result)
        return result

    def _benchmark(self, args: dict) -> dict:
        if not self.context.get("checkpoint"):
            return {"status": "error", "summary": "没有 checkpoint，请先训练"}
        from agents.benchmark_agent import BenchmarkAgent
        return BenchmarkAgent().run(self.context)

    def _data(self, args: dict) -> dict:
        from agents.data_agent import DataAgent
        result = DataAgent().run(self.context)
        self.context["data_stats"] = result.get("data", {})
        return result

    def _status(self, args: dict) -> dict:
        c = self.config
        ctx = self.context
        return {
            "status": "ok",
            "data": {
                "data_dir": c.data.train_data_dir,
                "output_dir": c.data.output_dir,
                "device": c.device,
                "model": {
                    "mol_layers": c.model.mol.encoder_layers,
                    "mol_dim": c.model.mol.encoder_embed_dim,
                    "mol_heads": c.model.mol.encoder_attention_heads,
                    "mol_dropout": c.model.mol.dropout,
                    "pocket_layers": c.model.pocket.encoder_layers,
                    "pocket_dim": c.model.pocket.encoder_embed_dim,
                    "pocket_heads": c.model.pocket.encoder_attention_heads,
                    "project_dim": c.model.project_dim,
                    "temperature": c.model.temperature,
                },
                "training": {
                    "batch_size": c.train.batch_size,
                    "lr": c.train.lr,
                    "epochs": c.train.epochs,
                    "weight_decay": c.train.weight_decay,
                    "warmup_epochs": c.train.warmup_epochs,
                },
                "checkpoint": ctx.get("checkpoint"),
                "n_train_runs": len(ctx.get("train_results", [])),
                "n_eval_runs": len(ctx.get("eval_results", [])),
                "reference_docs": list(ctx.get("reference_docs", {}).keys()),
                "has_loss_data": ctx.get("loss_data") is not None,
            },
            "summary": f"Model {c.model.mol.encoder_layers}L×{c.model.mol.encoder_embed_dim}d, "
                       f"lr={c.train.lr:.2e}, ckpt={'yes' if ctx.get('checkpoint') else 'no'}",
        }

    def _set_config(self, args: dict) -> dict:
        key, value = args["key"], args["value"]
        self._apply_config(key, value)
        return {"status": "ok", "summary": f"{key} = {value}", "data": {"key": key, "value": value}}

    def _load_ref(self, args: dict) -> dict:
        path = Path(args["path"])
        if not path.exists():
            return {"status": "error", "summary": f"文件不存在: {path}"}
        content = path.read_text(encoding="utf-8")
        self.context.setdefault("reference_docs", {})[str(path)] = content
        return {"status": "ok", "summary": f"已加载: {path} ({len(content)} 字符)",
                "data": {"path": str(path), "size": len(content)}}

    def _auto(self, args: dict) -> dict:
        """Run auto-optimize synchronously."""
        max_iter = int(args.get("max_iterations", 5))
        from agents.main_agent import MainAgent
        self.context.setdefault("history", [])
        MainAgent.MAX_ITERATIONS = max_iter
        main = MainAgent(logger=self.logger)
        return main.run_auto(self.context)

    def _apply_config(self, key: str, value):
        parts = key.split(".")
        obj = self.config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        current = getattr(obj, parts[-1])
        if isinstance(current, int):
            setattr(obj, parts[-1], int(float(value)))
        elif isinstance(current, float):
            setattr(obj, parts[-1], float(value))
        else:
            setattr(obj, parts[-1], value)

    def _log(self, msg: str):
        if self.logger:
            self.logger.log(msg)
        else:
            print(msg)

    # ── UI helpers ────────────────────────────────────────────────

    def _print_banner(self):
        """Rich startup banner with config summary and capability overview."""
        c = self.config
        m = c.model
        t = c.train

        print()
        print("═" * 60)
        print("  DrugCLIP Odyssey — 高通量虚拟筛选优化智能体")
        print("═" * 60)
        print(f"  LLM      : {self.model}")
        print(f"  设备      : {c.device}")
        print(f"  训练数据  : {c.data.train_data_dir}")
        print(f"  评测数据  : {c.data.benchmark_dir}")
        print(f"  输出目录  : {c.data.output_dir}")
        print("─" * 60)
        print(f"  模型配置  : {m.mol.encoder_layers} 层 × {m.mol.encoder_embed_dim} 维")
        print(f"             Mol {m.mol.encoder_attention_heads} heads | Pocket {m.pocket.encoder_attention_heads} heads")
        print(f"             project_dim={m.project_dim}  temperature={m.temperature}")
        print(f"             Mol dropout={m.mol.dropout}  Pocket dropout={m.pocket.dropout}")
        print(f"  训练配置  : batch={t.batch_size}  lr={t.lr:.2e}  epochs={t.epochs}")
        print(f"             weight_decay={t.weight_decay}  warmup={t.warmup_epochs}")
        print(f"             lr_scheduler={t.lr_scheduler}  grad_clip={t.grad_clip}")
        print(f"             mixed_precision={t.mixed_precision}")
        print("═" * 60)
        ckpt = self.context.get("checkpoint")
        n_train = len(self.context.get("train_results", []))
        n_eval = len(self.context.get("eval_results", []))
        print(f"  状态      : 训练 {n_train} 次 | 评估 {n_eval} 次 | "
              f"Checkpoint: {'✓ ' + str(Path(ckpt).name) if ckpt else '无'}")
        print("═" * 60)
        print()

    def _print_quick_help(self):
        """Print available commands and example workflows."""
        print()
        print("  ── 可用能力 ──")
        print(f"  {'命令':<20} {'说明':<42}")
        print(f"  {'─'*20} {'─'*42}")
        for cmd, desc in [
            ("auto / 自动优化", "一键全流程：训练→评估→调参→迭代→提交"),
            ("train / 训练",     "训练 DrugCLIP 对比学习模型"),
            ("eval / 评估",       "运行验证集评估 + badcase 分析"),
            ("tune / 调参",       "LLM 驱动超参数自动优化循环"),
            ("model / 改架构",    "设计新模型架构，自动生成代码并注册"),
            ("benchmark / 提交",  "跑 117 任务推理 → 生成 result.zip"),
            ("loss / loss分析",    "分析训练 loss 曲线 + 优化建议"),
            ("data / 数据检查",   "检查数据格式与统计信息"),
        ]:
            print(f"  {cmd:<20} {desc:<42}")

        print()
        print("  ── 快速操作 ──")
        print(f"  {'命令':<25} {'说明':<35}")
        print(f"  {'─'*25} {'─'*35}")
        for cmd, desc in [
            ("status / st / 状态",  "查看当前模型配置与训练状态"),
            ("set train.lr 1e-4",   "修改学习率"),
            ("set train.batch_size 64", "修改批次大小"),
            ("set model.mol.encoder_layers 12", "修改编码器层数"),
            ("set model.temperature 0.1", "修改温度参数"),
            ("help / h / ?",        "显示本帮助"),
            ("exit / q / 退出",     "退出"),
        ]:
            print(f"  {cmd:<25} {desc:<35}")

        print()
        print("  ── 推荐工作流 ──")
        print("  1. 快速上手 : data → train → eval → loss")
        print("  2. 自动优化 : auto（一键端到端）")
        print("  3. 手动调参 : train → eval → loss → set → train → eval")
        print("  4. 改架构   : model → train → eval → benchmark")
        print("  5. 只改参数 : set → train → eval（反复迭代）")
        print()

    def _print_status(self):
        """Enhanced status display with training history."""
        c = self.config
        m = c.model
        t = c.train
        ctx = self.context

        print()
        print("  ══ 当前状态 ══")
        print(f"  数据目录    : {c.data.train_data_dir}")
        print(f"  评测目录    : {c.data.benchmark_dir}")
        print(f"  输出目录    : {c.data.output_dir}")
        print(f"  设备        : {c.device}  |  Seed: {c.seed}")
        print()

        # Model summary
        print(f"  ── 模型: {c.model.model_name} ──")
        print(f"  Mol Encoder    : {m.mol.encoder_layers}L × {m.mol.encoder_embed_dim}d"
              f" × {m.mol.encoder_attention_heads} heads"
              f"  ffn={m.mol.encoder_ffn_embed_dim}  dropout={m.mol.dropout}")
        print(f"  Pocket Encoder : {m.pocket.encoder_layers}L × {m.pocket.encoder_embed_dim}d"
              f" × {m.pocket.encoder_attention_heads} heads"
              f"  ffn={m.pocket.encoder_ffn_embed_dim}  dropout={m.pocket.dropout}")
        print(f"  Projection     : {m.project_dim}d  |  Temperature: {m.temperature}")
        print()

        # Training config
        print(f"  ── 训练配置 ──")
        print(f"  batch_size={t.batch_size}  lr={t.lr:.2e}  epochs={t.epochs}")
        print(f"  weight_decay={t.weight_decay}  warmup_epochs={t.warmup_epochs}")
        print(f"  lr_scheduler={t.lr_scheduler}  grad_clip={t.grad_clip}")
        print(f"  mixed_precision={t.mixed_precision}")
        print()

        # Runtime state
        ckpt = ctx.get("checkpoint")
        tr_list = ctx.get("train_results", [])
        ev_list = ctx.get("eval_results", [])
        print(f"  ── 运行状态 ──")
        print(f"  Checkpoint    : {ckpt or '无'}")
        print(f"  训练次数      : {len(tr_list)}")
        print(f"  评估次数      : {len(ev_list)}")
        print(f"  参考文档      : {list(ctx.get('reference_docs', {}).keys()) or '无'}")

        # Recent training
        if tr_list:
            last = tr_list[-1]
            if last.get("status") == "ok":
                d = last["data"]
                print(f"  最近训练      : best_loss={d.get('best_loss', '?'):.4f}"
                      f" @ epoch {d.get('best_epoch', '?')}"
                      f"  ({d.get('epochs_trained', '?')} epochs total)")
        print()

    def _suggest_next(self, tool_name: str, result: dict) -> str:
        """Suggest next action based on tool result and current context."""
        ok = result.get("status") == "ok"
        has_ckpt = bool(self.context.get("checkpoint"))
        has_eval = bool(self.context.get("eval_results"))

        suggestions = {
            "check_data": lambda: (
                "数据检查通过 ✓  下一步建议: 'train' 开始训练  "
                "或 'auto' 一键自动优化"
                if ok else "数据检查失败，请确认数据路径和格式"
            ),
            "train_model": lambda: (
                "训练完成 ✓  下一步建议: 'eval' 查看验证集表现  "
                "或 'loss' 分析 loss 趋势"
                if ok else "训练失败，可以 'data' 检查数据 或 'status' 确认配置"
            ),
            "run_evaluation": lambda: (
                "评估完成 ✓  下一步建议: 'loss' 分析收敛趋势  "
                "或 'tune 2' 启动调参优化  "
                "或 'benchmark' 跑最终评测"
                if ok else "评估失败，确认 checkpoint 存在: 'status'"
            ),
            "analyze_loss": lambda: (
                "Loss 分析完成 ✓  根据建议决定: 'tune' 自动调参  "
                "或 'set <key> <value>' 手动修改参数"
                if ok else "Loss 分析失败，确认已完成训练"
            ),
            "tune_hyperparams": lambda: (
                "调参完成 ✓  下一步建议: 'train' 验证新参数  "
                "或不满意继续 'tune'"
                if ok else "调参未完成"
            ),
            "modify_architecture": lambda: (
                "新模型已注册 ✓  下一步建议: 'train' 用新模型训练  "
                "→ 'eval' 对比改善"
                if ok else "架构修改失败，检查参考文档和请求描述"
            ),
            "run_benchmark": lambda: (
                "Benchmark 推理完成 ✓  result.zip 已生成  "
                "可提交至比赛平台！"
                if ok else "Benchmark 推理失败，确认 checkpoint 和 benchmark 目录"
            ),
            "auto_optimize": lambda: (
                "自动优化完成 ✓  result.zip 已生成，可直接提交！"
                if ok else "自动优化中断，可以 'status' 查看进度后继续"
            ),
        }

        fn = suggestions.get(tool_name)
        if fn:
            return fn()
        return ""
