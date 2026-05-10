"""MainAgent — interactive REPL orchestrating all sub-agents and tools.

Usage:
    agent = MainAgent(logger=logger)
    agent.run_interactive(context)   # REPL mode
    agent.run_auto(context)          # automated pipeline (backward compat)
"""

import copy
import json
import re
import sys
from pathlib import Path

from .base_agent import BaseAgent
from .data_agent import DataAgent
from .tuning_agent import TuningAgent
from .model_agent import ModelAgent
from .benchmark_agent import BenchmarkAgent


# ── Command dispatch table ────────────────────────────────────────

COMMANDS = {
    "train":    ("训练", "开始训练"),
    "eval":     ("评估", "验证", "跑eval"),
    "tune":     ("调参", "调整参数", "超参优化"),
    "model":    ("改模型", "修改架构", "换模型"),
    "benchmark":("提交", "跑分", "生成提交", "benchmark"),
    "loss":     ("loss分析", "loss曲线", "loss"),
    "data":     ("数据检查", "数据"),
    "status":   ("状态", "当前状态", "上下文"),
    "config":   ("配置", "show config", "查看配置"),
    "auto":     ("自动", "一键优化", "auto optimize"),
    "help":     ("帮助", "?", "h"),
}

EXIT_KEYWORDS = ("exit", "退出", "quit", "q")


class MainAgent(BaseAgent):
    """Interactive DrugCLIP agent orchestrator.

    In REPL mode, accepts natural language commands and dispatches to
    the appropriate sub-agent or tool. Context persists across commands.
    """

    MAX_ITERATIONS = 10

    def __init__(self, logger=None):
        super().__init__()
        self.logger = logger
        self.context: dict = {}
        self._running = False

    # ── REPL mode ──────────────────────────────────────────────────

    def run_interactive(self, context: dict):
        """Start interactive REPL loop.

        Args:
            context: initial context dict with at least {"config": Config}
        """
        self.context = context
        self.context.setdefault("history", [])
        self.context.setdefault("reference_docs", {})
        self.context.setdefault("checkpoint", None)

        self._log("DrugCLIP Odyssey — Interactive Agent")
        self._log(f"  Data: {context['config'].data.train_data_dir}")
        self._log(f"  Device: {context['config'].device}")
        self._log(f"  Model: {context['config'].model.mol.encoder_layers} layers, "
                  f"{context['config'].model.mol.encoder_embed_dim} dim")
        self._log("  Type 'help' for commands, 'exit' to quit")

        self._running = True
        while self._running:
            try:
                raw = input("\n>>> ").strip()
            except (EOFError, KeyboardInterrupt):
                self._log("Exiting.")
                break

            if not raw:
                continue

            cmd, args = self._parse(raw)

            if cmd in EXIT_KEYWORDS:
                self._log("Goodbye.")
                break

            if cmd == "help":
                self._show_help()
            elif cmd == "status":
                self._show_status()
            elif cmd == "config":
                self._show_config(args)
            elif cmd == "auto":
                self._log("Starting automated optimization pipeline...")
                self.run_auto(self.context)
            elif cmd == "set":
                self._handle_set(args)
            elif cmd == "ref":
                self._handle_ref(args)
            elif cmd in COMMANDS:
                self._dispatch(cmd)
            else:
                self._log(f"Unknown: '{raw}'. Type 'help' for commands.")

        return {"status": "ok", "data": self.context, "summary": "Session ended"}

    # ── Automated pipeline mode (backward compat) ──────────────────

    def run(self, context: dict) -> dict:
        """Automated pipeline — kept for programmatic use and --mode agent."""
        return self.run_auto(context)

    def run_auto(self, context: dict) -> dict:
        """Full automated optimization pipeline."""
        config = context["config"]
        self.context = context
        self._log_section("Phase 1: Data Verification")
        data_result = DataAgent().run(context)
        context["data_stats"] = data_result.get("data", {})
        self._log_tool("DataAgent", data_result)

        if data_result["status"] != "ok":
            return {"status": "error", "data": {}, "summary": "Data verification failed"}

        self._log_section("Phase 2: Optimization Loop")
        context.setdefault("history", [])

        for iteration in range(self.MAX_ITERATIONS):
            context["iteration"] = iteration
            self._log_boundary(iteration)

            # Tune
            tuning_result = TuningAgent().run(context)
            tuning_decision = {}
            if tuning_result.get("decisions"):
                for d in tuning_result["decisions"]:
                    self._log_agent("TuningAgent", d)
                    tuning_decision = d

            # Train
            from train.tools.train_tool import train_tool
            train_result = train_tool(config)
            context["train_results"] = [train_result]
            if train_result["status"] == "ok":
                context["checkpoint"] = train_result["data"]["checkpoint_path"]
            self._log_tool("train_tool", train_result)
            if train_result["status"] != "ok":
                break

            # Log per-epoch training metrics in detail
            per_epoch = train_result["data"].get("per_epoch_metrics", [])
            if per_epoch and self.logger:
                first, last = per_epoch[0], per_epoch[-1]
                self._log(f"  Training: {len(per_epoch)} epochs, "
                          f"Loss {first['train_loss']:.4f}→{last['train_loss']:.4f}")
                if "val_ef1" in last:
                    self._log(f"  Val: EF1 {first.get('val_ef1',0):.3f}→{last['val_ef1']:.3f}  "
                              f"AUROC {first.get('val_auroc',0):.3f}→{last['val_auroc']:.3f}")

            # Loss
            from train.tools.loss_tool import loss_tool
            loss_result = loss_tool(per_epoch)
            context["loss_data"] = loss_result
            self._log_tool("loss_tool", loss_result)

            # Eval (every 3 iters)
            if iteration % 3 == 0 or iteration == self.MAX_ITERATIONS - 1:
                from train.tools.eval_tool import eval_tool
                eval_result = eval_tool(config, context.get("checkpoint"))
                context["eval_results"] = [eval_result]
                self._log_tool("eval_tool", eval_result)

            # ModelAgent (every 3 iters)
            if iteration % 3 == 2 or iteration == self.MAX_ITERATIONS - 1:
                model_result = ModelAgent().run(context)
                if model_result.get("decisions"):
                    for d in model_result["decisions"]:
                        self._log_agent("ModelAgent", d)

            snapshot = {
                "iteration": iteration,
                "config_snapshot": copy.deepcopy(config),
                "train_result": train_result,
                "loss_result": loss_result,
            }
            context["history"].append(snapshot)

            if self._check_convergence(context):
                self._log(f"Convergence detected at iteration {iteration}")
                break

        self._log_section("Phase 3: Benchmark Submission")
        benchmark_result = BenchmarkAgent().run(context)
        self._log_tool("BenchmarkAgent", benchmark_result)

        # Final trajectory summary
        history = context.get("history", [])
        if history and self.logger:
            self._log("")
            self._log("  ── Optimization Trajectory ──")
            self._log(f"  {'Iter':<5} {'Best Loss':>10} {'Val AUROC':>10} {'Status':>12}")
            self._log(f"  {'─'*5} {'─'*10} {'─'*10} {'─'*12}")
            for h in history:
                tr = h.get("train_result", {})
                d = tr.get("data", {}) if tr.get("status") == "ok" else {}
                per_epoch = d.get("per_epoch_metrics", [])
                last = per_epoch[-1] if per_epoch else {}
                loss = f"{d.get('best_loss', 0):.4f}"
                auroc = f"{last.get('val_auroc', 0):.3f}" if "val_auroc" in last else "N/A"
                status = "OK" if tr.get("status") == "ok" else "FAIL"
                self._log(f"  {h['iteration']:<5} {loss:>10} {auroc:>10} {status:>12}")
            self._log("")

        final = context["train_results"][-1] if context["train_results"] else {}
        self._log_section("Pipeline Complete")
        if final.get("status") == "ok":
            d = final["data"]
            self._log(f"  Best Loss: {d['best_loss']:.4f} (epoch {d.get('best_epoch', '?')})")
            self._log(f"  Epochs trained: {d.get('epochs_trained', '?')}")
        sub_path = benchmark_result.get('data', {}).get('submission_path', 'N/A')
        self._log(f"  Submission: {sub_path}")
        self._log(f"  Total iterations: {len(history)}")

        return {
            "status": "ok",
            "data": {
                "iterations": len(context["history"]),
                "checkpoint": context.get("checkpoint"),
                "submission": benchmark_result.get("data", {}).get("submission_path"),
            },
            "summary": f"Completed {len(context['history'])} iterations",
        }

    # ── Command dispatch ───────────────────────────────────────────

    def _dispatch(self, cmd: str):
        """Route command to appropriate sub-agent or tool."""
        config = self.context["config"]
        self._log_section(f"Command: {cmd}")

        if cmd == "train":
            from train.tools.train_tool import train_tool
            result = train_tool(config)
            if result["status"] == "ok":
                self.context["checkpoint"] = result["data"]["checkpoint_path"]
                self.context["train_results"] = [result]
            self._log_tool("train_tool", result)

        elif cmd == "eval":
            from train.tools.eval_tool import eval_tool
            ckpt = self.context.get("checkpoint")
            if not ckpt:
                self._log("  No checkpoint found. Train first.")
                return
            result = eval_tool(config, ckpt)
            self.context["eval_results"] = [result]
            self._log_tool("eval_tool", result)
            # Show badcase summary
            if result["status"] == "ok":
                bc = result["data"].get("badcase_analysis", {})
                if bc.get("badcases"):
                    self._log(f"  --- Badcases (worst {len(bc['badcases'])} samples) ---")
                    for b in bc["badcases"][:5]:
                        self._log(f"  sample_{b['sample_idx']}: "
                                  f"AUROC={b['metrics'].get('auroc',0):.3f} | "
                                  f"hypothesis: {b['hypothesis'][:80]}")

        elif cmd == "loss":
            from train.tools.loss_tool import loss_tool
            train_results = self.context.get("train_results", [])
            if not train_results:
                self._log("  No training results. Train first.")
                return
            per_epoch = train_results[-1]["data"].get("per_epoch_metrics", [])
            result = loss_tool(per_epoch)
            self.context["loss_data"] = result
            self._log_tool("loss_tool", result)
            if result["status"] == "ok":
                for s in result["data"].get("suggestions", []):
                    self._log(f"  → {s}")

        elif cmd == "tune":
            if not self.context.get("history"):
                self._log("  No training history. Run 'train' first.")
                return
            result = TuningAgent().run(self.context)
            if result.get("decisions"):
                for d in result["decisions"]:
                    self._log_agent("TuningAgent", d)
            if result["status"] == "ok" and result["data"].get("config_updates"):
                self._log(f"  Config updated: {json.dumps(result['data']['config_updates'])}")

        elif cmd == "model":
            result = ModelAgent().run(self.context)
            if result.get("decisions"):
                for d in result["decisions"]:
                    self._log_agent("ModelAgent", d)
            if result["status"] == "ok" and result["data"].get("config_updates"):
                self._log(f"  Architecture updated: {json.dumps(result['data']['config_updates'])}")

        elif cmd == "benchmark":
            result = BenchmarkAgent().run(self.context)
            self._log_tool("BenchmarkAgent", result)

        elif cmd == "data":
            result = DataAgent().run(self.context)
            self.context["data_stats"] = result.get("data", {})
            self._log_tool("DataAgent", result)

    # ── Command parsing ────────────────────────────────────────────

    def _parse(self, raw: str) -> tuple:
        """Parse raw input into (command, args)."""
        raw_lower = raw.lower().strip()

        # Check for exact match or alias match
        for cmd_name, aliases in COMMANDS.items():
            if raw_lower == cmd_name or raw_lower in aliases:
                return cmd_name, ""

        # Check for 'set <key> <value>'
        if raw_lower.startswith("set "):
            return "set", raw[4:].strip()

        # Check for 'ref <path>'
        if raw_lower.startswith("ref ") or raw_lower.startswith("参考"):
            path = raw.split(maxsplit=1)[1] if len(raw.split()) > 1 else ""
            return "ref", path.strip()

        # Check exit keywords
        if raw_lower in EXIT_KEYWORDS:
            return raw_lower, ""

        # Fuzzy match: if raw contains a command keyword
        for cmd_name, aliases in COMMANDS.items():
            candidates = (cmd_name,) + aliases
            for c in candidates:
                if c in raw_lower:
                    return cmd_name, ""

        return raw_lower, ""

    # ── Handlers ───────────────────────────────────────────────────

    def _handle_set(self, args: str):
        """Update a config field: set train.lr 1e-4"""
        try:
            key, val = args.split()
            self._set_config_field(self.context["config"], key, val)
            self._log(f"  Config updated: {key} = {val}")
        except ValueError:
            self._log("  Usage: set <dotted.key> <value>")
            self._log("  e.g.: set train.lr 1e-4")
            self._log("  e.g.: set model.mol.encoder_layers 8")

    def _handle_ref(self, path: str):
        """Load a reference document into context."""
        if not path:
            self._log("  Usage: ref <path/to/doc.md>")
            return
        doc_path = Path(path)
        if not doc_path.exists():
            self._log(f"  File not found: {doc_path}")
            return
        with open(doc_path) as f:
            content = f.read()
        self.context["reference_docs"] = self.context.get("reference_docs", {})
        self.context["reference_docs"][str(doc_path)] = content
        self._log(f"  Loaded reference doc: {doc_path} ({len(content)} chars)")

    @staticmethod
    def _set_config_field(config, key: str, value: str):
        parts = key.split(".")
        obj = config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        current = getattr(obj, parts[-1])
        # Infer type from current value
        if isinstance(current, int):
            setattr(obj, parts[-1], int(float(value)))
        elif isinstance(current, float):
            setattr(obj, parts[-1], float(value))
        elif isinstance(current, bool):
            setattr(obj, parts[-1], value.lower() in ("true", "1", "yes"))
        else:
            setattr(obj, parts[-1], value)

    # ── Display ────────────────────────────────────────────────────

    def _show_help(self):
        self._log("")
        self._log("  === DrugCLIP Interactive Commands ===")
        self._log("  train      — 开始训练")
        self._log("  eval       — 运行评估 + badcase 分析")
        self._log("  loss       — 分析 loss 曲线")
        self._log("  tune       — 根据 loss/eval 自动调参")
        self._log("  model      — 修改模型架构")
        self._log("  benchmark  — 跑 117 task 推理 → 生成 result.zip")
        self._log("  data       — 检查数据目录/统计")
        self._log("  status     — 查看当前状态")
        self._log("  config     — 查看/修改配置")
        self._log("  set <key> <value> — 修改配置项，如: set train.lr 1e-4")
        self._log("  ref <path> — 加载参考文档给 ModelAgent")
        self._log("  auto       — 一键自动化优化流水线")
        self._log("  help       — 显示此帮助")
        self._log("  exit       — 退出")

    def _show_status(self):
        ctx = self.context
        config = ctx["config"]
        self._log("")
        self._log("  === Current Status ===")
        self._log(f"  Data dir: {config.data.train_data_dir}")
        self._log(f"  Output dir: {config.data.output_dir}")
        self._log(f"  Device: {config.device}")
        self._log(f"  Model: {config.model.mol.encoder_layers} layers × "
                  f"{config.model.mol.encoder_embed_dim} dim × "
                  f"{config.model.mol.encoder_attention_heads} heads")
        self._log(f"  Train: batch={config.train.batch_size}, lr={config.train.lr:.2e}, "
                  f"epochs={config.train.epochs}")
        self._log(f"  Checkpoint: {ctx.get('checkpoint', 'None')}")
        self._log(f"  Reference docs: {list(ctx.get('reference_docs', {}).keys())}")
        self._log(f"  History: {len(ctx.get('history', []))} iterations")
        self._log(f"  Train results: {len(ctx.get('train_results', []))}")
        self._log(f"  Eval results: {len(ctx.get('eval_results', []))}")

    def _show_config(self, args: str):
        if args:
            # Show specific key
            try:
                parts = args.split(".")
                obj = self.context["config"]
                for part in parts:
                    obj = getattr(obj, part)
                self._log(f"  {args} = {obj}")
            except AttributeError:
                self._log(f"  Unknown config key: {args}")
        else:
            self._show_status()

    # ── Convergence ────────────────────────────────────────────────

    def _check_convergence(self, context: dict) -> bool:
        history = context.get("history", [])
        if len(history) < 3:
            return False
        recent = history[-3:]
        losses = [h.get("train_result", {}).get("data", {}).get("best_loss", 0)
                  for h in recent]
        return all(abs(losses[i] - losses[i-1]) < 0.005 for i in range(1, len(losses)))

    # ── Logging helpers ────────────────────────────────────────────

    def _log(self, msg: str):
        if self.logger:
            self.logger.log(msg)
        else:
            print(msg)

    def _log_section(self, title: str):
        if self.logger:
            self.logger.log_section(title)

    def _log_boundary(self, iteration: int):
        if self.logger:
            self.logger.log_iteration_boundary(iteration)

    def _log_agent(self, name: str, decision: dict):
        if self.logger:
            self.logger.log_agent_action(name, decision)

    def _log_tool(self, name: str, result: dict):
        if self.logger:
            self.logger.log_tool_result(name, result)
