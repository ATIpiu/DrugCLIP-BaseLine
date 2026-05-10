"""DrugCLIP Odyssey — Competition-Compliant result.log.

Records the full autonomous optimization process:
  - Strategy design & iteration plan
  - Training / finetuning with config snapshots
  - Agent decisions with hypothesis + rationale + before/after comparison
  - Evaluation results at each iteration
  - Inference / ranking strategy
  - Final submission generation

The log must prove the agent genuinely performed autonomous optimization,
not manual one-shot tuning.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List


class OdysseyLogger:
    """Competition-grade structured logger for multi-agent optimization."""

    def __init__(self, output_dir: str = "output", silent: bool = False):
        import os as _os
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "result.log"
        self._start_time = time.time()
        self._iteration = 0
        self._optimization_history: List[dict] = []
        self._silent = silent or _os.environ.get("DRUGCLIP_SILENT") == "1"

        # Header
        self._write("=" * 70)
        self._write("DrugCLIP Odyssey — Autonomous Virtual Screening Optimization")
        self._write(f"Start: {datetime.now():%Y-%m-%d %H:%M:%S}")
        self._write("=" * 70)

    # ── Core ─────────────────────────────────────────────────────────

    def _write(self, text: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {text}"
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if not self._silent:
            print(line)

    def _section(self, title: str):
        self._write("")
        self._write("-" * 60)
        self._write(f"  {title}")
        self._write("-" * 60)

    def _kv(self, key: str, value, indent: int = 2):
        prefix = " " * indent
        self._write(f"{prefix}{key}: {value}")

    def _json(self, data: dict, indent: int = 2):
        for k, v in data.items():
            if isinstance(v, float):
                self._kv(k, f"{v:.6f}" if abs(v) < 0.01 else f"{v:.4f}", indent)
            elif isinstance(v, dict):
                self._kv(k, "", indent)
                self._json(v, indent + 2)
            else:
                self._kv(k, v, indent)

    # ── Strategy ─────────────────────────────────────────────────────

    def log_strategy(self, plan: str, config_snapshot: dict = None):
        """Log the agent's overall optimization strategy.

        Args:
            plan: human-readable strategy description
            config_snapshot: optional dict of initial key config values
        """
        self._section("Optimization Strategy")
        self._write(f"  {plan}")
        if config_snapshot:
            self._write("  Initial Configuration:")
            self._json(config_snapshot, indent=4)

    # ── Baseline ─────────────────────────────────────────────────────

    def log_baseline(self, config, train_result: dict, eval_result: dict = None):
        """Log the baseline model before any optimization.

        Args:
            config: the Config object
            train_result: result dict from train_tool
            eval_result: optional result dict from eval_tool
        """
        self._section("Baseline Model (Iteration 0)")

        # Config
        self._write("  Model Configuration:")
        m = config.model
        self._kv("encoder_layers", f"mol={m.mol.encoder_layers}, pocket={m.pocket.encoder_layers}", 4)
        self._kv("encoder_dim", f"mol={m.mol.encoder_embed_dim}, pocket={m.pocket.encoder_embed_dim}", 4)
        self._kv("attention_heads", f"mol={m.mol.encoder_attention_heads}, pocket={m.pocket.encoder_attention_heads}", 4)
        self._kv("dropout", f"mol={m.mol.dropout}, pocket={m.pocket.dropout}", 4)
        self._kv("temperature", m.temperature, 4)
        self._kv("project_dim", m.project_dim, 4)
        self._write("  Training Configuration:")
        t = config.train
        self._kv("batch_size", t.batch_size, 4)
        self._kv("lr", t.lr, 4)
        self._kv("epochs", t.epochs, 4)
        self._kv("weight_decay", t.weight_decay, 4)
        self._kv("warmup_epochs", t.warmup_epochs, 4)

        # Training result
        if train_result.get("status") == "ok":
            d = train_result["data"]
            self._write("  Training Result:")
            self._kv("status", "OK", 4)
            self._kv("best_loss", f"{d['best_loss']:.4f}", 4)
            self._kv("best_epoch", d["best_epoch"], 4)
            if d.get("per_epoch_metrics"):
                first, last = d["per_epoch_metrics"][0], d["per_epoch_metrics"][-1]
                self._kv("loss_trend", f"{first['train_loss']:.4f} → {last['train_loss']:.4f}", 4)
            self._kv("checkpoint", d.get("checkpoint_path", ""), 4)
        else:
            self._write("  Training Result: FAILED")
            self._kv("error", train_result.get("error", ""), 4)

        # Eval result
        if eval_result and eval_result.get("status") == "ok":
            m = eval_result["data"].get("overall_metrics", {})
            self._write("  Evaluation Result:")
            self._kv("AUROC", f"{m.get('auroc', 0):.4f}", 4)
            self._kv("EF1%", f"{m.get('ef1', 0):.1f}", 4)
            self._kv("EF5%", f"{m.get('ef5', 0):.1f}", 4)
            self._kv("MRR", f"{m.get('mrr', 0):.4f}", 4)
            self._kv("Top-1", f"{m.get('top1', 0):.3f}", 4)
            self._kv("Top-5", f"{m.get('top5', 0):.3f}", 4)
            self._kv("BEDROC", f"{m.get('bedroc', 0):.4f}", 4)
            # Badcase summary
            bc = eval_result["data"].get("badcase_analysis", {})
            if bc.get("distribution"):
                dist = bc["distribution"]
                self._write("  Badcase Distribution:")
                self._kv("AUROC mean", f"{dist.get('auroc_mean', 0):.4f}", 4)
                self._kv("AUROC std", f"{dist.get('auroc_std', 0):.4f}", 4)
                self._kv("AUROC p10", f"{dist.get('auroc_p10', 0):.4f}", 4)
                self._kv("AUROC p50", f"{dist.get('auroc_p50', 0):.4f}", 4)
            if bc.get("badcases"):
                self._write("  Top Badcases:")
                for b in bc["badcases"][:5]:
                    self._kv(f"sample_{b['sample_idx']}",
                             f"AUROC={b['metrics'].get('auroc',0):.3f} | {b.get('hypothesis','')}", 4)

    # ── Optimization iteration ───────────────────────────────────────

    def log_iteration_start(self, iteration: int, hypothesis: str, changes: dict, rationale: str):
        """Log the start of an optimization iteration.

        Args:
            iteration: 1-indexed iteration number
            hypothesis: what the agent hypothesizes will improve performance
            changes: dict of {param_path: new_value} being applied
            rationale: why these changes, referencing specific data
        """
        self._iteration = iteration
        self._section(f"Optimization Iteration {iteration}")
        self._write(f"  Hypothesis: {hypothesis}")
        self._write(f"  Rationale: {rationale}")
        self._write("  Changes Applied:")
        for k, v in changes.items():
            self._kv(k, v, 4)

    def log_iteration_train(self, config, train_result: dict):
        """Log training result within an iteration."""
        if train_result.get("status") == "ok":
            d = train_result["data"]
            self._write("  Training:")
            self._kv("best_loss", f"{d['best_loss']:.4f}", 4)
            self._kv("best_epoch", d["best_epoch"], 4)
            if d.get("per_epoch_metrics"):
                first, last = d["per_epoch_metrics"][0], d["per_epoch_metrics"][-1]
                self._kv("loss_trend", f"{first['train_loss']:.4f} → {last['train_loss']:.4f}", 4)
        else:
            self._write("  Training: FAILED")

    def log_iteration_eval(self, eval_result: dict, baseline_metrics: dict = None):
        """Log evaluation result within an iteration, with optional comparison.

        Args:
            eval_result: result dict from eval_tool
            baseline_metrics: dict of baseline metrics for delta calculation
        """
        if eval_result.get("status") != "ok":
            self._write("  Evaluation: FAILED")
            return

        m = eval_result["data"].get("overall_metrics", {})
        self._write("  Evaluation:")
        cur_auroc = m.get("auroc", 0)
        cur_ef1 = m.get("ef1", 0)
        self._kv("AUROC", f"{cur_auroc:.4f}", 4)
        self._kv("EF1%", f"{cur_ef1:.1f}", 4)
        self._kv("EF5%", f"{m.get('ef5', 0):.1f}", 4)
        self._kv("MRR", f"{m.get('mrr', 0):.4f}", 4)
        self._kv("Top-1", f"{m.get('top1', 0):.3f}", 4)
        self._kv("Top-5", f"{m.get('top5', 0):.3f}", 4)
        self._kv("BEDROC", f"{m.get('bedroc', 0):.4f}", 4)

        # Before/after comparison
        if baseline_metrics:
            self._write("  Improvement vs Baseline:")
            delta_auroc = cur_auroc - baseline_metrics.get("auroc", cur_auroc)
            delta_ef1 = cur_ef1 - baseline_metrics.get("ef1", cur_ef1)
            self._kv("Δ AUROC", f"{delta_auroc:+.4f}", 4)
            self._kv("Δ EF1%", f"{delta_ef1:+.1f}", 4)

        # Badcase
        bc = eval_result["data"].get("badcase_analysis", {})
        if bc.get("badcases"):
            self._write("  Badcases:")
            for b in bc["badcases"][:3]:
                self._kv(f"sample_{b['sample_idx']}",
                         f"AUROC={b['metrics'].get('auroc',0):.3f} | {b.get('hypothesis','')}", 4)

    # ── Architecture change ──────────────────────────────────────────

    def log_architecture_change(self, changes: dict, rationale: str, references: list = None):
        """Log a model architecture modification decision."""
        self._section("Architecture Modification")
        self._write("  Changes:")
        self._json(changes, indent=4)
        self._write(f"  Rationale: {rationale}")
        if references:
            self._write(f"  References: {references}")

    # ── Final ────────────────────────────────────────────────────────

    def log_optimization_summary(self, history: list, best_metrics: dict, best_config: dict):
        """Log the complete optimization trajectory summary.

        Args:
            history: list of per-iteration dicts with keys:
                {iteration, hypothesis, changes, auroc_before, auroc_after,
                 ef1_before, ef1_after}
            best_metrics: final best metrics dict
            best_config: final best config snapshot dict
        """
        self._section("Optimization Trajectory Summary")

        if history:
            self._write("  Iteration History:")
            self._write(f"  {'Iter':<5} {'Hypothesis':<40} {'AUROC Δ':>8} {'EF1% Δ':>8}")
            self._write(f"  {'-'*5} {'-'*40} {'-'*8} {'-'*8}")
            for h in history:
                hyp = h.get("hypothesis", "")[:38]
                a_delta = h.get("auroc_delta", 0)
                e_delta = h.get("ef1_delta", 0)
                self._write(f"  {h.get('iteration', '?'):<5} {hyp:<40} {a_delta:>+8.4f} {e_delta:>+8.1f}")

        self._write("")
        self._write("  Final Best Configuration:")
        self._json(best_config, indent=4)
        self._write("")
        self._write("  Final Best Metrics:")
        self._kv("AUROC", f"{best_metrics.get('auroc', 0):.4f}", 4)
        self._kv("EF1%", f"{best_metrics.get('ef1', 0):.1f}", 4)
        self._kv("EF5%", f"{best_metrics.get('ef5', 0):.1f}", 4)
        self._kv("MRR", f"{best_metrics.get('mrr', 0):.4f}", 4)

    def log_inference_strategy(self, strategy: str, details: dict = None):
        """Log the inference/ranking strategy used for final submission."""
        self._section("Inference Strategy")
        self._write(f"  {strategy}")
        if details:
            self._json(details, indent=4)

    def log_submission(self, result_csv: str, zip_path: str, total_tasks: int,
                       total_ligands: int, total_time: float):
        """Log final submission generation."""
        self._section("Submission Generation")
        self._kv("result.csv", result_csv, 2)
        self._kv("result.zip", zip_path, 2)
        self._kv("total_tasks", total_tasks, 2)
        self._kv("total_ligands", total_ligands, 2)
        self._kv("total_time", f"{total_time:.1f}s ({total_time/3600:.2f}h)", 2)
        self._write("")
        self._write(f"End: {datetime.now():%Y-%m-%d %H:%M:%S}")
        self._write(f"Status: OPTIMIZATION COMPLETE")
        self._write("=" * 70)

    # ── Legacy compatibility ─────────────────────────────────────────

    def log(self, message: str):
        """Simple message log (for non-agent mode)."""
        self._write(message)

    def log_section(self, title: str):
        self._section(title)

    def log_config(self, config):
        """Quick config dump (used by Trainer)."""
        self._section("Training Configuration")
        m = config.model
        self._kv("encoder_layers", f"mol={m.mol.encoder_layers}, pocket={m.pocket.encoder_layers}")
        self._kv("encoder_dim", f"mol={m.mol.encoder_embed_dim}, pocket={m.pocket.encoder_embed_dim}")
        self._kv("attention_heads", f"mol={m.mol.encoder_attention_heads}, pocket={m.pocket.encoder_attention_heads}")
        self._kv("gbf_k", m.gbf_k)
        self._kv("project_dim", m.project_dim)
        self._kv("temperature", m.temperature)
        self._kv("batch_size", config.train.batch_size)
        self._kv("lr", config.train.lr)
        self._kv("epochs", config.train.epochs)
        self._kv("seed", config.seed)

    def log_agent_action(self, agent_name: str, decision: dict):
        """Log agent decision."""
        self._section(f"Agent Decision: {agent_name}")
        if decision.get("hypothesis"):
            self._kv("Hypothesis", decision["hypothesis"])
        if decision.get("rationale"):
            self._kv("Rationale", decision["rationale"])
        if decision.get("action"):
            self._kv("Action", json.dumps(decision["action"], default=str, ensure_ascii=False))
        if decision.get("reference_docs"):
            self._kv("References", str(decision["reference_docs"]))

    def log_tool_result(self, tool_name: str, result: dict):
        pass  # Handled by iteration methods above

    def log_iteration_boundary(self, iteration: int):
        self._section(f"Iteration {iteration}")

    def log_inference_progress(self, task_id: str, num_ligands: int, elapsed: float):
        self._write(f"  {task_id}: {num_ligands} ligands, {elapsed:.1f}s")

    def log_final_summary(self, total_tasks: int, total_ligands: int, total_time: float,
                          model_path: Optional[str] = None):
        """Legacy final summary (used by inference.py)."""
        self._section("Inference Complete")
        self._kv("total_tasks", total_tasks)
        self._kv("total_ligands", total_ligands)
        self._kv("total_time", f"{total_time:.1f}s ({total_time/3600:.2f}h)")
        if model_path:
            self._kv("model", model_path)
        self._write(f"End: {datetime.now():%Y-%m-%d %H:%M:%S}")
