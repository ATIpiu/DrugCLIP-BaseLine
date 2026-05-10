"""TuningAgent — LLM-powered hyperparameter optimization loop.

Flow (per iteration):
  1. train_tool(config) → loss curve
  2. eval_tool(config, checkpoint) → eval metrics + badcase analysis
  3. LLM analyzes loss + eval → decides which params to change
  4. Apply changes → retrain → evaluate → compare → report improvement

No user interaction needed during the loop.
All decisions and results logged to orchestration log.
"""

import json
import os
import re
from pathlib import Path
from typing import Optional

# Load .env if present
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

from .base_agent import BaseAgent
from .orch_logger import OrchLogger
from .agent_logger import AgentLogger
from .llm_utils import call_llm_json


# ── System prompt for the tuning LLM ──────────────────────────────

TUNING_SYSTEM_PROMPT = """你是 DrugCLIP 模型的超参数调优专家。

## 重要：模型坍塌检测
如果 loss 极低（<0.1）但 AUROC 接近 0.5，说明模型坍塌为平凡解（输出恒定向量或捷径特征）。
此时必须：增大 temperature (如 0.07→0.2)、增大 dropout (如 0.1→0.3)、增大 weight_decay (如 1e-5→1e-4)
绝对不要在这种情况下增加模型容量（层数/维度）！更多参数会让坍塌更严重。

## 参数合理范围
- train.lr: 1e-5 ~ 3e-4
- train.batch_size: 16 ~ 64
- model.*.encoder_layers: 2 ~ 8 (不要超过8)
- model.*.encoder_embed_dim: 128 ~ 512
- model.temperature: 0.05 ~ 0.5 (越低越容易坍塌)
- model.*.dropout: 0.1 ~ 0.5

## 任务

## 你可调整的参数
- train.lr: 学习率 (如 3e-4, 1e-4, 5e-5)
- train.batch_size: 批次大小 (如 32, 64, 128)
- train.weight_decay: 权重衰减 (如 1e-5, 1e-4, 1e-3)
- train.warmup_epochs: 预热轮数 (如 3, 5, 10)
- model.mol.encoder_layers: 分子编码器层数 (2-15)
- model.mol.encoder_embed_dim: 分子编码器维度 (128-768)
- model.mol.encoder_attention_heads: 分子编码器注意力头数 (需整除 embed_dim)
- model.mol.dropout: 分子编码器 dropout (0.0-0.3)
- model.pocket.encoder_layers: 口袋编码器层数
- model.pocket.encoder_embed_dim: 口袋编码器维度
- model.pocket.encoder_attention_heads: 口袋编码器注意力头数
- model.pocket.dropout: 口袋编码器 dropout
- model.temperature: 对比学习温度 (0.02-0.2)

## 调参策略
1. 如果 loss 在上升 → 减小学习率
2. 如果 loss 进入 plateau → 减小学习率 0.5x
3. 如果 loss 震荡 → 减小学习率 + 增大 batch_size
4. 如果 AUROC < 0.6 → 增大模型容量 (层数 + 维度)
5. 如果 train loss 远低于 val loss (过拟合) → 增大 dropout + weight_decay
6. 如果 badcase 中大多数是"small ligand"问题 → 考虑调整温度
7. 每次只修改 1-3 个参数，便于追踪效果

## 输出格式
你必须只输出一个 JSON 对象，不要输出任何其他内容：
{
  "analysis": "对当前状态的分析，1-2句话",
  "changes": {"参数路径": 新值, ...},
  "rationale": "为什么这样调整，引用具体数据",
  "expected_impact": "预期影响"
}"""


class TuningAgent(BaseAgent):
    """LLM-powered tuning agent — runs N iterations of train→eval→tune.

    Usage:
        agent = TuningAgent(logger=orch_logger, api_client=client, model="deepseek-v4-pro")
        result = agent.run(context, num_iterations=3)
    """

    def __init__(
        self,
        orch_logger: Optional[OrchLogger] = None,
        api_client: Optional[OpenAI] = None,
        model: str = "deepseek-v4-pro",
        output_dir: str = "output",
    ):
        super().__init__()
        self.log = orch_logger
        self.client = api_client
        self.model = model
        self._init_client()
        # Per-agent detailed logger
        self.agent_log = AgentLogger(output_dir, "tuning")

    def _init_client(self):
        if self.client:
            return
        api_key = (os.environ.get("ANTHROPIC_AUTH_TOKEN")
                   or os.environ.get("OPENAI_API_KEY"))
        if not api_key:
            return
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        model = (os.environ.get("ANTHROPIC_DEFAULT_SONNET_MODEL")
                 or os.environ.get("ANTHROPIC_MODEL") or self.model)
        self.model = re.sub(r"\[\d+m\]", "", model).strip()

    def run(self, context: dict, num_iterations: int = 3, silent: bool = False) -> dict:
        """Run tuning loop.

        Args:
            context: standard agent context dict
            num_iterations: how many train→eval→tune cycles to run
            silent: if True, suppress all terminal output (background mode)

        Returns:
            {
                "status": "ok",
                "data": {
                    "iterations": [{...}, ...],  # per-iteration results
                    "best_config": {...},        # best config found
                    "best_improvement": {...},   # delta from baseline
                },
                "summary": "Tuned 3 iterations, AUROC improved 0.78→0.85"
            }
        """
        if not self.client:
            return {"status": "error", "summary": "No API client for tuning LLM",
                    "data": {}}

        config = context["config"]
        self.agent_log.section(f"Tuning Loop — {num_iterations} iterations requested")
        self.agent_log.json_data(self._config_snapshot(config), "Initial Config:")

        # ── Phase 0: Recommend + Confirm ──
        recommended = self._recommend_params(config, num_iterations,
                                              self.agent_log.summary if hasattr(self.agent_log, 'summary') else "")
        # Present recommendation to user
        print("\n  === 调参方案建议 ===")
        for k, v in recommended.get("default_params", {}).items():
            print(f"  {k}: {v}")
        print(f"  模型: {recommended.get('model_name', config.model.model_name)}")
        print(f"  调参轮数: {num_iterations}")
        print(f"  调参范围: {recommended.get('tuning_scope', '')}")
        print(f"  预期目标: {recommended.get('expected_outcome', '')}")
        print(f"  风险提示: {recommended.get('risks', '')}")
        print()

        # Also send to LLM for natural language recommendation
        self._present_to_main(recommended)

        self.log and self.log.agent_call("TuningAgent", "self", "start_tuning",
                                          {"iterations": num_iterations,
                                           "recommended": recommended})

        # ── Phase 1: Baseline ──
        self.agent_log.info("Phase 0: Baseline Training")
        self.log and self.log.tool_call("train_tool", {"stage": "baseline"})
        from train.tools.train_tool import train_tool
        baseline_train = train_tool(config)
        self.agent_log.tool_result("train_tool", baseline_train)
        self.log and self.log.tool_result("train_tool", baseline_train)

        if baseline_train["status"] != "ok":
            return {"status": "error", "summary": "Baseline training failed",
                    "data": {}, "error": baseline_train.get("error")}

        context["checkpoint"] = baseline_train["data"]["checkpoint_path"]
        per_epoch = baseline_train["data"].get("per_epoch_metrics", [])

        # Baseline eval
        from train.tools.eval_tool import eval_tool
        baseline_eval = eval_tool(config, context["checkpoint"])
        self.agent_log.tool_result("eval_tool", baseline_eval)
        baseline_auroc = baseline_eval.get("data", {}).get("overall_metrics", {}).get("auroc", 0)
        self.agent_log.kv("Baseline AUROC", f"{baseline_auroc:.4f}")

        iterations = []
        best_config = self._config_snapshot(config)
        best_auroc = baseline_auroc

        # ── Phase 1: Tuning loop ──
        for i in range(num_iterations):
            self.agent_log.section(f"Iteration {i+1}/{num_iterations}")
            self.log and self.log.iteration("TuningAgent", i + 1)

            # 1. Ask LLM for parameter changes
            self.agent_log.info("Asking LLM for parameter changes...")
            param_changes = self._ask_llm(per_epoch, baseline_eval, config, iterations)

            if not param_changes:
                self.agent_log.error("LLM returned no valid JSON changes")
                self.log and self.log.error("TuningAgent", "LLM returned no changes")
                continue

            self.agent_log.decision(
                param_changes.get("analysis", ""),
                param_changes.get("changes", {}),
                param_changes.get("rationale", ""),
            )
            self.log and self.log.llm_decision("TuningAgent", {
                "hypothesis": param_changes.get("analysis", ""),
                "action": param_changes.get("changes", {}),
                "rationale": param_changes.get("rationale", ""),
            })

            # 2. Apply changes
            self.agent_log.info("Applying parameter changes...")
            for key, value in param_changes.get("changes", {}).items():
                self._apply_config(config, key, value)
                self.agent_log.kv(key, value)
            self.agent_log.json_data(self._config_snapshot(config), "Config After Changes:")

            # 3. Retrain
            self.log and self.log.tool_call("train_tool", {"stage": f"iter_{i+1}"})
            train_result = train_tool(config)
            self.agent_log.tool_result("train_tool", train_result)
            self.log and self.log.tool_result("train_tool", train_result)

            if train_result["status"] != "ok":
                self.agent_log.error(f"Training failed at iteration {i+1}")
                self.log and self.log.error("TuningAgent",
                                             f"Training failed at iteration {i+1}")
                continue

            context["checkpoint"] = train_result["data"]["checkpoint_path"]
            per_epoch = train_result["data"].get("per_epoch_metrics", [])

            # 4. Evaluate
            eval_result = eval_tool(config, context["checkpoint"])
            self.agent_log.tool_result("eval_tool", eval_result)
            self.log and self.log.tool_result("eval_tool", eval_result)

            current_auroc = eval_result.get("data", {}).get("overall_metrics", {}).get("auroc", 0)
            current_ef1 = eval_result.get("data", {}).get("overall_metrics", {}).get("ef1", 0)

            # ── Print iteration result clearly ──
            print(f"\n  === 迭代 {i+1} 结果 ===")
            if param_changes.get("analysis"):
                print(f"  分析: {param_changes['analysis']}")
            if param_changes.get("rationale"):
                print(f"  理由: {param_changes['rationale']}")
            if param_changes.get("changes"):
                print(f"  调整: {json.dumps(param_changes['changes'], ensure_ascii=False)}")
            print(f"  AUROC: {baseline_auroc:.4f} → {current_auroc:.4f} (Δ{current_auroc-baseline_auroc:+.4f})")
            print(f"  EF1%:  {baseline_eval.get('data',{}).get('overall_metrics',{}).get('ef1',0):.1f} → {current_ef1:.1f}")

            improvement = {
                "iteration": i + 1,
                "hypothesis": param_changes.get("analysis", ""),
                "auroc_delta": round(current_auroc - baseline_auroc, 4),
                "ef1_delta": round(current_ef1 - baseline_eval.get("data", {}).get("overall_metrics", {}).get("ef1", 0), 1
                ),
                "current_auroc": current_auroc,
                "current_ef1": eval_result.get("data", {}).get("overall_metrics", {}).get("ef1", 0),
                "changes": param_changes.get("changes", {}),
            }
            self.agent_log.json_data(improvement, "Iteration Improvement:")

            if current_auroc > best_auroc:
                best_auroc = current_auroc
                best_config = self._config_snapshot(config)
                self.agent_log.info(f"NEW BEST: AUROC={best_auroc:.4f}")

            iterations.append(improvement)
            self.log and self.log.iteration_result("TuningAgent", i + 1, improvement)

        # ── Final summary ──
        final_improvement = best_auroc - baseline_auroc
        summary = (f"Tuned {len(iterations)} iterations, "
                   f"AUROC {baseline_auroc:.4f}→{best_auroc:.4f} "
                   f"({final_improvement:+.4f})")

        self.agent_log.section("Final Summary")
        self.agent_log.kv("Baseline AUROC", f"{baseline_auroc:.4f}")
        self.agent_log.kv("Best AUROC", f"{best_auroc:.4f}")
        self.agent_log.kv("Improvement", f"{final_improvement:+.4f}")
        self.agent_log.json_data(iterations, "All Iterations:")
        self.agent_log.close()

        self.log and self.log.agent_result("TuningAgent", {"status": "ok", "summary": summary})

        return {
            "status": "ok",
            "data": {
                "iterations": iterations,
                "baseline_auroc": baseline_auroc,
                "best_auroc": best_auroc,
                "improvement": final_improvement,
                "best_config_updates": best_config,
            },
            "summary": summary,
        }

    # ── LLM interaction ─────────────────────────────────────────────

    def _ask_llm(self, per_epoch: list, eval_result: dict, config, history: list) -> dict:
        """Send current state to LLM, get parameter change suggestions."""

        # Format the data for LLM consumption
        metrics = eval_result.get("data", {}).get("overall_metrics", {})
        badcases = eval_result.get("data", {}).get("badcase_analysis", {})

        loss_info = ""
        if per_epoch:
            first, last = per_epoch[0], per_epoch[-1]
            loss_info = (f"Loss: {first['train_loss']:.4f} → {last['train_loss']:.4f} "
                         f"(Δ{(last['train_loss'] - first['train_loss']):.4f}), "
                         f"{len(per_epoch)} epochs")

        history_info = ""
        if history:
            history_info = "历史调参:\n"
            for h in history[-3:]:
                history_info += (f"  - {h.get('analysis','')}: "
                                 f"AUROC {h.get('current_auroc',0):.4f} "
                                 f"(Δ{h.get('auroc_delta',0):+.4f})\n")

        badcase_info = ""
        if badcases.get("badcases"):
            badcase_info = f"Badcases ({len(badcases['badcases'])}):\n"
            for b in badcases["badcases"][:3]:
                badcase_info += f"  - {b.get('hypothesis', '?')}\n"

        current_config = {
            "train": {"lr": config.train.lr, "batch_size": config.train.batch_size,
                       "weight_decay": config.train.weight_decay,
                       "warmup_epochs": config.train.warmup_epochs},
            "mol": {"layers": config.model.mol.encoder_layers,
                     "dim": config.model.mol.encoder_embed_dim,
                     "heads": config.model.mol.encoder_attention_heads,
                     "dropout": config.model.mol.dropout},
            "pocket": {"layers": config.model.pocket.encoder_layers,
                        "dim": config.model.pocket.encoder_embed_dim,
                        "heads": config.model.pocket.encoder_attention_heads,
                        "dropout": config.model.pocket.dropout},
            "temperature": config.model.temperature,
        }

        prompt = f"""## 当前训练状态
{loss_info}

## 评估指标
AUROC: {metrics.get('auroc', 'N/A')}
EF1%: {metrics.get('ef1', 'N/A')}
EF5%: {metrics.get('ef5', 'N/A')}
MRR: {metrics.get('mrr', 'N/A')}
Top-1: {metrics.get('top1', 'N/A')}
Top-5: {metrics.get('top5', 'N/A')}

## Badcase 分析
{badcase_info or '无'}

## 当前配置
{json.dumps(current_config, indent=2)}

## 历史调参
{history_info or '首次调参'}

请分析当前状态，给出本次调参建议。只输出 JSON。"""

        try:
            self.agent_log.llm_prompt(TUNING_SYSTEM_PROMPT, prompt)
            result = call_llm_json(self.client, self.model, TUNING_SYSTEM_PROMPT, prompt)
            self.agent_log.llm_response(json.dumps(result, ensure_ascii=False))
            return result
        except Exception as e:
            self.agent_log.error(f"LLM call failed: {e}")
            self.log and self.log.error("TuningAgent.LLM", str(e))
            return {}

    @staticmethod
    def _parse_json(content: str) -> dict:
        """Extract JSON object from LLM response (may have markdown fences)."""
        # Try direct parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        # Try extract from code fence
        if "```json" in content:
            try:
                start = content.index("```json") + 7
                end = content.index("```", start)
                return json.loads(content[start:end].strip())
            except (ValueError, json.JSONDecodeError):
                pass
        if "```" in content:
            try:
                start = content.index("```") + 3
                end = content.index("```", start)
                return json.loads(content[start:end].strip())
            except (ValueError, json.JSONDecodeError):
                pass
        # Try find { } block
        try:
            start = content.index("{")
            end = content.rindex("}") + 1
            return json.loads(content[start:end])
        except (ValueError, json.JSONDecodeError):
            pass
        return {}

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _apply_config(config, key: str, value):
        parts = key.split(".")
        obj = config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        current = getattr(obj, parts[-1])
        if isinstance(current, int):
            setattr(obj, parts[-1], int(float(value)))
        elif isinstance(current, float):
            setattr(obj, parts[-1], float(value))
        else:
            setattr(obj, parts[-1], value)

    # ── Parameter recommendation ──────────────────────────────────────

    @staticmethod
    def _recommend_params(config, num_iterations: int, history_hint: str = "") -> dict:
        """Analyze current config and generate parameter recommendations."""
        m = config.model
        t = config.train

        issues = []
        default_params = {}

        # Check for common issues
        if t.lr >= 3e-4:
            issues.append("学习率偏高，建议降低")
            default_params["train.lr"] = 1e-4
        if t.epochs < 30:
            issues.append("训练轮数较少，建议增加到 30-50")
            default_params["train.epochs"] = 30

        if m.mol.encoder_layers < 4:
            issues.append("模型层数较少（<4），可能欠拟合")
            default_params["model.mol.encoder_layers"] = 4
            default_params["model.pocket.encoder_layers"] = 4

        if m.mol.dropout < 0.1:
            issues.append("dropout 较低，有 overfitting 风险")
            default_params["model.mol.dropout"] = 0.2

        if m.temperature < 0.05:
            issues.append("温度过低可能导致梯度消失")
            default_params["model.temperature"] = 0.07

        if t.batch_size <= 16:
            issues.append("batch_size 较小，loss 可能不稳定")
            default_params["train.batch_size"] = 32

        return {
            "model_name": config.model.model_name,
            "default_params": default_params,
            "tuning_scope": "学习率、模型容量、dropout、温度、batch_size",
            "expected_outcome": f"经过 {num_iterations} 轮调参后 AUROC 预期提升 0.03-0.10",
            "risks": "每轮调参需重新训练，耗时较长；若 baseline 已很好则提升空间有限",
            "issues_found": issues,
        }

    @staticmethod
    def _present_to_main(recommended: dict):
        """Output recommendation in a format the main LLM agent can parse."""
        print()
        print("--- TUNING_RECOMMENDATION ---")
        print(f"MODEL: {recommended.get('model_name', 'drugclip')}")
        print(f"SCOPE: {recommended.get('tuning_scope', '')}")
        print(f"PARAMS: {recommended.get('default_params', {})}")
        print(f"OUTCOME: {recommended.get('expected_outcome', '')}")
        print(f"RISKS: {recommended.get('risks', '')}")
        if recommended.get("issues_found"):
            print(f"ISSUES: {'; '.join(recommended['issues_found'])}")
        print("--- END ---")

    @staticmethod
    def _config_snapshot(config) -> dict:
        return {
            "train.lr": config.train.lr,
            "train.batch_size": config.train.batch_size,
            "train.weight_decay": config.train.weight_decay,
            "model.mol.encoder_layers": config.model.mol.encoder_layers,
            "model.mol.encoder_embed_dim": config.model.mol.encoder_embed_dim,
            "model.mol.encoder_attention_heads": config.model.mol.encoder_attention_heads,
            "model.mol.dropout": config.model.mol.dropout,
            "model.pocket.encoder_layers": config.model.pocket.encoder_layers,
            "model.pocket.encoder_embed_dim": config.model.pocket.encoder_embed_dim,
            "model.pocket.encoder_attention_heads": config.model.pocket.encoder_attention_heads,
            "model.pocket.dropout": config.model.pocket.dropout,
            "model.temperature": config.model.temperature,
        }
