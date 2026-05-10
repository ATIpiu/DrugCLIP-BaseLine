"""ModelAgent — LLM-powered model generation with read/edit tools, no wasted tokens.

The LLM has access to:
  - read_file(path) — read any project file (config, existing models, encoders)
  - write_file(path, content) — create new model file
  - edit_file(path, old, new) — targeted replacement
  - delete_file(path) — cleanup
  - quick_test(model_name) — import + forward pass

Flow:
  1. LLM reads config.py, unimol_encoder.py, drugclip.py (understands the real API)
  2. LLM designs architecture with correct API
  3. LLM writes model file
  4. Quick test → if fail, LLM reads error + edits file (not full regeneration)
  5. Max 5 repair rounds, using targeted edits only
"""

import json
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from openai import OpenAI

from .base_agent import BaseAgent
from .agent_logger import AgentLogger
from .llm_utils import stream_llm


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


# ── Internal tools the LLM can call ───────────────────────────────

INTERNAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取项目中的任意文件。用于了解现有代码API（如 config.py, unimol_encoder.py 等）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对于项目根目录的路径，如 train/model/unimol_encoder.py"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "创建新的模型文件 train/model/model_<name>.py。第一行 MODEL_NAME，第二行 DESCRIPTION，然后是完整代码。",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "完整的 Python 模型代码"},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "对已创建的模型文件做精确替换修改。不重新生成整个文件，只替换需要改的部分。",
            "parameters": {
                "type": "object",
                "properties": {
                    "old_string": {"type": "string", "description": "要替换的原代码片段（精确匹配）"},
                    "new_string": {"type": "string", "description": "替换后的新代码片段"},
                },
                "required": ["old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "quick_test",
            "description": "对当前模型文件做快速测试（导入+实例化+前向传播）。返回成功或错误信息。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "register_model",
            "description": "测试通过后将模型注册到注册表，使其可用于训练。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

SYSTEM_PROMPT = """你是 PyTorch 模型架构专家。你可以使用以下工具来设计和实现新模型。

## 工作流程
1. 先 read_file 查看现有代码了解真实 API:
   - train/config.py (配置类)
   - train/model/unimol_encoder.py (UniMolEncoder, GaussianLayer)
   - train/model/drugclip.py (DrugCLIP 参考实现)
   - train/model/transformer.py (TransformerEncoderWithPair)
   - train/model/registry.py (模型注册表)
   注意: config.py 在 train/ 根目录, 不在 train/model/ 下
2. 设计新架构（确保调用真实存在的 API）
3. write_file 创建模型文件
4. quick_test 测试
5. 如果测试失败，用 edit_file 精确修改（不要重新 write_file 整个文件）
6. 测试通过后用 register_model 注册

## 模型文件格式
第一行: MODEL_NAME = "short_english_name"
第二行: DESCRIPTION = "..."
然后是完整 Python 代码。类名必须是 Model。
forward(batch) 返回 (mol_emb, pocket_emb)，两者 (B, project_dim) 且 L2 归一化。

## 关键约束
- 每次只调用一个工具，不要并发调用多个
- 只能新增 train/model/model_<name>.py，不修改现有文件
- 必须使用项目中真实存在的类和 API
- 修复时用 edit_file 精准替换，不要反复 write_file 整个文件"""


class ModelAgent(BaseAgent):
    """Self-contained model generation with internal read/edit/test tools."""

    MAX_REPAIR_ROUNDS = 5

    def __init__(self, output_dir: str = "output",
                 openai_client: Optional[OpenAI] = None, openai_model: str = None):
        super().__init__()
        self.output_dir = output_dir
        self.agent_log = AgentLogger(output_dir, "model")
        api_key = (os.environ.get("LLM_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
        base_url = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
        self.openai = openai_client or OpenAI(api_key=api_key, base_url=base_url)
        self.openai_model = openai_model or self._resolve_model()
        self.model_dir = Path(__file__).parent.parent / "train" / "model"
        self._current_file = None  # The file being worked on
        self._current_name = None
        self._messages = []  # LLM conversation history

    def _resolve_model(self) -> str:
        m = os.environ.get("LLM_MODEL") or os.environ.get("ANTHROPIC_DEFAULT_SONNET_MODEL") or "deepseek-v4-pro"
        return re.sub(r"\[\d+m\]", "", m).strip()

    # ── Main entry ──────────────────────────────────────────────────

    def run(self, context: dict) -> dict:
        user_request = context.get("user_request", context.get("last_user_message", ""))
        ref_docs = context.get("reference_docs", {})

        self.agent_log.section("ModelAgent")
        self.agent_log.kv("Request", user_request)

        if not user_request:
            return {"status": "error", "summary": "No model description", "data": {}}

        self._messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"请设计新模型。需求: {user_request}\n\n先 read_file 读 train/config.py 和 train/model/unimol_encoder.py 了解 API，再设计实现。"},
        ]

        experience = []
        last_error = ""
        repair_count = 0  # Only count quick_test failures

        while True:
            resp, tcalls, reasoning = stream_llm(self.openai, self.openai_model, self._messages,
                                                   tools=INTERNAL_TOOLS, temperature=0.2)

            asst = {"role": "assistant", "content": resp}
            if reasoning:
                asst["reasoning_content"] = reasoning
            if tcalls:
                asst["tool_calls"] = [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}}
                    for tc in tcalls
                ]
            self._messages.append(asst)

            if not tcalls:
                self._messages.append({"role": "user", "content": "请调用一个工具"})
                continue

            # Only execute FIRST tool call — one at a time
            tc = tcalls[0]
            name = tc["name"]
            args = tc["arguments"]
            print(f"  🔧 {name}", end="", flush=True)

            result = self._exec_internal(name, args)
            self._messages.append({"role": "tool", "tool_call_id": tc["id"],
                                   "content": json.dumps(result, ensure_ascii=False, default=str)})

            if result.get("status") == "error":
                print(f" ❌ {result.get('summary', '')[:80]}")
                last_error = result.get("error", result.get("summary", ""))
                if name == "quick_test":
                    repair_count += 1
                    self.agent_log.section(f"Repair {repair_count}/{self.MAX_REPAIR_ROUNDS}")
                    experience.append(f"Test {repair_count}: {result.get('summary','')[:100]}")
                    if repair_count >= self.MAX_REPAIR_ROUNDS:
                        self.agent_log.section("FAILED")
                        self.agent_log.error(f"All {self.MAX_REPAIR_ROUNDS} repair rounds exhausted: {last_error}")
                        self.agent_log.close()
                        return {"status": "error",
                                "summary": f"Failed after {self.MAX_REPAIR_ROUNDS} repair rounds. {last_error[:200]}",
                                "data": {"experience": "; ".join(experience), "last_error": last_error}}
                    self._messages.append({"role": "user",
                        "content": f"quick_test 失败 ({repair_count}/{self.MAX_REPAIR_ROUNDS}): {last_error[:300]}\n请用 edit_file 精准修改（不要重新生成整个文件）。"})
            else:
                print(f" ✅")

            if name == "register_model":
                self.agent_log.close()
                return {
                    "status": "ok",
                    "data": {
                        "model_name": self._current_name,
                        "registry_key": self._current_name,
                        "usage": f"config.model.model_name = '{self._current_name}'",
                        "experience": "; ".join(experience),
                    },
                    "decisions": [{
                        "hypothesis": f"New architecture: {self._current_name}",
                        "action": {"model_name": self._current_name},
                        "rationale": "; ".join(experience) if experience else f"Model '{self._current_name}' generated and tested",
                    }],
                    "summary": f"Model '{self._current_name}' created, tested, registered.",
                }

    # ── Internal tool executor ──────────────────────────────────────

    def _exec_internal(self, name: str, args: dict) -> dict:
        handlers = {
            "read_file": self._read_file,
            "write_file": self._write_file,
            "edit_file": self._edit_file,
            "quick_test": self._quick_test,
            "register_model": self._register,
        }
        h = handlers.get(name)
        if not h:
            return {"status": "error", "summary": f"Unknown: {name}"}
        try:
            return h(args)
        except Exception as e:
            return {"status": "error", "summary": str(e), "error": traceback.format_exc()[-300:]}

    # ── Tool implementations ────────────────────────────────────────

    def _read_file(self, args: dict) -> dict:
        path = Path(args["path"])
        if not path.is_absolute():
            path = Path(__file__).parent.parent / path
        # Restrict to train/ only
        train_dir = Path(__file__).parent.parent / "train"
        try:
            path.resolve().relative_to(train_dir.resolve())
        except ValueError:
            return {"status": "error", "summary": f"Access denied — only train/ files allowed. Try: train/config.py, train/model/unimol_encoder.py"}

        if not path.exists():
            # Suggest nearby files
            parent = path.parent
            nearby = list(parent.glob("*.py"))[:5] if parent.exists() else []
            hint = f" | Files in {path.parent.name}/: {[f.name for f in nearby]}" if nearby else ""
            return {"status": "error", "summary": f"Not found: {path}{hint}"}

        content = path.read_text(encoding="utf-8")
        return {"status": "ok", "data": {"path": str(path), "content": content, "lines": len(content.splitlines())},
                "summary": f"Read {path.name} ({len(content)} chars, {len(content.splitlines())} lines)"}

    def _write_file(self, args: dict) -> dict:
        code = args["code"]
        # Extract MODEL_NAME
        m = re.search(r'MODEL_NAME\s*=\s*"([^"]+)"', code)
        name = m.group(1) if m else f"v{datetime.now().strftime('%H%M%S')}"
        name = re.sub(r'[^a-z0-9_]', '', name.lower())[:20]

        filepath = self.model_dir / f"model_{name}.py"
        filepath.write_text(code, encoding="utf-8")
        self._current_file = filepath
        self._current_name = name
        self.agent_log.kv("Created", str(filepath))
        return {"status": "ok", "data": {"file": str(filepath), "name": name, "size": len(code)},
                "summary": f"Created model_{name}.py ({len(code)} chars)"}

    def _edit_file(self, args: dict) -> dict:
        if not self._current_file or not self._current_file.exists():
            return {"status": "error", "summary": "No current model file to edit. Use write_file first."}
        old = args["old_string"]
        new = args["new_string"]
        content = self._current_file.read_text(encoding="utf-8")
        if old not in content:
            return {"status": "error", "summary": "old_string not found in file",
                    "data": {"file_preview": content[:500]}}
        content = content.replace(old, new, 1)
        self._current_file.write_text(content, encoding="utf-8")
        return {"status": "ok", "summary": f"Replaced {len(old)}→{len(new)} chars in {self._current_file.name}"}

    def _quick_test(self, args: dict) -> dict:
        if not self._current_file or not self._current_file.exists():
            return {"status": "error", "summary": "No model file to test"}
        try:
            import torch
            from train.config import ModelConfig
            from train.model.registry import register as reg_model
            reg_model(name="_test", module_path=f"train.model.{self._current_file.stem}",
                      class_name="Model", source="test", description="test")
            config = ModelConfig()
            config.mol.encoder_layers = 2; config.mol.encoder_embed_dim = 64
            config.mol.encoder_ffn_embed_dim = 256; config.mol.encoder_attention_heads = 4
            config.pocket.encoder_layers = 2; config.pocket.encoder_embed_dim = 64
            config.pocket.encoder_ffn_embed_dim = 256; config.pocket.encoder_attention_heads = 4
            config.project_dim = 64; config.model_name = "_test"
            from train.model.registry import get_model
            model = get_model("_test", config)
            n = sum(p.numel() for p in model.parameters())
            B, N = 2, 10
            batch = {"mol_tokens": torch.randint(0, 24, (B, N)),
                     "mol_distances": torch.randn(B, N, N).abs(),
                     "mol_edge_types": torch.randint(0, 576, (B, N, N)),
                     "pocket_tokens": torch.randint(0, 24, (B, N+5)),
                     "pocket_distances": torch.randn(B, N+5, N+5).abs(),
                     "pocket_edge_types": torch.randint(0, 576, (B, N+5, N+5))}
            model.eval()
            with torch.no_grad():
                me, pe = model(batch)
            assert me.shape == (B, 64), f"mol shape {me.shape}"
            assert pe.shape == (B, 64), f"pocket shape {pe.shape}"
            return {"status": "ok", "data": {"params_m": f"{n/1e6:.2f}"},
                    "summary": f"PASS — {n/1e6:.2f}M params"}
        except Exception:
            return {"status": "error", "summary": traceback.format_exc()[-200:]}

    def _register(self, args: dict) -> dict:
        if not self._current_name:
            return {"status": "error", "summary": "No model to register"}
        from train.model.registry import register as reg_model
        reg_model(name=self._current_name,
                  module_path=f"train.model.{self._current_file.stem}",
                  class_name="Model", source="agent",
                  description=f"Agent-generated: {self._current_name}")
        return {"status": "ok", "summary": f"Registered '{self._current_name}'"}
