"""Orchestration logger — complete audit trail to output/agent.log.

Format: one JSON line per event. ALL data preserved, no truncation.
"""

import json
import time
from datetime import datetime
from pathlib import Path


class OrchLogger:
    """Complete agent orchestration audit log.

    Every user message, LLM response (full), tool call (full params),
    tool result (full data), agent decision, and error is recorded.
    """

    def __init__(self, output_dir: str = "output"):
        self.log_path = Path(output_dir) / "agent.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._write({"event": "session_start", "session": self.session_id})

    # ── User interaction ──────────────────────────────────────────────

    def user_message(self, content: str):
        self._write({"event": "user_message", "content": content})

    # ── LLM interaction ───────────────────────────────────────────────

    def llm_stream_chunk(self, text: str):
        """Individual text chunk from streaming LLM (batched per call)."""
        self._write({"event": "llm_stream_chunk", "text": text})

    def llm_response(self, text: str, tool_calls: list = None, reasoning: str = None):
        """Complete LLM response after streaming finishes."""
        entry = {"event": "llm_response", "text": text}
        if tool_calls:
            entry["tool_calls"] = tool_calls
        if reasoning:
            entry["reasoning"] = reasoning
        self._write(entry)

    def llm_tool_call(self, name: str, arguments: dict):
        """LLM decided to call a tool."""
        self._write({"event": "llm_tool_call", "name": name, "arguments": arguments})

    # ── Agent calls ────────────────────────────────────────────────────

    def agent_call(self, caller: str, callee: str, action: str, params: dict = None):
        self._write({"event": "agent_call", "caller": caller, "callee": callee,
                      "action": action, "params": params or {}})

    def agent_result(self, agent: str, result: dict):
        self._write({"event": "agent_result", "agent": agent,
                      "status": result.get("status", "?") if isinstance(result, dict) else "?",
                      "result": result})

    # ── Tool execution ────────────────────────────────────────────────

    def tool_call(self, tool: str, params: dict = None):
        self._write({"event": "tool_call", "tool": tool, "params": params or {}})

    def tool_result(self, tool: str, result: dict):
        self._write({"event": "tool_result", "tool": tool,
                      "status": result.get("status", "?") if isinstance(result, dict) else "?",
                      "result": result})

    # ── Agent decisions ───────────────────────────────────────────────

    def llm_decision(self, agent: str, decision: dict):
        self._write({"event": "llm_decision", "agent": agent, "decision": decision})

    def iteration(self, agent: str, iteration: int, metrics: dict = None):
        self._write({"event": "iteration", "agent": agent, "iteration": iteration,
                      "metrics": metrics or {}})

    def iteration_result(self, agent: str, iteration: int, improvement: dict):
        self._write({"event": "iteration_result", "agent": agent,
                      "iteration": iteration, "improvement": improvement})

    # ── Errors ────────────────────────────────────────────────────────

    def error(self, source: str, error: str):
        self._write({"event": "error", "source": source, "error": error})

    # ── Internal ──────────────────────────────────────────────────────

    def _write(self, entry: dict):
        entry["ts"] = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            f.flush()
