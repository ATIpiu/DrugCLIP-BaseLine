"""Per-agent logger — each sub-agent writes its own detailed log file.

Directory structure:
    output/agent_logs/
        main/<session>.log        # Main agent orchestration
        tuning/<session>.log      # Tuning agent full prompt/response
        model/<session>.log       # Model agent architecture changes
        benchmark/<session>.log   # Benchmark inference details
        data/<session>.log        # Data verification details

Each agent log is a chronological plaintext file with structured sections.
The global agent.log (JSON lines) remains the master index.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional


class AgentLogger:
    """Dedicated logger for a single sub-agent.

    Usage:
        log = AgentLogger("output", "tuning")
        log.section("Tuning Iteration 1")
        log.kv("AUROC", 0.85)
        log.llm_prompt("...")
        log.llm_response('{"changes": {...}}')
        log.result({"status": "ok", "summary": "..."})
    """

    def __init__(self, output_dir: str, agent_name: str, session_id: str = None):
        self.agent_name = agent_name
        self.session_id = session_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = Path(output_dir) / "agent_logs" / agent_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"{self.session_id}.log"

        self._write(f"{'='*60}")
        self._write(f"Agent: {agent_name} | Session: {self.session_id}")
        self._write(f"Start: {datetime.now():%Y-%m-%d %H:%M:%S}")
        self._write(f"{'='*60}")

    # ── Core write ───────────────────────────────────────────────────

    def _write(self, text: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {text}"
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def section(self, title: str):
        self._write("")
        self._write(f"{'─'*50}")
        self._write(f"  {title}")
        self._write(f"{'─'*50}")

    def info(self, message: str):
        self._write(f"  {message}")

    def kv(self, key: str, value, indent: int = 2):
        prefix = " " * indent
        self._write(f"{prefix}{key}: {value}")

    def json_data(self, data: dict, title: str = None, indent: int = 2):
        if title:
            self.info(title)
        formatted = json.dumps(data, indent=2, ensure_ascii=False, default=str)
        for line in formatted.split("\n"):
            self._write(" " * indent + line)

    def error(self, msg: str, trace: str = None):
        self.section("ERROR")
        self._write(f"  {msg}")
        if trace:
            self._write(f"  Traceback:\n{trace}")

    # ── Agent-specific ───────────────────────────────────────────────

    def llm_prompt(self, system: str, user: str):
        """Log full LLM prompt."""
        self.section("LLM Request")
        self._write("  --- SYSTEM ---")
        for line in system.split("\n"):
            self._write(f"  {line}")
        self._write("  --- USER ---")
        for line in user.split("\n"):
            self._write(f"  {line}")

    def llm_response(self, content: str, tool_calls: list = None):
        """Log LLM response."""
        self.section("LLM Response")
        if content:
            self._write(f"  Content: {content[:500]}")
        if tool_calls:
            self._write(f"  Tool Calls: {len(tool_calls)}")
            for tc in tool_calls:
                self._write(f"    - {tc.get('name', tc)}")

    def decision(self, hypothesis: str, changes: dict, rationale: str):
        """Log an agent decision."""
        self.section("Decision")
        self.kv("Hypothesis", hypothesis)
        self.kv("Rationale", rationale)
        if changes:
            self.json_data(changes, "Changes:")

    def tool_result(self, tool_name: str, result: dict):
        """Log a tool execution result."""
        self.section(f"Tool: {tool_name}")
        self.kv("Status", result.get("status", "?"))
        summary = result.get("summary", "")
        if summary:
            self.kv("Summary", summary)
        data = result.get("data", {})
        if data:
            # Show keys only, not full data (too verbose)
            self.kv("Data keys", list(data.keys()))

    def result(self, data: dict):
        """Log the agent's final result."""
        self.section("Result")
        self.json_data(data)

    def close(self):
        self._write("")
        self._write(f"End: {datetime.now():%Y-%m-%d %H:%M:%S}")
        self._write(f"{'='*60}")


# ── Utility: aggregate multiple agent logs into a summary ──────────

def aggregate_agent_logs(output_dir: str) -> dict:
    """Read all agent logs and return a summary for the main agent."""
    log_dir = Path(output_dir) / "agent_logs"
    if not log_dir.exists():
        return {"error": "No agent_logs directory"}

    summary = {}
    for agent_dir in sorted(log_dir.iterdir()):
        if not agent_dir.is_dir():
            continue
        sessions = sorted(agent_dir.glob("*.log"))
        if sessions:
            latest = sessions[-1]
            with open(latest, encoding="utf-8") as f:
                lines = f.readlines()
            summary[agent_dir.name] = {
                "sessions": len(sessions),
                "latest": str(latest),
                "lines": len(lines),
            }
    return summary
