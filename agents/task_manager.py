"""Background task manager — spawn, monitor, kill sub-agent tasks."""

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Callable


class BackgroundTask:
    """A background agent task with monitoring and kill support."""

    def __init__(self, task_id: str, agent_name: str, log_path: str = None,
                 on_complete: Callable = None):
        self.task_id = task_id
        self.agent_name = agent_name
        self.log_path = log_path
        self.status = "pending"  # pending | running | done | error | killed
        self.started_at = None
        self.finished_at = None
        self.result = None
        self.error = None
        self.thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._on_complete = on_complete  # callback(status_dict)
        self._notified = False

    def is_running(self) -> bool:
        return self.status == "running" and self.thread and self.thread.is_alive()

    def stop(self):
        self._stop_event.set()
        self.status = "killed"
        self.finished_at = datetime.now()

    def read_log_tail(self, lines: int = 20) -> str:
        if not self.log_path or not Path(self.log_path).exists():
            return f"[{self.agent_name}] No log file yet."
        with open(self.log_path, encoding="utf-8") as f:
            return "".join(f.readlines()[-lines:])

    def mark_notified(self):
        self._notified = True

    def smart_summary(self) -> str:
        """Generate a concise, filtered summary — no raw JSON."""
        if self.status == "running":
            return f"⏳ {self.agent_name} 运行中..."
        if self.status == "killed":
            return f"🛑 {self.agent_name} 已被终止"

        result = self.result or {}
        data = result.get("data", {}) if isinstance(result, dict) else {}

        parts = []

        # Status icon
        if self.status == "done":
            parts.append("✅")
        elif self.status == "error":
            parts.append("❌")

        parts.append(f"[{self.agent_name}]")

        # Key info from result
        if isinstance(result, dict):
            parts.append(result.get("summary", ""))

            # Pick out important metrics
            if data.get("improvement"):
                imp = data["improvement"]
                parts.append(f"| AUROC提升: {imp:+.4f}")
            if data.get("best_auroc"):
                parts.append(f"| 最佳AUROC: {data['best_auroc']:.4f}")
            if data.get("baseline_auroc"):
                parts.append(f"| 基线: {data['baseline_auroc']:.4f}")
            if data.get("model_name"):
                parts.append(f"| 模型: {data['model_name']}")
            if data.get("registry_key"):
                parts.append(f"| 已注册为 '{data['registry_key']}'")
            if data.get("iterations"):
                parts.append(f"| {len(data['iterations'])}轮迭代")
            if data.get("experience"):
                parts.append(f"| 经验: {data['experience']}")

        if self.error:
            parts.append(f"| 错误: {self.error[:100]}")

        return " ".join(parts)


class TaskManager:
    """Manage background agent tasks with completion callbacks."""

    def __init__(self):
        self._tasks: Dict[str, BackgroundTask] = {}
        self._counter = 0

    def spawn(self, agent_name: str, target_fn, args: tuple = (),
              log_path: str = None) -> str:
        """Spawn a background task. Returns task_id."""
        self._counter += 1
        task_id = f"{agent_name}-{self._counter}"
        task = BackgroundTask(task_id, agent_name, log_path)
        self._tasks[task_id] = task

        def wrapper():
            task.status = "running"
            task.started_at = datetime.now()
            try:
                task.result = target_fn(*args)
                if task.status != "killed":
                    task.status = "done"
            except Exception as e:
                if task.status != "killed":
                    task.status = "error"
                    task.error = str(e)
            finally:
                task.finished_at = datetime.now()

        task.thread = threading.Thread(target=wrapper, daemon=True)
        task.thread.start()
        return task_id

    def poll_completed(self) -> list:
        """Return smart summaries of newly completed tasks (not yet notified)."""
        notifications = []
        for task in self._tasks.values():
            if task.status in ("done", "error", "killed") and not task._notified:
                notifications.append(task.smart_summary())
                task.mark_notified()
        return notifications

    def get(self, task_id: str) -> Optional[BackgroundTask]:
        return self._tasks.get(task_id)

    def status(self, task_id: str) -> dict:
        """Get task status as a dict."""
        task = self._tasks.get(task_id)
        if not task:
            return {"status": "error", "summary": f"Task {task_id} not found"}
        return {
            "task_id": task.task_id,
            "agent": task.agent_name,
            "status": task.status,
            "started": str(task.started_at) if task.started_at else None,
            "finished": str(task.finished_at) if task.finished_at else None,
            "result_summary": task.result.get("summary", "") if task.result else None,
            "error": task.error,
            "log_tail": task.read_log_tail(15) if task.log_path else "",
        }

    def kill(self, task_id: str) -> dict:
        """Kill a running task."""
        task = self._tasks.get(task_id)
        if not task:
            return {"status": "error", "summary": f"Task {task_id} not found"}
        if not task.is_running():
            return {"status": "error", "summary": f"Task {task_id} is not running ({task.status})"}
        task.stop()
        return {"status": "ok", "summary": f"Task {task_id} killed",
                "data": {"task_id": task_id}}

    def list_tasks(self) -> list:
        return [self.status(tid) for tid in self._tasks]

    def check_progress(self, task_id: str, tail_lines: int = 15) -> dict:
        """Check a running task's progress via its log file."""
        task = self._tasks.get(task_id)
        if not task:
            return {"status": "error", "summary": f"Task {task_id} not found"}
        return {
            "status": "ok",
            "data": {
                "task_id": task_id,
                "agent": task.agent_name,
                "task_status": task.status,
                "log_tail": task.read_log_tail(tail_lines),
            },
            "summary": f"{task.agent_name}: {task.status}",
        }
