"""BenchmarkAgent — run full inference pipeline + generate submission zip."""

from pathlib import Path

from .base_agent import BaseAgent


class BenchmarkAgent(BaseAgent):
    """Run benchmark inference on all tasks and produce submission zip."""

    def run(self, context: dict) -> dict:
        config = context["config"]
        checkpoint_path = context.get("checkpoint")

        if not checkpoint_path or not Path(checkpoint_path).exists():
            return {
                "status": "error",
                "data": {},
                "summary": "No valid checkpoint for inference",
                "error": f"Checkpoint not found: {checkpoint_path}",
            }

        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from train.inference import InferenceEngine

            engine = InferenceEngine(config, checkpoint_path=checkpoint_path)
            result_csv = engine.run_all_tasks()
            submission_path = engine.create_submission_zip()

            return {
                "status": "ok",
                "data": {
                    "result_csv": result_csv,
                    "submission_path": submission_path,
                },
                "summary": f"Benchmark complete → {submission_path}",
            }
        except Exception as e:
            return {
                "status": "error",
                "data": {},
                "summary": "Benchmark inference failed",
                "error": str(e),
            }
