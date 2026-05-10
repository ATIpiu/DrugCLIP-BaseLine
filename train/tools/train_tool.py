"""Training tool — returns structured loss/checkpoint data for agents."""

from pathlib import Path
from ..config import Config
from ..core.trainer import Trainer


def train_tool(config: Config) -> dict:
    """Run DrugCLIP training with given config. Returns structured result dict.

    Args:
        config: Full Config object (may be modified by TuningAgent decisions).

    Returns:
        {
            "status": "ok" | "error",
            "data": {
                "best_loss": float,
                "best_epoch": int,
                "epochs_trained": int,
                "checkpoint_path": str,
                "per_epoch_metrics": [{"epoch": 1, ...}, ...],
            },
            "summary": "best_loss=0.2345 epoch=87",
            "error": None
        }
    """
    try:
        val_dir = config.data.val_data_dir
        if val_dir and not Path(val_dir).exists():
            val_dir = None  # fall back to train/val split
        trainer = Trainer(config)
        train_result = trainer.train(
            train_dir=config.data.train_data_dir,
            val_dir=val_dir,
        )

        model_name = config.model.model_name
        model_dir = str(Path(trainer.config.data.output_dir) / "models" / model_name)
        checkpoint_path = model_dir + "/checkpoints/best.pt"

        return {
            "status": "ok",
            "data": {
                "best_loss": train_result["best_loss"],
                "best_epoch": train_result["best_epoch"],
                "epochs_trained": train_result["epochs_trained"],
                "checkpoint_path": checkpoint_path,
                "per_epoch_metrics": train_result["per_epoch_metrics"],
            },
            "summary": (
                f"Trained {train_result['epochs_trained']} epochs, "
                f"best_loss={train_result['best_loss']:.4f} "
                f"at epoch {train_result['best_epoch']}"
            ),
        }
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return {
            "status": "error",
            "data": {},
            "summary": f"Training failed: {e}",
            "error": tb[-500:],  # last 500 chars of traceback (root cause)
        }
