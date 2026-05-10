"""Loss analysis tool — parses per-epoch metrics for trend/suggestion output."""

import numpy as np


def loss_tool(per_epoch_metrics: list) -> dict:
    """Analyze per-epoch training metrics and return trends + suggestions.

    Args:
        per_epoch_metrics: list of dicts from Trainer.epoch_metrics.
            Each dict: {"epoch": int, "train_loss": float, "train_infonce": float,
                        "train_triplet": float, ...}

    Returns:
        {
            "status": "ok",
            "data": {
                "trend": "decreasing" | "plateau" | "increasing" | "oscillating",
                "convergence_score": float (0-1),
                "loss_delta": float (change over last 5 epochs),
                "infonce_trend": str,
                "suggestions": [str, ...],
            },
            "summary": "Loss decreasing (-0.012/epoch), good convergence",
            "error": None
        }
    """
    try:
        if not per_epoch_metrics or len(per_epoch_metrics) < 3:
            return {
                "status": "ok",
                "data": {"trend": "insufficient_data", "suggestions": []},
                "summary": "Need at least 3 epochs for analysis",
            }

        losses = [m["train_loss"] for m in per_epoch_metrics]
        n = len(losses)

        # Trend detection with window=3 moving average
        def moving_avg(vals, w=3):
            return np.convolve(vals, np.ones(w)/w, mode='valid')

        recent = losses[-5:] if n >= 5 else losses
        smoothed = moving_avg(recent) if len(recent) >= 3 else recent
        delta = smoothed[-1] - smoothed[0] if len(smoothed) >= 2 else 0

        if delta < -0.005 * n:
            trend = "decreasing"
        elif abs(delta) < 0.005 * n:
            trend = "plateau"
        elif delta > 0.005 * n:
            trend = "increasing"
        else:
            # Check for oscillation
            diffs = np.diff(losses[-5:] if n >= 5 else losses)
            sign_changes = np.sum(np.abs(np.diff(np.sign(diffs))) > 0)
            trend = "oscillating" if sign_changes >= 2 else "plateau"

        convergence_score = max(0.0, min(1.0, 1.0 - abs(delta / max(abs(smoothed[0]), 1e-8))))
        if trend == "increasing":
            convergence_score = 0.0

        # Loss delta per epoch over last 5
        if n >= 5:
            loss_delta = (losses[-1] - losses[-5]) / 5
        else:
            loss_delta = (losses[-1] - losses[0]) / (n - 1)

        # Infonce trend
        infonce_vals = [m.get("train_infonce", 0) for m in per_epoch_metrics]
        info_delta = infonce_vals[-1] - infonce_vals[-5] if n >= 5 else 0

        # Suggestions
        suggestions = _generate_suggestions(trend, loss_delta, convergence_score,
                                            per_epoch_metrics, infonce_vals)

        return {
            "status": "ok",
            "data": {
                "trend": trend,
                "convergence_score": round(convergence_score, 4),
                "loss_delta": round(loss_delta, 6),
                "loss_start": round(losses[0], 4),
                "loss_end": round(losses[-1], 4),
                "infonce_trend": "decreasing" if info_delta < 0 else "increasing",
                "val_loss": per_epoch_metrics[-1].get("val_loss"),
                "temperature": per_epoch_metrics[-1].get("temperature"),
                "suggestions": suggestions,
            },
            "summary": (
                f"Loss {losses[0]:.4f}→{losses[-1]:.4f} ({trend}), "
                f"convergence={convergence_score:.2f}"
            ),
        }
    except Exception as e:
        return {"status": "error", "data": {}, "summary": "Loss analysis failed", "error": str(e)}


def _generate_suggestions(trend, loss_delta, convergence, metrics, infonce_vals):
    suggestions = []
    last = metrics[-1]
    epoch = last["epoch"]

    if trend == "increasing":
        suggestions.append("Loss rising: reduce LR by 2x or check for data issues")
    if trend == "plateau" and epoch > 10:
        suggestions.append(f"Loss plateaued at epoch {epoch}: reduce LR by 0.5x")
    if convergence > 0.95:
        suggestions.append("Near convergence: consider early stopping")
    if trend == "oscillating":
        suggestions.append("Loss oscillating: reduce LR or increase batch_size for stability")
    if infonce_vals and infonce_vals[-1] < 0.1:
        suggestions.append("InfoNCE very low: model may be overfitting, increase dropout")
    if last.get("val_loss") and last["train_loss"] < 0.5 * last["val_loss"]:
        suggestions.append("Large gap train/val loss: increase weight_decay or dropout")

    if not suggestions:
        suggestions.append("Training progressing normally, no action needed")

    return suggestions
