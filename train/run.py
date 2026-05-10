"""DrugCLIP Odyssey — Main Entry Point.

Usage:
    # Training only
    python -m train.run --mode train --train-data data/train

    # Inference only (with trained checkpoint)
    python -m train.run --mode inference --ckpt output/checkpoints/best.pt

    # Full pipeline (train + inference)
    python -m train.run --mode full --train-data data/train
"""

# ── Suppress RDKit noise BEFORE any package imports ──────────────
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="rdkit")
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

import argparse
import os
import sys
from pathlib import Path

from .config import Config
from .core import Trainer
from .inference import InferenceEngine


def parse_args():
    p = argparse.ArgumentParser(description="DrugCLIP Odyssey")
    p.add_argument(
        "--mode", choices=["train", "inference", "full", "agent"], default="full",
        help="Run mode: agent = multi-agent optimization loop"
    )
    p.add_argument("--train-data", default="data/train", help="Training data directory")
    p.add_argument("--val-data", default="data/val", help="Validation data directory")
    p.add_argument("--ckpt", default=None, help="Checkpoint path for inference")
    p.add_argument("--benchmark-dir", default="data/benchmark/benchmark",
                   help="Benchmark data directory")
    p.add_argument("--output-dir", default="output", help="Output directory")
    p.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--encoder-layers", type=int, default=8)
    p.add_argument("--encoder-dim", type=int, default=384)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-samples", type=int, default=0,
                   help="Max samples to load (0 = all)")
    return p.parse_args()


def build_config(args) -> Config:
    config = Config()
    config.seed = args.seed
    config.device = args.device
    config.data.train_data_dir = args.train_data
    config.data.val_data_dir = args.val_data
    config.data.benchmark_dir = args.benchmark_dir
    config.data.output_dir = args.output_dir
    config.train.batch_size = args.batch_size
    config.train.epochs = args.epochs
    config.train.lr = args.lr
    config.model.mol.encoder_layers = args.encoder_layers
    config.model.mol.encoder_embed_dim = args.encoder_dim
    config.model.pocket.encoder_layers = args.encoder_layers
    config.model.pocket.encoder_embed_dim = args.encoder_dim
    return config


def main():
    args = parse_args()
    config = build_config(args)

    # Set seed
    import torch
    import numpy as np
    import random
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    checkpoint_path = None

    if args.mode == "agent":
        from agents.llm_agent import LLMMainAgent
        from train.logger import OdysseyLogger
        logger = OdysseyLogger(config.data.output_dir)
        agent = LLMMainAgent(config, logger=logger)
        agent.run()
        return

    if args.mode in ("train", "full"):
        trainer = Trainer(config)
        trainer.train(
            train_dir=config.data.train_data_dir,
            val_dir=config.data.val_data_dir if Path(config.data.val_data_dir).exists() else None,
            max_samples=args.max_samples,
        )
        model_name = config.model.model_name
        checkpoint_path = str(Path(config.data.output_dir) / "models" / model_name / "checkpoints" / "best.pt")

    if args.mode in ("inference", "full"):
        ckpt = args.ckpt or checkpoint_path
        engine = InferenceEngine(config, checkpoint_path=ckpt)
        result_csv = engine.run_all_tasks()
        engine.create_submission_zip()
        print(f"\nResults: {result_csv}")
        print(f"Submission: {Path(config.data.output_dir) / 'result.zip'}")


if __name__ == "__main__":
    main()
