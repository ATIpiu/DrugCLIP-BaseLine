"""Agent mode entry point for DrugCLIP Odyssey.

Usage:
    python -m agents_entry --train-data data/pbpp-2020 --output-dir output --epochs 10
"""

import argparse
import sys
from pathlib import Path

# Ensure train/ is importable
sys.path.insert(0, str(Path(__file__).parent))

from train.config import Config
from train.logger import OdysseyLogger
from agents.main_agent import MainAgent


def parse_args():
    p = argparse.ArgumentParser(description="DrugCLIP Odyssey — Agent Mode")
    p.add_argument("--train-data", default="data/train")
    p.add_argument("--val-data", default="data/val")
    p.add_argument("--benchmark-dir", default="data/benchmark/benchmark")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--encoder-layers", type=int, default=4)
    p.add_argument("--encoder-dim", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-iterations", type=int, default=10,
                   help="Max agent optimization iterations")
    p.add_argument("--reference-doc", default=None,
                   help="Path to reference document for ModelAgent guidance")
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

    # Seed
    import torch, numpy as np, random
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    config = build_config(args)
    logger = OdysseyLogger(config.data.output_dir)

    # Load reference docs if provided
    reference_docs = {}
    if args.reference_doc:
        doc_path = Path(args.reference_doc)
        if doc_path.exists():
            with open(doc_path) as f:
                reference_docs[str(doc_path)] = f.read()

    # Build context
    context = {
        "iteration": 0,
        "config": config,
        "checkpoint": None,
        "history": [],
        "reference_docs": reference_docs,
    }

    # Override max iterations
    MainAgent.MAX_ITERATIONS = args.max_iterations

    main_agent = MainAgent(logger=logger)
    result = main_agent.run(context)

    print(f"\nPipeline result: {result['summary']}")
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
