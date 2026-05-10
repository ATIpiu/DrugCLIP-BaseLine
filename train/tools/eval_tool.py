"""Evaluation tool — runs validation screening + badcase analysis."""

import numpy as np
import torch

from ..config import Config
from ..model import DrugCLIP
from ..data import create_dataloader
from ..core.eval import VirtualScreeningEvaluator


def eval_tool(config: Config, checkpoint_path: str = None) -> dict:
    """Run validation evaluation with badcase analysis.

    Args:
        config: Config with model + data settings
        checkpoint_path: optional path to model checkpoint

    Returns:
        {
            "status": "ok" | "error",
            "data": {
                "overall_metrics": {"auroc": 0.85, "ef1": 12.5, ...},
                "badcase_analysis": {"badcases": [...], "distribution": {...}},
                "n_samples": int,
            },
            "summary": "AUROC=0.8523, EF1%=12.5, 5 badcases",
            "error": None
        }
    """
    try:
        device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        model = DrugCLIP(config.model).to(device)
        model.eval()

        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if "config" in ckpt:
                stored_config = ckpt["config"]
                del model
                model = DrugCLIP(stored_config.model).to(device)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)

        # Load validation data
        val_loader = create_dataloader(
            config.data.train_data_dir,
            batch_size=config.train.batch_size,
            num_workers=0,
            shuffle=False,
            split="val",
            seed=config.seed,
        )

        # Collect pocket coords + elements + SMILES
        pocket_coords_list = []
        pocket_elements_list = []
        active_smiles_list = []

        for batch in val_loader:
            # We need the raw data, not the tokenized version
            # Use the underlying dataset
            pass

        # Access the dataset directly
        dataset = val_loader.dataset
        max_samples = min(len(dataset), 200)

        for i in range(max_samples):
            item = dataset._file_list[i]
            _, smiles, _ = item
            pocket = dataset._pocket_data[i]
            # coords include BOS/EOS padding → strip [0] and [-1]
            coords = pocket["coords"][1:-1]
            pocket_coords_list.append(coords)
            from ..data.utils import ATOM_LIST
            tokens = pocket["tokens"]
            elements = [ATOM_LIST[t] for t in tokens if ATOM_LIST[t] not in ("[PAD]", "[BOS]", "[EOS]", "[MASK]")]
            assert len(elements) == len(coords), f"elements({len(elements)}) != coords({len(coords)}) at sample {i}"
            pocket_elements_list.append(elements)
            active_smiles_list.append(smiles)

        if len(active_smiles_list) == 0:
            return {"status": "error", "data": {}, "summary": "No validation samples", "error": "Empty dataset"}

        evaluator = VirtualScreeningEvaluator(
            model,
            max_pocket_atoms=config.model.max_pocket_atoms,
            num_decoys=min(99, len(active_smiles_list) - 1),
            device=device,
        )
        result = evaluator.evaluate(
            pocket_coords_list, pocket_elements_list, active_smiles_list
        )
        return result

    except Exception as e:
        import traceback
        return {
            "status": "error",
            "data": {},
            "summary": "Eval failed",
            "error": f"{e}\n{traceback.format_exc()}",
        }
