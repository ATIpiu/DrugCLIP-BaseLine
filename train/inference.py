"""DrugCLIP Odyssey — Benchmark Inference (UniMol API).

Processes all benchmark tasks and generates result.csv with scores.
"""

import csv
import os
import time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.amp import autocast

from .config import Config
from .model import DrugCLIP
from .data import (
    load_manifest,
    load_task_ligands,
    load_task_receptors,
    load_reference_coords,
    extract_pocket_atoms,
    prepare_molecule,
    prepare_pocket,
)

# Fast inference: skip MMFF optimization for speed
from functools import partial as _partial
from .data.dataset import collate_mol_fn
from .logger import OdysseyLogger


class InferenceEngine:
    """Run DrugCLIP inference on all benchmark tasks."""

    def __init__(
        self,
        config: Config,
        checkpoint_path: Optional[str] = None,
    ):
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        self.logger = OdysseyLogger(config.data.output_dir)

        self.model = DrugCLIP(config.model).to(self.device)
        self.model.eval()

        # SMILES → mol_data cache: each ligand prepared once, reused across tasks
        self._mol_cache: dict = {}

        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        else:
            self.logger.log("Warning: No checkpoint loaded — using random weights")

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        # Load full model config from config.json
        import json
        config_dir = Path(path).parent.parent
        config_json = config_dir / "config.json"
        if config_json.exists():
            with open(config_json, encoding="utf-8") as f:
                config_dict = json.load(f)
            # config_dict is nested: {"model": {...}, "train": {...}, ...}
            model_cfg = config_dict.get("model", config_dict)
            mol_cfg = model_cfg.get("mol", {})
            pocket_cfg = model_cfg.get("pocket", {})

            # Apply all encoder arch fields
            for prefix, enc, src in [("mol", self.config.model.mol, mol_cfg),
                                      ("pocket", self.config.model.pocket, pocket_cfg)]:
                for key in ["encoder_layers", "encoder_embed_dim", "encoder_ffn_embed_dim",
                           "encoder_attention_heads", "dropout", "emb_dropout",
                           "attention_dropout", "activation_dropout", "activation_fn"]:
                    if key in src:
                        setattr(enc, key, src[key])

            # Apply model-level fields
            for key in ["project_dim", "gbf_k", "temperature", "mol_atom_types",
                       "pocket_atom_types", "use_bos_pool", "max_pocket_atoms",
                       "dist_threshold", "model_name"]:
                if key in model_cfg:
                    setattr(self.config.model, key, model_cfg[key])

            del self.model
            self.model = DrugCLIP(self.config.model).to(self.device)

            # Load atom dicts for inference data pipeline if pretrained
            if model_cfg.get("pretrained_path"):
                from pathlib import Path as _Path
                ref_data = _Path(__file__).resolve().parent.parent / "ref" / "DrugCLIP-main" / "DrugCLIP-main" / "data"
                mol_dict_file = ref_data / "dict_mol.txt"
                pocket_dict_file = ref_data / "dict_pkt.txt"
                if mol_dict_file.exists() and pocket_dict_file.exists():
                    from .data.utils import load_atom_dict
                    self._mol_atom_dict = load_atom_dict(str(mol_dict_file))
                    self._pocket_atom_dict = load_atom_dict(str(pocket_dict_file))
                    self.logger.log(f"Atom dicts loaded for inference")
                else:
                    self._mol_atom_dict = None
                    self._pocket_atom_dict = None
            else:
                self._mol_atom_dict = None
                self._pocket_atom_dict = None

        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        self.logger.log(f"Loaded checkpoint: {path} (epoch {ckpt['epoch']})")

    def run_all_tasks(self) -> str:
        manifest_path = Path(self.config.data.benchmark_dir) / "manifest.jsonl"
        tasks = load_manifest(str(manifest_path))
        self.logger.log_section("Inference — All Tasks")
        self.logger.log(f"Total tasks: {len(tasks)}")

        results: List[Tuple[str, str, float]] = []
        total_ligands = 0
        t0 = time.time()

        for i, task_info in enumerate(tasks):
            task_id = task_info["task_id"]
            task_dir = Path(self.config.data.benchmark_dir) / "tasks" / task_id

            t_task = time.time()
            scores = self._score_single_task(task_dir, task_info)
            dt = time.time() - t_task

            num = len(scores)
            total_ligands += num
            for ligand_id, score in scores:
                results.append((task_id, ligand_id, score))

            self.logger.log_inference_progress(task_id, num, dt)

            if (i + 1) % 20 == 0:
                self.logger.log(f"Progress: {i+1}/{len(tasks)} tasks done")

        total_time = time.time() - t0
        self._write_result_csv(results)
        self.logger.log_final_summary(
            len(tasks), total_ligands, total_time, model_path=None,
        )
        return str(Path(self.config.data.output_dir) / "result.csv")

    def _score_single_task(
        self, task_dir: Path, task_info: dict
    ) -> List[Tuple[str, float]]:
        receptors = load_task_receptors(str(task_dir), task_info)
        ref_coords = load_reference_coords(str(task_dir), task_info)
        if not receptors:
            return []

        ligand_pairs = load_task_ligands(str(task_dir / task_info["ligand_file"]))
        if not ligand_pairs:
            return []
        ligand_ids, smiles_list = zip(*ligand_pairs)
        ligand_ids = list(ligand_ids)
        smiles_list = list(smiles_list)

        # Prepare all ligands as UniMol batches
        pocket_radius = self.config.model.pocket.pocket_radius
        mol_batches = self._prepare_ligand_batches(smiles_list)

        with torch.no_grad():
            with autocast(device_type="cuda", enabled=self.config.train.mixed_precision):
                # Encode all ligands
                mol_embs = []
                for batch in mol_batches:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    emb = self.model.encode_mol(
                        batch["mol_tokens"], batch["mol_distances"], batch["mol_edge_types"]
                    )
                    mol_embs.append(emb)
                mol_embs = torch.cat(mol_embs, dim=0)  # (N_ligands, D)

                # Encode each receptor pocket and score
                all_scores = []
                for rec_name, rec_coords, rec_elements in receptors:
                    pocket_coords, pocket_elements = extract_pocket_atoms(
                        rec_coords, rec_elements, ref_coords, pocket_radius
                    )
                    if len(pocket_coords) == 0:
                        pocket_coords, pocket_elements = rec_coords, rec_elements

                    pocket_data = prepare_pocket(
                        pocket_coords, pocket_elements,
                        self.config.model.max_pocket_atoms,
                        atom_dict=self._pocket_atom_dict,
                    )
                    pb = {k: torch.from_numpy(pocket_data[k]).unsqueeze(0).to(self.device)
                          for k in ["tokens", "distances", "edge_types"]}
                    pocket_emb = self.model.encode_pocket(
                        pb["tokens"], pb["distances"], pb["edge_types"]
                    )
                    sim = (pocket_emb @ mol_embs.T).squeeze(0).cpu().numpy()
                    all_scores.append(sim)

        if len(all_scores) > 1:
            final_scores = np.max(np.stack(all_scores, axis=0), axis=0)
        else:
            final_scores = all_scores[0]

        return list(zip(ligand_ids, final_scores.tolist()))

    def _prepare_ligand_batches(
        self, smiles_list: List[str], batch_size: int = 512
    ) -> List[dict]:
        """Prepare ligands in batches with multiprocessing + SMILES cache."""
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # Separate: uncached need RDKit (slow), cached are instant
        uncached = [(i, smi) for i, smi in enumerate(smiles_list) if smi not in self._mol_cache]
        total = len(smiles_list)
        cache_hits = total - len(uncached)

        if uncached:
            n_workers = min(os.cpu_count() or 4, 8)
            self.logger.log(f"  Preparing {len(uncached)} ligands ({n_workers} workers)...")
            mol_atom_dict = self._mol_atom_dict  # capture for picklable partial
            _mol_fn = _partial(prepare_molecule, fast=True, atom_dict=mol_atom_dict)
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_mol_fn, smi): (idx, smi)
                           for idx, smi in uncached}
                for f in as_completed(futures):
                    idx, smi = futures[f]
                    mol_data = f.result()
                    if mol_data is None:
                        self._mol_cache[smi] = None
                    else:
                        self._mol_cache[smi] = {
                            "mol_tokens": torch.from_numpy(mol_data["tokens"]),
                            "mol_distances": torch.from_numpy(mol_data["distances"]),
                            "mol_edge_types": torch.from_numpy(mol_data["edge_types"]),
                        }
        elif cache_hits > 0:
            self.logger.log(f"  [cache] all {total} ligands from SMILES cache")

        # Build batches from (now fully populated) cache
        batches = []
        current_batch = []
        for smi in smiles_list:
            mol_entry = self._mol_cache[smi]
            if mol_entry is None:
                mol_tokens = torch.tensor([0, 20, 21, 22], dtype=torch.long)
                mol_dist = torch.zeros(4, 4)
                mol_et = torch.zeros(4, 4, dtype=torch.long)
            else:
                mol_tokens = mol_entry["mol_tokens"]
                mol_dist = mol_entry["mol_distances"]
                mol_et = mol_entry["mol_edge_types"]
            current_batch.append({
                "mol_tokens": mol_tokens,
                "mol_distances": mol_dist,
                "mol_edge_types": mol_et,
            })
            if len(current_batch) >= batch_size:
                batches.append(collate_mol_fn(current_batch))
                current_batch = []
        if current_batch:
            batches.append(collate_mol_fn(current_batch))

        return batches

    def _write_result_csv(self, results: List[Tuple[str, str, float]]):
        csv_path = Path(self.config.data.output_dir) / "result.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task_id", "ligand_id", "score"])
            writer.writerows(results)
        self.logger.log(f"result.csv: {csv_path} ({len(results)} entries)")

    def create_submission_zip(self):
        import shutil
        import zipfile
        output_dir = Path(self.config.data.output_dir)
        zip_path = output_dir / "result.zip"

        # Prefer agent.log (structured audit trail) as result.log for submission
        agent_log = output_dir / "agent.log"
        result_log = output_dir / "result.log"
        if agent_log.exists() and agent_log.stat().st_size > 0:
            shutil.copyfile(agent_log, result_log)
            self.logger.log("result.log: copied from agent.log (agent audit trail)")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in ["result.csv", "result.log"]:
                fpath = output_dir / fname
                if fpath.exists():
                    zf.write(fpath, fname)
        self.logger.log(f"Submission: {zip_path}")
        return str(zip_path)
