"""DrugCLIP Odyssey — Benchmark Inference (UniMol API).

Processes all benchmark tasks and generates result.csv with scores.
"""

import csv
import gc
import hashlib
import os
import pickle
import sqlite3
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict

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

        # Level-1 in-memory cache: SMILES → mol tensor data (populated from disk cache)
        self._mol_cache: dict = {}
        self._mol_atom_dict = None
        self._pocket_atom_dict = None

        # Persistent caches
        cache_dir = Path(config.data.output_dir) / "mol_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Level-1 disk cache (checkpoint-independent): SQLite DB
        # Maps smiles_md5 → pickled {mol_tokens, mol_distances, mol_edge_types} numpy arrays
        self._token_db = self._open_token_db(cache_dir / "tokens.db")
        self._token_pending: int = 0  # uncommitted inserts, flushed every batch

        # Level-2 disk cache (checkpoint-dependent): per-task pickle files
        # Maps smiles_md5 → embedding vector (numpy float32)
        self._ckpt_tag = self._compute_ckpt_tag(checkpoint_path)
        self._emb_cache_dir = cache_dir / "emb" / self._ckpt_tag
        self._emb_cache_dir.mkdir(parents=True, exist_ok=True)

        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        else:
            self.logger.log("Warning: No checkpoint loaded — using random weights")

    # ── Persistent cache helpers ──────────────────────────────────────

    @staticmethod
    def _open_token_db(db_path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")    # concurrent reads while writing
        conn.execute("PRAGMA synchronous=NORMAL")  # faster writes, safe enough
        conn.execute("""
            CREATE TABLE IF NOT EXISTS mol_tokens (
                smiles_hash TEXT PRIMARY KEY,
                data        BLOB NOT NULL
            )
        """)
        conn.commit()
        return conn

    @staticmethod
    def _compute_ckpt_tag(ckpt_path: Optional[str]) -> str:
        if not ckpt_path:
            return "no_ckpt"
        p = Path(ckpt_path)
        if not p.exists():
            return hashlib.md5(ckpt_path.encode()).hexdigest()[:8]
        stat = p.stat()
        sig = f"{ckpt_path}|{stat.st_mtime}|{stat.st_size}"
        return hashlib.md5(sig.encode()).hexdigest()[:8]

    @staticmethod
    def _smiles_key(smiles: str) -> str:
        return hashlib.md5(smiles.encode()).hexdigest()

    def _token_cache_get(self, smiles: str) -> Optional[dict]:
        key = self._smiles_key(smiles)
        row = self._token_db.execute(
            "SELECT data FROM mol_tokens WHERE smiles_hash = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        arr = pickle.loads(row[0])
        return {
            "mol_tokens":    torch.from_numpy(arr["t"]),
            "mol_distances": torch.from_numpy(arr["d"]),
            "mol_edge_types":torch.from_numpy(arr["e"]),
        }

    def _token_cache_put(self, smiles: str, mol_entry: dict):
        """Store mol tensor data to SQLite. mol_entry has torch Tensor values."""
        key = self._smiles_key(smiles)
        blob = pickle.dumps({
            "t": mol_entry["mol_tokens"].numpy(),
            "d": mol_entry["mol_distances"].numpy(),
            "e": mol_entry["mol_edge_types"].numpy(),
        }, protocol=4)
        self._token_db.execute(
            "INSERT OR IGNORE INTO mol_tokens (smiles_hash, data) VALUES (?, ?)",
            (key, blob)
        )
        self._token_pending += 1
        if self._token_pending >= 200:
            self._token_db.commit()
            self._token_pending = 0

    def _token_cache_flush(self):
        if self._token_pending > 0:
            self._token_db.commit()
            self._token_pending = 0

    def _load_emb_cache(self, task_id: str) -> Dict[str, np.ndarray]:
        path = self._emb_cache_dir / f"{task_id}.pkl"
        if path.exists():
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                return {}
        return {}

    def _save_emb_cache(self, task_id: str, emb_dict: Dict[str, np.ndarray]):
        path = self._emb_cache_dir / f"{task_id}.pkl"
        with open(path, "wb") as f:
            pickle.dump(emb_dict, f, protocol=4)

    def __del__(self):
        try:
            self._token_cache_flush()
            self._token_db.close()
        except Exception:
            pass

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
                data_dir = _Path(__file__).resolve().parent.parent / "data"
                mol_dict_file = data_dir / "dict_mol.txt"
                pocket_dict_file = data_dir / "dict_pkt.txt"
                if mol_dict_file.exists() and pocket_dict_file.exists():
                    from .data.utils import load_atom_dict
                    self._mol_atom_dict = load_atom_dict(str(mol_dict_file))
                    self._pocket_atom_dict = load_atom_dict(str(pocket_dict_file))
                    # Ensure model's num_atom_types is not smaller than dict's num_types
                    mol_nt = self._mol_atom_dict["num_types"]
                    pkt_nt = self._pocket_atom_dict["num_types"]
                    if self.config.model.mol_atom_types < mol_nt:
                        self.config.model.mol_atom_types = mol_nt
                    if self.config.model.pocket_atom_types < pkt_nt:
                        self.config.model.pocket_atom_types = pkt_nt
                    self.logger.log(f"Atom dicts loaded for inference (mol={mol_nt}, pocket={pkt_nt})")
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
        csv_path = Path(self.config.data.output_dir) / "result.csv"

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

            # Incremental save every task: crash-safe, never lose results
            self._write_result_csv(results)

            # Free system RAM: clear in-memory token cache every 10 tasks.
            # Disk cache (SQLite + emb pkl) is NOT cleared — next task reloads from disk.
            if (i + 1) % 10 == 0:
                self._mol_cache.clear()
                gc.collect()
                self.logger.log(f"Progress: {i+1}/{len(tasks)} tasks done (mem cache cleared)")
            elif (i + 1) % 5 == 0:
                self.logger.log(f"Progress: {i+1}/{len(tasks)} tasks done")

        total_time = time.time() - t0
        self._write_result_csv(results)
        self.logger.log_final_summary(
            len(tasks), total_ligands, total_time, model_path=None,
        )
        return str(csv_path)

    def _score_single_task(
        self, task_dir: Path, task_info: dict
    ) -> List[Tuple[str, float]]:
        task_id = task_info["task_id"]
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

        pocket_radius = self.config.model.pocket.pocket_radius

        # ── Level-2 embedding cache lookup ───────────────────────────
        emb_cache = self._load_emb_cache(task_id)
        keys = [self._smiles_key(s) for s in smiles_list]

        cached_mask = np.array([k in emb_cache for k in keys], dtype=bool)
        n_cached = cached_mask.sum()
        n_total = len(smiles_list)

        if n_cached == n_total:
            # All embeddings cached → skip RDKit and model forward pass entirely
            self.logger.log(f"  [emb cache] {n_total}/{n_total} ligands fully cached")
            mol_embs_np = np.stack([emb_cache[k] for k in keys], axis=0)
            mol_embs = torch.from_numpy(mol_embs_np).to(self.device)
        else:
            # Need to encode at least some ligands
            if n_cached > 0:
                self.logger.log(
                    f"  [emb cache] {n_cached}/{n_total} hits, encoding {n_total - n_cached} new"
                )

            # Prepare only uncached ligands via token pipeline
            uncached_indices = [i for i, hit in enumerate(cached_mask) if not hit]
            uncached_smiles = [smiles_list[i] for i in uncached_indices]

            mol_batches = self._prepare_ligand_batches(uncached_smiles)

            with torch.no_grad():
                with autocast(device_type="cuda", enabled=self.config.train.mixed_precision):
                    new_emb_parts = []
                    for batch in mol_batches:
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        emb = self.model.encode_mol(
                            batch["mol_tokens"], batch["mol_distances"], batch["mol_edge_types"]
                        )
                        new_emb_parts.append(emb.cpu().numpy())
            new_embs_np = np.concatenate(new_emb_parts, axis=0)  # (n_uncached, D)

            # Write new embeddings into cache dict
            for i, smi_idx in enumerate(uncached_indices):
                emb_cache[keys[smi_idx]] = new_embs_np[i]

            # Save updated embedding cache to disk
            self._save_emb_cache(task_id, emb_cache)

            # Assemble full mol_embs in original order
            mol_embs_np = np.empty((n_total, new_embs_np.shape[1]), dtype=np.float32)
            new_ptr = 0
            for i, k in enumerate(keys):
                if cached_mask[i]:
                    mol_embs_np[i] = emb_cache[k]
                else:
                    mol_embs_np[i] = new_embs_np[new_ptr]
                    new_ptr += 1

            mol_embs = torch.from_numpy(mol_embs_np).to(self.device)

        # ── Pocket encoding & scoring ─────────────────────────────────
        with torch.no_grad():
            with autocast(device_type="cuda", enabled=self.config.train.mixed_precision):
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
        """Prepare ligands in batches.

        Hit order: in-memory cache → SQLite token DB → RDKit (slowest).
        New RDKit results are written back to SQLite for future runs.
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # Step 1: fill in-memory cache from SQLite for SMILES not yet seen this run
        db_hits = 0
        for smi in smiles_list:
            if smi not in self._mol_cache:
                entry = self._token_cache_get(smi)
                if entry is not None:
                    self._mol_cache[smi] = entry
                    db_hits += 1

        # Step 2: remaining SMILES need RDKit conformer generation
        uncached = [(i, smi) for i, smi in enumerate(smiles_list) if smi not in self._mol_cache]
        total = len(smiles_list)
        mem_hits = total - len(uncached) - db_hits

        if uncached:
            n_workers = min(os.cpu_count() or 2, 2)
            self.logger.log(
                f"  Preparing {len(uncached)} ligands via RDKit "
                f"({n_workers} workers, mem={mem_hits} db={db_hits} new={len(uncached)})..."
            )
            mol_atom_dict = self._mol_atom_dict
            _mol_fn = _partial(prepare_molecule, fast=True, atom_dict=mol_atom_dict)
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_mol_fn, smi): (idx, smi) for idx, smi in uncached}
                for f in as_completed(futures):
                    idx, smi = futures[f]
                    mol_data = f.result()
                    if mol_data is None:
                        self._mol_cache[smi] = None
                    else:
                        entry = {
                            "mol_tokens":    torch.from_numpy(mol_data["tokens"]),
                            "mol_distances": torch.from_numpy(mol_data["distances"]),
                            "mol_edge_types":torch.from_numpy(mol_data["edge_types"]),
                        }
                        self._mol_cache[smi] = entry
                        self._token_cache_put(smi, entry)  # persist to SQLite
            self._token_cache_flush()
        else:
            self.logger.log(
                f"  [cache] {total} ligands (mem={mem_hits} db={db_hits})"
            )

        # Step 3: build collated batches
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
