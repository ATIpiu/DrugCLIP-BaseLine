"""DrugCLIP — Training datasets and dataloader factory."""

import os
import signal
import sys
from pathlib import Path
from typing import List

_SILENT = lambda: os.environ.get("DRUGCLIP_SILENT") == "1"

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem

from .utils import (
    parse_pdb_atoms,
    sdf_to_smiles,
    mol2_to_smiles,
    prepare_molecule,
    prepare_pocket,
    NUM_ATOM_TYPES,
)


# ── Cached PDBbind Dataset (Transformer-ready) ────────────────────

class CachedPDBbindDataset(Dataset):
    """PDBbind dataset pre-processing all data into 3D conformer + tokens.

    All data is pre-loaded into memory at init time.
    Shows progress during loading to avoid appearing hung.
    """

    def __init__(
        self,
        data_dir: str,
        max_pocket_atoms: int = 256,
        max_samples: int = 0,
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 42,
        **kwargs,
    ):
        self.max_pocket_atoms = max_pocket_atoms
        self.split = split
        self.val_ratio = val_ratio
        self.seed = seed

        # Build file index — fast, no RDKit yet
        dirs = sorted([d for d in Path(data_dir).iterdir() if d.is_dir()])
        rng = np.random.RandomState(self.seed)
        indices = rng.permutation(len(dirs))
        n_val = max(1, int(len(dirs) * self.val_ratio))
        if self.split == "val":
            indices = indices[:n_val]
        else:
            indices = indices[n_val:]
        dirs = [dirs[i] for i in sorted(indices)]

        # Sample at directory level — only scan what we need
        if max_samples > 0 and max_samples < len(dirs):
            dirs = rng.choice(dirs, max_samples, replace=False).tolist()

        # Suppress RDKit C++ stderr during SDF parsing (bypasses Python logger)
        stderr_backup = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)

        try:
            file_list = self._scan_directories(dirs)
        finally:
            os.dup2(stderr_backup, 2)
            os.close(devnull)
            os.close(stderr_backup)

        # Pre-load with progress & interrupt support
        total = len(file_list)
        self._mol_data: List[dict] = []
        self._pocket_data: List[dict] = []
        valid_files = []
        report_every = max(1, total // 20)

        if not _SILENT():
            print(f"  Loading {total} samples ({self.split})...", flush=True)
        for i, (pocket_pdb, smiles, pdb_id) in enumerate(file_list):
            mol_result = prepare_molecule(smiles)
            if mol_result is None:
                continue

            coords, elements, _ = parse_pdb_atoms(pocket_pdb)
            if len(coords) == 0:
                continue
            pocket_result = prepare_pocket(coords, elements, max_pocket_atoms)

            self._mol_data.append(mol_result)
            self._pocket_data.append(pocket_result)
            valid_files.append((pocket_pdb, smiles, pdb_id))

            if not _SILENT() and (i + 1) % report_every == 0:
                pct = (i + 1) * 100 // total
                print(f"    {pct}% ({i+1}/{total})", flush=True)

        if not _SILENT():
            print(f"  Loaded {len(self._mol_data)} valid samples ({self.split})")
        self._file_list = valid_files

    def _scan_directories(self, dirs):
        """Scan directory structure (noisy — RDKit stderr suppressed by caller)."""
        file_list = []
        for protein_dir in dirs:
            pdb_id = protein_dir.name
            pocket_pdb = protein_dir / f"{pdb_id}_pocket.pdb"
            ligand_sdf = protein_dir / f"{pdb_id}_ligand.sdf"
            ligand_mol2 = protein_dir / f"{pdb_id}_ligand.mol2"

            if not pocket_pdb.exists():
                continue
            smiles = ""
            if ligand_sdf.exists():
                smiles = sdf_to_smiles(str(ligand_sdf))
            if not smiles and ligand_mol2.exists():
                smiles = mol2_to_smiles(str(ligand_mol2))
            if not smiles:
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            file_list.append((str(pocket_pdb), smiles, pdb_id))
        return file_list

    def __len__(self) -> int:
        return len(self._mol_data)

    def __getitem__(self, idx: int):
        mol = self._mol_data[idx]
        pocket = self._pocket_data[idx]
        return {
            "mol_tokens": torch.from_numpy(mol["tokens"]),
            "mol_distances": torch.from_numpy(mol["distances"]),
            "mol_edge_types": torch.from_numpy(mol["edge_types"]),
            "pocket_tokens": torch.from_numpy(pocket["tokens"]),
            "pocket_distances": torch.from_numpy(pocket["distances"]),
            "pocket_edge_types": torch.from_numpy(pocket["edge_types"]),
        }


# ── DataLoader collate (pad variable-length sequences) ────────────

def _pad_1d(tensors, pad_val):
    max_len = max(t.shape[0] for t in tensors)
    padded = torch.full((len(tensors), max_len), pad_val, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        padded[i, :t.shape[0]] = t
    return padded


def _pad_2d(tensors, pad_val):
    max_len = max(t.shape[0] for t in tensors)
    padded = torch.full(
        (len(tensors), max_len, max_len), pad_val, dtype=tensors[0].dtype
    )
    for i, t in enumerate(tensors):
        L = t.shape[0]
        padded[i, :L, :L] = t
    return padded


def collate_mol_fn(batch: list) -> dict:
    """Pad ligand-only batches — used by inference when no pocket data."""
    from .utils import PAD_IDX
    return {
        "mol_tokens": _pad_1d([b["mol_tokens"] for b in batch], PAD_IDX),
        "mol_distances": _pad_2d([b["mol_distances"] for b in batch], 0.0),
        "mol_edge_types": _pad_2d([b["mol_edge_types"] for b in batch], 0),
    }


def collate_fn(batch: list) -> dict:
    """Pad tokens and distances to max length in batch (mol + pocket)."""
    from .utils import PAD_IDX

    mol_tokens = [b["mol_tokens"] for b in batch]
    pocket_tokens = [b["pocket_tokens"] for b in batch]
    mol_dist = [b["mol_distances"] for b in batch]
    pocket_dist = [b["pocket_distances"] for b in batch]
    mol_et = [b["mol_edge_types"] for b in batch]
    pocket_et = [b["pocket_edge_types"] for b in batch]

    return {
        "mol_tokens": _pad_1d(mol_tokens, PAD_IDX),
        "mol_distances": _pad_2d(mol_dist, 0.0),
        "mol_edge_types": _pad_2d(mol_et, 0),
        "pocket_tokens": _pad_1d(pocket_tokens, PAD_IDX),
        "pocket_distances": _pad_2d(pocket_dist, 0.0),
        "pocket_edge_types": _pad_2d(pocket_et, 0),
    }


# ── DataLoader Factory ────────────────────────────────────────────

def _detect_dataset_type(data_dir: str) -> str:
    p = Path(data_dir)
    if not p.exists():
        return "pdbbind"
    dirs = [d for d in p.iterdir() if d.is_dir()]
    if not dirs:
        return "pdbbind"
    first = dirs[0]
    name = first.name
    if (first / f"{name}_pocket.pdb").exists() or (first / f"{name}_protein.pdb").exists():
        return "pdbbind"
    if (first / "actives.smi").exists():
        return "pocket_ligand"
    return "pdbbind"


def create_dataloader(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle: bool = True,
    max_pocket_atoms: int = 256,
    max_samples: int = 0,
    split: str = "train",
    val_ratio: float = 0.1,
    seed: int = 42,
    drop_last: bool = True,
    **kwargs,
) -> DataLoader:
    dataset = CachedPDBbindDataset(
        data_dir=data_dir,
        max_pocket_atoms=max_pocket_atoms,
        max_samples=max_samples,
        split=split,
        val_ratio=val_ratio,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=drop_last,
    )
