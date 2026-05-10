"""DrugCLIP — Benchmark data loading utilities."""

import csv
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .utils import parse_pdb_atoms, parse_mol2_coords


def load_manifest(manifest_path: str) -> List[dict]:
    """Load manifest.jsonl as list of task dicts."""
    tasks = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def load_task_ligands(ligands_csv: str) -> List[Tuple[str, str]]:
    """Load ligands.csv, return list of (ligand_id, smiles)."""
    pairs = []
    with open(ligands_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append((row["ligand_id"], row["smiles"]))
    return pairs


def load_task_receptors(
    task_dir: str, task_info: dict
) -> List[Tuple[str, np.ndarray, List[str]]]:
    """Load all receptor structures for a task.

    Returns:
        list of (receptor_name, coords, elements)
    """
    receptors = []
    task_path = Path(task_dir)
    for rel_path in task_info["receptor_files"]:
        full_path = task_path / rel_path
        if not full_path.exists():
            continue
        if full_path.suffix == ".pdb":
            coords, elements, _ = parse_pdb_atoms(str(full_path))
        elif full_path.suffix == ".mol2":
            coords = parse_mol2_coords(str(full_path))
            elements = ["C"] * len(coords)
        else:
            continue
        receptors.append((rel_path, coords, elements))
    return receptors


def load_reference_coords(task_dir: str, task_info: dict) -> np.ndarray:
    """Load reference ligand coordinates for pocket extraction.

    Returns:
        (M, 3) concatenated reference coords, or empty array.
    """
    all_coords = []
    task_path = Path(task_dir)
    for rel_path in task_info.get("reference_ligand_files", []):
        full_path = task_path / rel_path
        if full_path.exists() and full_path.suffix == ".mol2":
            coords = parse_mol2_coords(str(full_path))
            if coords.size > 0:
                all_coords.append(coords)
    if all_coords:
        return np.concatenate(all_coords, axis=0)
    return np.zeros((0, 3), dtype=np.float32)
