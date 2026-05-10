"""DrugCLIP — Molecular data processing utilities.

Fingerprint computation, 3D conformer generation, atom tokenization,
PDB/MOL2/SDF parsing, pocket extraction.
"""

from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

# Prefer new MorganGenerator API (RDKit >= 2024.03), fallback to legacy
try:
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    _MORGAN_GEN_CACHE = {}
    _USE_NEW_API = True
except ImportError:
    _USE_NEW_API = False

# ── Atom type mapping ─────────────────────────────────────────────

ATOM_LIST = [
    "H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I",
    "B", "Si", "Se", "Fe", "Zn", "Mg", "Mn", "Ca", "Na", "K",
    "[PAD]", "[BOS]", "[EOS]", "[MASK]",
]

ATOM_TO_IDX = {a: i for i, a in enumerate(ATOM_LIST)}
NUM_ATOM_TYPES = len(ATOM_LIST)
PAD_IDX = ATOM_TO_IDX["[PAD]"]
BOS_IDX = ATOM_TO_IDX["[BOS]"]
EOS_IDX = ATOM_TO_IDX["[EOS]"]


def element_to_atom_idx(element: str) -> int:
    """Map element symbol to atom type index (capitalize first)."""
    e = element.capitalize()
    return ATOM_TO_IDX.get(e, ATOM_TO_IDX["C"])  # default to C


# ── 3D Conformer Generation ───────────────────────────────────────

def smiles_to_conformer(
    smiles: str, add_h: bool = True, fast: bool = False
) -> Optional[Tuple[List[str], np.ndarray]]:
    """Generate a 3D conformer from SMILES via RDKit ETKDG.

    Args:
        smiles: input SMILES
        add_h: add hydrogens before embedding
        fast: if True, skip MMFF optimization (much faster, ~5-10x speedup)

    Returns:
        (elements, coords) or None on failure.
        elements: list of atom element symbols (heavy atoms only)
        coords: (N, 3) float32 array
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.useRandomCoords = True
        params.randomSeed = 42
        status = AllChem.EmbedMolecule(mol, params)
        if status != 0:
            status = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            if status != 0:
                return None
        if not fast:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        return None

    conf = mol.GetConformer()
    elements = [atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    if len(elements) == 0:
        return None

    heavy_idx = [i for i, atom in enumerate(mol.GetAtoms()) if atom.GetAtomicNum() > 1]
    coords = np.array([conf.GetAtomPosition(i) for i in heavy_idx], dtype=np.float32)
    if coords.size == 0:
        return None

    return elements, coords


# ── Atom Tokenization ─────────────────────────────────────────────

def tokenize_atoms(elements: List[str]) -> np.ndarray:
    """Convert element symbols to atom type indices."""
    return np.array([element_to_atom_idx(e) for e in elements], dtype=np.int64)


# ── Distance & Edge Type ──────────────────────────────────────────

def compute_distances(coords: np.ndarray) -> np.ndarray:
    """Pair-wise Euclidean distance matrix.

    Args:
        coords: (N, 3) float array
    Returns:
        (N, N) float32 distance matrix
    """
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1) + 1e-8).astype(np.float32)


def compute_edge_types(tokens: np.ndarray, num_types: int) -> np.ndarray:
    """Pair-wise edge type indices.

    Args:
        tokens: (N,) int array of atom type indices
        num_types: total number of atom types
    Returns:
        (N, N) int64 edge type matrix (src_idx * num_types + dst_idx)
    """
    return (tokens[:, None] * num_types + tokens[None, :]).astype(np.int64)


# ── Prepend/Append BOS/EOS tokens ─────────────────────────────────

def add_bos_eos(
    tokens: np.ndarray, coords: np.ndarray, zero_coord: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Add BOS at start and EOS at end."""
    N = len(tokens)
    new_tokens = np.zeros(N + 2, dtype=np.int64)
    new_tokens[0] = BOS_IDX
    new_tokens[1:-1] = tokens
    new_tokens[-1] = EOS_IDX

    new_coords = np.zeros((N + 2, 3), dtype=np.float32)
    new_coords[1:-1] = coords
    return new_tokens, new_coords


# ── Pocket cropping ────────────────────────────────────────────────

def crop_pocket(
    elements: List[str], coords: np.ndarray, max_atoms: int, seed: int = 42
) -> Tuple[List[str], np.ndarray]:
    """Randomly crop pocket atoms to max_atoms if too many."""
    if len(elements) <= max_atoms:
        return elements, coords
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(elements), max_atoms, replace=False)
    idx = np.sort(idx)
    return [elements[i] for i in idx], coords[idx]


# ── Full molecule preparation pipeline ────────────────────────────

def prepare_molecule(
    smiles: str, max_atoms: Optional[int] = None, fast: bool = False
) -> Optional[dict]:
    """SMILES → tokenized atoms + distances + edge_types.

    Args:
        smiles: input SMILES string
        max_atoms: optional max atom count
        fast: if True, skip MMFF optimization (much faster, good for inference)

    Returns dict with keys: tokens, distances, edge_types, coords (all numpy)
    or None on failure.
    """
    result = smiles_to_conformer(smiles, fast=fast)
    if result is None:
        return None
    elements, coords = result

    tokens = tokenize_atoms(elements)
    tokens, coords = add_bos_eos(tokens, coords)

    distances = compute_distances(coords)
    edge_types = compute_edge_types(tokens, NUM_ATOM_TYPES)

    return {
        "tokens": tokens,
        "distances": distances,
        "edge_types": edge_types,
        "coords": coords,
    }


# ── Full pocket preparation pipeline ──────────────────────────────

def prepare_pocket(
    coords: np.ndarray,
    elements: List[str],
    max_atoms: int = 256,
) -> dict:
    """Pocket atoms → tokenized + distances + edge_types.

    Returns dict with keys: tokens, distances, edge_types, coords (all numpy).
    """
    elements, coords = crop_pocket(elements, coords, max_atoms)

    tokens = tokenize_atoms(elements)
    tokens, coords = add_bos_eos(tokens, coords)

    distances = compute_distances(coords)
    edge_types = compute_edge_types(tokens, NUM_ATOM_TYPES)

    return {
        "tokens": tokens,
        "distances": distances,
        "edge_types": edge_types,
        "coords": coords,
    }


# ── Molecular Fingerprint (legacy, for eval) ──────────────────────

def smiles_to_fingerprint(smiles: str, radius: int = 2, nbits: int = 2048) -> np.ndarray:
    """Convert SMILES to Morgan fingerprint (ECFP-like)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nbits, dtype=np.float32)

    if _USE_NEW_API:
        key = (radius, nbits)
        if key not in _MORGAN_GEN_CACHE:
            _MORGAN_GEN_CACHE[key] = GetMorganGenerator(radius=radius, fpSize=nbits)
        fp = _MORGAN_GEN_CACHE[key].GetFingerprint(mol)
    else:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)

    arr = np.zeros(nbits, dtype=np.float32)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def smiles_list_to_fingerprints(
    smiles_list: List[str], radius: int = 2, nbits: int = 2048
) -> np.ndarray:
    """Batch convert SMILES to fingerprints."""
    fps = np.zeros((len(smiles_list), nbits), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        fps[i] = smiles_to_fingerprint(smi, radius, nbits)
    return fps


# ── PDB Parsing ───────────────────────────────────────────────────

def parse_pdb_atoms(pdb_path: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Parse ATOM/HETATM records from a PDB file."""
    coords = []
    elements = []
    residues = []

    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM  ", "HETATM")):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    continue
                element = line[76:78].strip() or line[12:16].strip()[:1]
                if not element:
                    element = "C"
                res_id = int(line[22:26])
                coords.append([x, y, z])
                elements.append(element.capitalize())
                residues.append(res_id)

    if not coords:
        return np.zeros((0, 3)), [], np.array([], dtype=int)

    return np.array(coords, dtype=np.float32), elements, np.array(residues, dtype=int)


# ── Pocket Extraction ─────────────────────────────────────────────

def extract_pocket_atoms(
    receptor_coords: np.ndarray,
    receptor_elements: List[str],
    ref_coords: np.ndarray,
    radius: float = 10.0,
) -> Tuple[np.ndarray, List[str]]:
    """Extract pocket atoms within `radius` of reference ligand coordinates."""
    if ref_coords.size == 0 or receptor_coords.size == 0:
        return receptor_coords, receptor_elements

    ref_center = ref_coords.mean(axis=0)
    dists = np.linalg.norm(receptor_coords - ref_center, axis=1)
    mask = dists <= radius

    if mask.sum() == 0:
        mask = np.zeros(len(receptor_coords), dtype=bool)
        for rc in ref_coords:
            d = np.linalg.norm(receptor_coords - rc, axis=1)
            mask |= (d <= radius)

    return receptor_coords[mask], [receptor_elements[i] for i in np.where(mask)[0]]


# ── MOL2 Parsing ──────────────────────────────────────────────────

def parse_mol2_coords(mol2_path: str) -> np.ndarray:
    """Extract heavy atom coordinates from a mol2 file."""
    coords = []
    in_atom = False
    with open(mol2_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("@<TRIPOS>ATOM"):
                in_atom = True
                continue
            if line.startswith("@<TRIPOS>BOND"):
                in_atom = False
                continue
            if in_atom and line:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                        coords.append([x, y, z])
                    except (ValueError, IndexError):
                        continue
    return np.array(coords, dtype=np.float32)


# ── SDF / MOL2 → SMILES ──────────────────────────────────────────

def sdf_to_smiles(sdf_path: str) -> str:
    """Extract SMILES from first molecule in an SDF file."""
    supplier = Chem.SDMolSupplier(sdf_path, sanitize=False, removeHs=False)
    for mol in supplier:
        if mol is None:
            continue
        try:
            mol.UpdatePropertyCache(strict=False)
            Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_SIMPLE)
            mol = Chem.RemoveHs(mol)
            return Chem.MolToSmiles(mol)
        except Exception:
            try:
                return Chem.MolToSmiles(mol)
            except Exception:
                continue
    return ""


def mol2_to_smiles(mol2_path: str) -> str:
    """Extract SMILES from a MOL2 file via RDKit."""
    mol = Chem.MolFromMol2File(mol2_path, removeHs=False, sanitize=True)
    if mol is None:
        return ""
    try:
        mol = Chem.RemoveHs(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return ""
