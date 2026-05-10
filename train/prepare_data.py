"""DrugCLIP Odyssey — Training Data Preparation.

Prepares training data from PDBbind or similar protein-ligand datasets.

Expected PDBbind structure (refined set):
    PDBbind/
        index/INDEX_general_PL_data.<year>
        refined-set/
            <pdb_id>/
                <pdb_id>_protein.pdb
                <pdb_id>_ligand.mol2  (or .sdf)

Output structure (for DrugCLIP training):
    data/train/
        <pdb_id>/
            receptor.pdb          (copied protein)
            ref_ligand.mol2       (copied ligand, used for pocket extraction)
            actives.smi           (SMILES of the active ligand)
            protein.pdb           (alternative name)
"""

import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

from rdkit import Chem
from rdkit.Chem import MolFromSmiles, MolToSmiles


def parse_pdbbind_index(index_path: str) -> List[Tuple[str, float, str]]:
    """Parse PDBbind INDEX file.

    Format: PDB_code resolution release_year -logKd/Ki Kd/Ki reference ligand_name
    Returns: list of (pdb_code, pK, ligand_name)
    """
    entries = []
    with open(index_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 5:
                pdb = parts[0]
                try:
                    pk = float(parts[3])
                except ValueError:
                    pk = 0.0
                entries.append((pdb, pk, parts[4] if len(parts) > 4 else ""))
    return entries


def mol2_to_smiles(mol2_path: str) -> str:
    """Extract SMILES from mol2 file via RDKit."""
    mol = Chem.MolFromMol2File(mol2_path, removeHs=False)
    if mol is None:
        return ""
    mol = Chem.RemoveHs(mol)
    return MolToSmiles(mol)


def sdf_to_smiles(sdf_path: str) -> str:
    """Extract first molecule SMILES from SDF."""
    supplier = Chem.SDMolSupplier(sdf_path)
    for mol in supplier:
        if mol is not None:
            mol = Chem.RemoveHs(mol)
            return MolToSmiles(mol)
    return ""


def prepare_pdbbind(
    pdbbind_dir: str,
    output_dir: str = "data/train",
    subset: str = "refined-set",
    min_pk: float = 0.0,
    max_pk: float = 15.0,
    verbose: bool = True,
) -> int:
    """Convert PDBbind refined-set to DrugCLIP training format.

    Args:
        pdbbind_dir: Path to PDBbind root (contains refined-set/, index/)
        output_dir: Output directory for training data
        subset: Subdirectory name ("refined-set" or "general-set")
        min_pk, max_pk: Binding affinity filter range
        verbose: Print progress

    Returns:
        Number of successfully converted entries
    """
    pdbbind = Path(pdbbind_dir)
    subset_dir = pdbbind / subset
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Find index file
    index_files = list(pdbbind.glob("index/INDEX_general_PL*.????"))
    if not index_files:
        index_files = list(pdbbind.glob("**/INDEX_general_PL*"))
    index_path = str(index_files[0]) if index_files else None

    entries = []
    if index_path:
        entries = parse_pdbbind_index(index_path)
        # Filter by affinity
        entries = [(p, pk, n) for p, pk, n in entries if min_pk <= pk <= max_pk]

    count = 0
    for pdb_dir in sorted(subset_dir.iterdir()):
        if not pdb_dir.is_dir():
            continue
        pdb_id = pdb_dir.name[:4].lower()  # Standardize to lowercase

        # Find protein PDB
        protein_files = list(pdb_dir.glob("*_protein.pdb")) + list(pdb_dir.glob("*.pdb"))
        if not protein_files:
            continue
        protein_pdb = protein_files[0]

        # Find ligand file
        ligand_files = (
            list(pdb_dir.glob("*_ligand.mol2"))
            + list(pdb_dir.glob("*.mol2"))
            + list(pdb_dir.glob("*_ligand.sdf"))
            + list(pdb_dir.glob("*.sdf"))
        )
        if not ligand_files:
            continue
        ligand_file = ligand_files[0]

        # Extract SMILES
        if ligand_file.suffix == ".mol2":
            smiles = mol2_to_smiles(str(ligand_file))
        else:
            smiles = sdf_to_smiles(str(ligand_file))

        if not smiles:
            continue

        # Create output directory
        out = output / pdb_id
        out.mkdir(exist_ok=True)

        # Copy files
        shutil.copy2(protein_pdb, out / "receptor.pdb")
        shutil.copy2(ligand_file, out / "ref_ligand.mol2")

        # Write active SMILES
        with open(out / "actives.smi", "w") as f:
            f.write(f"{smiles}\t{pdb_id}_active\n")

        if verbose and count < 5:
            print(f"  {pdb_id}: {smiles[:50]}...")

        count += 1

    print(f"Prepared {count} training samples → {output}")
    return count


def create_mock_data(output_dir: str = "data/train", num_samples: int = 10):
    """Create mock training data for pipeline testing.

    Generates random pocket coords and SMILES for development purposes.
    """
    import numpy as np

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    mock_smiles = [
        "Cc1ccccc1", "CC(=O)Nc1ccc(O)cc1", "O=C(O)c1ccccc1",
        "CN1C(=O)NC(=O)C1", "c1ccncc1", "CC(C)Cc1ccc(Cl)cc1",
        "COc1ccc(CCN)cc1", "O=C1NC(=O)c2ccccc21", "CN(C)c1ccc(cc1)C=O",
        "Cc1c(Cl)cccc1N", "O=S(=O)(c1ccccc1)N", "N#Cc1ccccc1",
        "Cc1noc(C)n1", "COc1ccccc1OC", "FC(F)(F)c1ccccc1",
    ]

    for i in range(num_samples):
        protein_dir = output / f"mock_protein_{i:04d}"
        protein_dir.mkdir(exist_ok=True)

        # Generate random pocket atoms (~50 heavy atoms in a 15A sphere)
        n_atoms = 30 + np.random.randint(0, 40)
        r = 7.0
        phi = np.random.uniform(0, 2 * np.pi, n_atoms)
        costheta = np.random.uniform(-1, 1, n_atoms)
        theta = np.arccos(costheta)
        radii = r * np.random.uniform(0, 1, n_atoms) ** (1 / 3)
        x = radii * np.sin(theta) * np.cos(phi)
        y = radii * np.sin(theta) * np.sin(phi)
        z = radii * np.cos(theta)

        lines = []
        elements = ["C", "N", "O", "S"]
        for j in range(n_atoms):
            elem = elements[j % len(elements)]
            lines.append(
                f"ATOM  {j+1:5d}  {elem:<3s} UNK     1    "
                f"{x[j]:8.3f}{y[j]:8.3f}{z[j]:8.3f}"
                f"  1.00  0.00          {elem:<2s}  "
            )

        with open(protein_dir / "receptor.pdb", "w") as f:
            f.write("\n".join(lines))

        # Random ref ligand coords
        n_ref = 5 + np.random.randint(0, 5)
        ref_lines = [
            "@<TRIPOS>MOLECULE",
            "ref",
            f"  {n_ref}    0    0    0    0",
            "SMALL",
            "NO_CHARGES",
            "",
            "@<TRIPOS>ATOM",
        ]
        for j in range(n_ref):
            ref_lines.append(
                f"     {j+1} C{j+1}      "
                f"{np.random.uniform(-3,3):8.4f}"
                f"{np.random.uniform(-3,3):8.4f}"
                f"{np.random.uniform(-3,3):8.4f}"
                f" C.3       1  UNL1         0.0000"
            )
        ref_lines.extend(["@<TRIPOS>BOND"])

        with open(protein_dir / "ref_ligand.mol2", "w") as f:
            f.write("\n".join(ref_lines))

        smi = mock_smiles[i % len(mock_smiles)]
        with open(protein_dir / "actives.smi", "w") as f:
            f.write(f"{smi}\tactive_{i}\n")

    print(f"Created {num_samples} mock training samples → {output}")
    print("Use this ONLY for pipeline testing — real training needs PDBbind data.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Prepare DrugCLIP training data")
    p.add_argument("--source", choices=["pdbbind", "mock"], default="mock")
    p.add_argument("--pdbbind-dir", default="PDBbind")
    p.add_argument("--output-dir", default="data/train")
    p.add_argument("--subset", default="refined-set")
    p.add_argument("--num-mock", type=int, default=10)
    args = p.parse_args()

    if args.source == "pdbbind":
        if not Path(args.pdbbind_dir).exists():
            print(f"Error: PDBbind directory not found: {args.pdbbind_dir}")
            sys.exit(1)
        prepare_pdbbind(args.pdbbind_dir, args.output_dir, args.subset)
    elif args.source == "mock":
        create_mock_data(args.output_dir, args.num_mock)
