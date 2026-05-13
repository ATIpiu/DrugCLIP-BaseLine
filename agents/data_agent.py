"""DataAgent — verify/prepare training data for the pipeline."""

import os
from pathlib import Path

from .base_agent import BaseAgent


class DataAgent(BaseAgent):
    """Verify training data directory structure and report statistics."""

    def run(self, context: dict) -> dict:
        data_dir = Path(context["config"].data.train_data_dir)

        if not data_dir.exists():
            return {"status": "error", "data": {}, "summary": f"Data directory not found: {data_dir}"}

        def _is_valid_sample(d: Path) -> bool:
            name = d.name
            has_ligand = (d / f"{name}_ligand.sdf").exists() or (d / f"{name}_ligand.mol2").exists()
            has_protein = (
                (d / f"{name}_pocket.pdb").exists() or
                (d / f"{name}_protein_processed_fix.pdb").exists() or
                (d / f"{name}_protein.pdb").exists()
            )
            return has_ligand and has_protein

        subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
        if not subdirs:
            return {"status": "error", "data": {}, "summary": f"Data directory is empty: {data_dir}"}

        valid = [d for d in subdirs if _is_valid_sample(d)]
        total = len(subdirs)
        n_valid = len(valid)

        if n_valid == 0:
            probe = subdirs[0]
            files = [f.name for f in probe.iterdir()] if probe.exists() else []
            return {
                "status": "error", "data": {},
                "summary": f"No valid samples found. Files in {probe.name}/: {files}",
            }

        return {
            "status": "ok",
            "data": {"format": "pdbbind", "total": total, "valid": n_valid},
            "summary": f"Data OK: {n_valid}/{total} valid samples in {data_dir}",
        }
