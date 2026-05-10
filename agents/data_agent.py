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

        # Quick format check — probe a known PDBbind dir (10gs is common)
        known_name = "10gs"
        known_dir = data_dir / known_name
        if known_dir.is_dir() and \
           (known_dir / f"{known_name}_pocket.pdb").exists() and \
           ((known_dir / f"{known_name}_ligand.sdf").exists() or
            (known_dir / f"{known_name}_ligand.mol2").exists()):
            return {
                "status": "ok",
                "data": {"format": "pdbbind"},
                "summary": f"Data OK: PDBbind format confirmed ({known_name})",
            }

        # Fallback: try any subdirectory
        try:
            probe = next(d for d in data_dir.iterdir() if d.is_dir())
            name = probe.name
            if (probe / f"{name}_pocket.pdb").exists():
                return {"status": "ok", "data": {"format": "pdbbind"}, "summary": f"Data OK ({name})"}
        except StopIteration:
            pass

        return {"status": "error", "data": {}, "summary": "Data not in PDBbind format"}
