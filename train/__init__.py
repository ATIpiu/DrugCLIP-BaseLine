"""DrugCLIP Odyssey — Autonomous Virtual Screening Baseline."""

__version__ = "0.2.0"

# ── Kill RDKit noise BEFORE importing submodules ──────────────────
import os as _os
_os.environ["RDKIT_PYTHON_IGNORE_WARNINGS"] = "1"
import warnings as _warnings
_warnings.filterwarnings("ignore", category=DeprecationWarning)
_warnings.filterwarnings("ignore", category=UserWarning, module="rdkit")
from rdkit import RDLogger as _RDLogger
_RDLogger.DisableLog("rdApp.*")

from .config import Config
from .model import DrugCLIP, UniMolEncoder, GaussianLayer
from .core import Trainer
from .inference import InferenceEngine
from .logger import OdysseyLogger
