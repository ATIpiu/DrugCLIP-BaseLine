"""DrugCLIP Odyssey — Configuration."""

from dataclasses import dataclass, field
from typing import Optional, List


# ── Atom dictionary for tokenization ──────────────────────────────

# Common elements in drug-like molecules and proteins
ATOM_DICT = [
    "H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I",
    "B", "Si", "Se", "Fe", "Zn", "Mg", "Mn", "Ca", "Na", "K",
    "[PAD]", "[BOS]", "[EOS]", "[MASK]",
]


@dataclass
class EncoderConfig:
    """Shared Transformer encoder hyperparameters (used for both mol & pocket)."""

    encoder_layers: int = 8
    encoder_embed_dim: int = 384
    encoder_ffn_embed_dim: int = 1536
    encoder_attention_heads: int = 32
    dropout: float = 0.2
    emb_dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.0
    pooler_dropout: float = 0.1
    max_seq_len: int = 512
    activation_fn: str = "gelu"
    pooler_activation_fn: str = "tanh"
    post_ln: bool = False
    pocket_radius: float = 10.0  # radius (A) for pocket extraction around ligand


@dataclass
class ModelConfig:
    """DrugCLIP model hyperparameters."""

    model_name: str = "drugclip"  # registry key, agent-generated models use custom names
    mol: EncoderConfig = field(default_factory=EncoderConfig)
    pocket: EncoderConfig = field(default_factory=EncoderConfig)
    gbf_k: int = 128
    project_dim: int = 128
    temperature: float = 0.2
    max_pocket_atoms: int = 256
    dist_threshold: float = 6.0  # distance threshold for pocket extraction


@dataclass
class TrainConfig:
    """Training loop hyperparameters."""

    batch_size: int = 32
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    lr_scheduler: str = "cosine"  # "cosine" | "step" | "plateau"
    grad_clip: float = 1.0
    num_workers: int = 0  # 0 since data is cached in memory
    log_interval: int = 10
    save_interval: int = 10
    mixed_precision: bool = True


@dataclass
class DataConfig:
    """Data loading configuration."""

    train_data_dir: str = "data/train"
    val_data_dir: str = "data/val"
    benchmark_dir: str = "data/benchmark/benchmark"
    output_dir: str = "output"


@dataclass
class Config:
    """Master configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    seed: int = 42
    device: str = "cuda"
    run_name: str = "drugclip-odyssey-v0.2"
