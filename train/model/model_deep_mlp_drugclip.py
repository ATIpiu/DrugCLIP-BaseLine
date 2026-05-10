MODEL_NAME = "deep_mlp_drugclip"
DESCRIPTION = "DrugCLIP with deeper projection heads: bottleneck MLP (Linear->ReLU->Linear) + original projection (Linear->Tanh->Linear) for both mol and pocket encoders"

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig, ATOM_DICT
from .unimol_encoder import UniMolEncoder


class Model(nn.Module):
    """DrugCLIP with deeper MLP projection heads.

    In addition to the original DrugCLIP projection (Linear -> Tanh -> Linear),
    this model inserts a bottleneck MLP (Linear -> ReLU -> Linear) immediately
    after the encoder [BOS] output, creating a 4-layer deep projection head.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        num_atom_types = len(ATOM_DICT)

        # ── UniMol encoders (unchanged from DrugCLIP) ──
        self.mol_model = UniMolEncoder(config.mol, num_atom_types, config.gbf_k)
        self.pocket_model = UniMolEncoder(config.pocket, num_atom_types, config.gbf_k)

        embed_dim = config.mol.encoder_embed_dim
        # Bottleneck hidden dimension: compress then expand back
        bottleneck_dim = embed_dim // 2  # 384 → 192 → 384

        # ── NEW: Bottleneck MLP added after encoder output ──
        self.mol_mlp = nn.Sequential(
            nn.Linear(embed_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, embed_dim),
        )
        self.pocket_mlp = nn.Sequential(
            nn.Linear(embed_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, embed_dim),
        )

        # ── Original DrugCLIP projection heads ──
        self.mol_project = nn.Sequential(
            nn.Linear(embed_dim, config.project_dim),
            nn.Tanh(),
            nn.Linear(config.project_dim, config.project_dim),
        )
        self.pocket_project = nn.Sequential(
            nn.Linear(embed_dim, config.project_dim),
            nn.Tanh(),
            nn.Linear(config.project_dim, config.project_dim),
        )

        # ── Learnable temperature ──
        self.logit_scale = nn.Parameter(
            torch.tensor(1.0 / config.temperature).log()
        )

    def encode_mol(self, tokens, distances, edge_types):
        """Encode molecule: tokens + coords → L2-normalized embedding.

        Pipeline:
            UniMolEncoder → [BOS] → bottleneck MLP → projection → L2 norm
        """
        rep = self.mol_model(tokens, distances, edge_types)  # (B, N, D)
        cls_rep = rep[:, 0, :]                                # [BOS] token
        cls_rep = self.mol_mlp(cls_rep)                       # bottleneck MLP (new)
        emb = self.mol_project(cls_rep)                       # original projection
        return F.normalize(emb, dim=-1)

    def encode_pocket(self, tokens, distances, edge_types):
        """Encode pocket: tokens + coords → L2-normalized embedding."""
        rep = self.pocket_model(tokens, distances, edge_types)
        cls_rep = rep[:, 0, :]
        cls_rep = self.pocket_mlp(cls_rep)                    # bottleneck MLP (new)
        emb = self.pocket_project(cls_rep)
        return F.normalize(emb, dim=-1)

    def forward(self, batch):
        """Contrastive forward pass.

        Args:
            batch: dict with keys:
                mol_tokens, mol_distances, mol_edge_types,
                pocket_tokens, pocket_distances, pocket_edge_types

        Returns:
            mol_emb, pocket_emb: both (B, project_dim), L2-normalized
        """
        mol_emb = self.encode_mol(
            batch["mol_tokens"],
            batch["mol_distances"],
            batch["mol_edge_types"],
        )
        pocket_emb = self.encode_pocket(
            batch["pocket_tokens"],
            batch["pocket_distances"],
            batch["pocket_edge_types"],
        )
        return mol_emb, pocket_emb

    @property
    def temperature(self):
        return 1.0 / self.logit_scale.exp()

    @temperature.setter
    def temperature(self, value):
        self.logit_scale.data = torch.tensor(1.0 / value).log()
