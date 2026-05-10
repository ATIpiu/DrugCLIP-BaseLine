"""DrugCLIP — Contrastive pocket–molecule alignment with Transformer encoders."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig, ATOM_DICT
from .unimol_encoder import UniMolEncoder


class DrugCLIP(nn.Module):
    """DrugCLIP: dual UniMol encoders + contrastive projection heads."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        num_atom_types = len(ATOM_DICT)

        self.mol_model = UniMolEncoder(config.mol, num_atom_types, config.gbf_k)
        self.pocket_model = UniMolEncoder(config.pocket, num_atom_types, config.gbf_k)

        self.mol_project = nn.Sequential(
            nn.Linear(config.mol.encoder_embed_dim, config.project_dim),
            nn.Tanh(),
            nn.Linear(config.project_dim, config.project_dim),
        )
        self.pocket_project = nn.Sequential(
            nn.Linear(config.pocket.encoder_embed_dim, config.project_dim),
            nn.Tanh(),
            nn.Linear(config.project_dim, config.project_dim),
        )

        self.logit_scale = nn.Parameter(
            torch.tensor(1.0 / config.temperature).log()
        )

    def _pool(self, rep, tokens):
        """Mean pool over non-padding, non-special tokens."""
        from ..data.utils import PAD_IDX, BOS_IDX, EOS_IDX
        mask = (tokens != PAD_IDX) & (tokens != BOS_IDX) & (tokens != EOS_IDX)
        mask = mask.float().unsqueeze(-1)
        pooled = (rep * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled

    def encode_mol(self, tokens, distances, edge_types):
        rep = self.mol_model(tokens, distances, edge_types)
        emb = self.mol_project(self._pool(rep, tokens))
        return F.normalize(emb, dim=-1)

    def encode_pocket(self, tokens, distances, edge_types):
        rep = self.pocket_model(tokens, distances, edge_types)
        emb = self.pocket_project(self._pool(rep, tokens))
        return F.normalize(emb, dim=-1)

    def forward(self, batch):
        mol_emb = self.encode_mol(
            batch["mol_tokens"], batch["mol_distances"], batch["mol_edge_types"])
        pocket_emb = self.encode_pocket(
            batch["pocket_tokens"], batch["pocket_distances"], batch["pocket_edge_types"])
        return mol_emb, pocket_emb

    @property
    def temperature(self):
        return 1.0 / self.logit_scale.exp()

    @temperature.setter
    def temperature(self, value):
        self.logit_scale.data = torch.tensor(1.0 / value).log()
