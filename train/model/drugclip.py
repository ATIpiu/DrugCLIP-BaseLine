"""DrugCLIP — Contrastive pocket–molecule alignment with Transformer encoders."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .unimol_encoder import UniMolEncoder, NonLinearHead


class DrugCLIP(nn.Module):
    """DrugCLIP: dual UniMol encoders + contrastive projection heads."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.mol_model = UniMolEncoder(config.mol, config.mol_atom_types, config.gbf_k,
                                       activation_fn=config.mol.activation_fn)
        self.pocket_model = UniMolEncoder(config.pocket, config.pocket_atom_types, config.gbf_k,
                                          activation_fn=config.pocket.activation_fn)

        # NonLinearHead projection (matches original DrugCLIP checkpoint)
        self.mol_project = NonLinearHead(
            config.mol.encoder_embed_dim, config.project_dim, activation="relu")
        self.pocket_project = NonLinearHead(
            config.pocket.encoder_embed_dim, config.project_dim, activation="relu")

        self.logit_scale = nn.Parameter(
            torch.tensor(1.0 / config.temperature).log()
        )

        # Per-encoder special token indices: PAD/BOS/EOS are always last 4 tokens
        self._mol_pad = config.mol_atom_types - 4
        self._mol_bos = config.mol_atom_types - 3
        self._mol_eos = config.mol_atom_types - 2
        self._pocket_pad = config.pocket_atom_types - 4
        self._pocket_bos = config.pocket_atom_types - 3
        self._pocket_eos = config.pocket_atom_types - 2

    def _pool(self, rep, tokens, pad_idx, bos_idx, eos_idx):
        """Mean pool over non-padding, non-special tokens."""
        mask = (tokens != pad_idx) & (tokens != bos_idx) & (tokens != eos_idx)
        mask = mask.float().unsqueeze(-1)
        pooled = (rep * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled

    def _bos_pool(self, rep):
        """Use [BOS] token (index 0) representation — matches original DrugCLIP."""
        return rep[:, 0, :]

    def _get_pooled(self, rep, tokens, pad_idx, bos_idx, eos_idx):
        if self.config.use_bos_pool:
            return self._bos_pool(rep)
        return self._pool(rep, tokens, pad_idx, bos_idx, eos_idx)

    def encode_mol(self, tokens, distances, edge_types):
        rep = self.mol_model(tokens, distances, edge_types)
        emb = self.mol_project(
            self._get_pooled(rep, tokens, self._mol_pad, self._mol_bos, self._mol_eos))
        return F.normalize(emb, dim=-1)

    def encode_pocket(self, tokens, distances, edge_types):
        rep = self.pocket_model(tokens, distances, edge_types)
        emb = self.pocket_project(
            self._get_pooled(rep, tokens, self._pocket_pad, self._pocket_bos, self._pocket_eos))
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
