"""UniMol-style encoder: token embedding + GBF distance encoding + Transformer."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import EncoderConfig, ATOM_DICT
from .transformer import TransformerEncoderWithPair, LayerNorm


class NonLinearHead(nn.Module):
    """Two-layer MLP matching original DrugCLIP/unimol NonLinearHead.

    Structure: Linear(input_dim→input_dim) → activation → Linear(input_dim→output_dim)
    State-dict keys: linear1.weight/bias, linear2.weight/bias
    """

    def __init__(self, input_dim: int, output_dim: int, activation: str = "relu"):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.linear2 = nn.Linear(input_dim, output_dim)
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.Tanh()

    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    """Gaussian Basis Function to encode inter-atomic distances."""

    def __init__(self, K: int = 128, num_edge_types: int = 1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(num_edge_types, 1)
        self.bias = nn.Embedding(num_edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_type):
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


class UniMolEncoder(nn.Module):
    """Atom-type → embedding → Transformer encoder with GBF distance bias.

    Takes atom type indices + 3D coordinates, outputs per-atom representations.
    """

    def __init__(self, config: EncoderConfig, num_atom_types: int, gbf_k: int = 128,
                 activation_fn: str = "gelu"):
        super().__init__()
        self.padding_idx = num_atom_types - 4  # [PAD] is 4th from end
        self.embed_tokens = nn.Embedding(
            num_atom_types, config.encoder_embed_dim, self.padding_idx
        )
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=config.encoder_layers,
            embed_dim=config.encoder_embed_dim,
            ffn_embed_dim=config.encoder_ffn_embed_dim,
            attention_heads=config.encoder_attention_heads,
            emb_dropout=config.emb_dropout,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            activation_dropout=config.activation_dropout,
            max_seq_len=config.max_seq_len,
            activation_fn=config.activation_fn,
        )

        num_edge_types = num_atom_types * num_atom_types
        self.gbf = GaussianLayer(gbf_k, num_edge_types)
        # NonLinearHead matches original DrugCLIP checkpoint
        self.gbf_proj = NonLinearHead(gbf_k, config.encoder_attention_heads, activation_fn)

    def forward(self, tokens, distances, edge_types):
        """Forward pass.

        Args:
            tokens: (B, N) atom type indices
            distances: (B, N, N) pair-wise distances
            edge_types: (B, N, N) edge type indices (= src_type * n_types + dst_type)

        Returns:
            encoder_repr: (B, N, embed_dim) per-atom representations
        """
        padding_mask = tokens.eq(self.padding_idx)
        x = self.embed_tokens(tokens)

        # GBF distance → attention bias
        n_node = distances.size(-1)
        gbf_feat = self.gbf(distances, edge_types)  # (B, N, N, K)
        gbf_result = self.gbf_proj(gbf_feat)  # (B, N, N, heads)
        attn_bias = gbf_result.permute(0, 3, 1, 2).contiguous()  # (B, heads, N, N)
        attn_bias = attn_bias.view(-1, n_node, n_node)  # (B*heads, N, N)

        encoder_repr, _ = self.encoder(x, attn_mask=attn_bias, padding_mask=padding_mask)
        return encoder_repr
