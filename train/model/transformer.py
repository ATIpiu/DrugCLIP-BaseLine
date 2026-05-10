"""Transformer with pair-wise attention bias (no unicore dependency)."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """LayerNorm with fp32-safe internal computation."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        return F.layer_norm(
            x.float(), (self.weight.shape[0],), self.weight.float(),
            self.bias.float(), self.eps
        ).to(dtype)


def get_activation_fn(name: str):
    if name == "relu":
        return F.relu
    elif name == "gelu":
        return F.gelu
    elif name == "tanh":
        return torch.tanh
    raise ValueError(f"Unknown activation: {name}")


class TransformerEncoderLayer(nn.Module):
    """Transformer layer with additive attention bias (not mask)."""

    def __init__(
        self,
        embed_dim: int = 512,
        ffn_embed_dim: int = 2048,
        attention_heads: int = 64,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        activation_fn: str = "gelu",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.head_dim = embed_dim // attention_heads
        assert self.head_dim * attention_heads == embed_dim

        self.self_attn = nn.MultiheadAttention(
            embed_dim, attention_heads, dropout=attention_dropout,
            batch_first=True,
        )
        self.self_attn_layer_norm = LayerNorm(embed_dim)

        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        self.final_layer_norm = LayerNorm(embed_dim)
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.activation_fn = get_activation_fn(activation_fn)

    def forward(self, x, attn_bias=None, padding_mask=None):
        residual = x
        # Apply attention bias if provided
        attn_mask = None
        key_padding_mask = None
        if attn_bias is not None:
            # attn_bias: (B, head, L, L) or (B*head, L, L) — additive
            # MultiheadAttention expects float mask as additive
            attn_mask = attn_bias
        if padding_mask is not None:
            key_padding_mask = padding_mask

        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        return x


class TransformerEncoderWithPair(nn.Module):
    """Transformer that takes pair-wise attention bias from GBF."""

    def __init__(
        self,
        encoder_layers: int = 8,
        embed_dim: int = 384,
        ffn_embed_dim: int = 1536,
        attention_heads: int = 32,
        emb_dropout: float = 0.1,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        max_seq_len: int = 512,
        activation_fn: str = "gelu",
        post_ln: bool = False,
    ):
        super().__init__()
        self.emb_dropout = emb_dropout
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.emb_layer_norm = LayerNorm(embed_dim)
        if not post_ln:
            self.final_layer_norm = LayerNorm(embed_dim)
        else:
            self.final_layer_norm = None
        self.final_head_layer_norm = LayerNorm(attention_heads)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                ffn_embed_dim=ffn_embed_dim,
                attention_heads=attention_heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation_dropout=activation_dropout,
                activation_fn=activation_fn,
            )
            for _ in range(encoder_layers)
        ])

    def forward(self, emb, attn_mask=None, padding_mask=None):
        bsz, seq_len, _ = emb.shape

        x = self.emb_layer_norm(emb)
        x = F.dropout(x, p=self.emb_dropout, training=self.training)

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        # Merge padding_mask into attn_mask
        if attn_mask is not None and padding_mask is not None:
            attn_mask = attn_mask.view(bsz, -1, seq_len, seq_len)
            attn_mask.masked_fill_(
                padding_mask.unsqueeze(1).unsqueeze(2).bool(), float("-inf")
            )
            attn_mask = attn_mask.view(-1, seq_len, seq_len)
            padding_mask = None

        for layer in self.layers:
            x = layer(x, attn_bias=attn_mask, padding_mask=padding_mask)

        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)

        return x, attn_mask
