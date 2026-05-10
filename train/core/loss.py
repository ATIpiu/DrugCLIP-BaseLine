"""DrugCLIP — Contrastive loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """Symmetric InfoNCE (NT-Xent) loss for contrastive learning."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pocket_emb: torch.Tensor,
        mol_emb: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> torch.Tensor:
        B = pocket_emb.size(0)
        logits = logit_scale.exp() * (pocket_emb @ mol_emb.T)
        labels = torch.arange(B, device=pocket_emb.device)
        loss_p2m = F.cross_entropy(logits, labels)
        loss_m2p = F.cross_entropy(logits.T, labels)
        return (loss_p2m + loss_m2p) / 2.0


class TripletMarginLoss(nn.Module):
    """Optional triplet margin loss for fine-grained optimization."""

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        pocket_emb: torch.Tensor,
        mol_pos: torch.Tensor,
        mol_neg: torch.Tensor,
    ) -> torch.Tensor:
        pos_sim = (pocket_emb * mol_pos).sum(dim=-1)
        neg_sim = (pocket_emb * mol_neg).sum(dim=-1)
        return F.relu(self.margin - pos_sim + neg_sim).mean()


class CombinedLoss(nn.Module):
    """Combine InfoNCE with optional auxiliary losses."""

    def __init__(self, triplet_weight: float = 0.1, triplet_margin: float = 0.5):
        super().__init__()
        self.infonce = InfoNCELoss()
        self.triplet = TripletMarginLoss(margin=triplet_margin)
        self.triplet_weight = triplet_weight

    def forward(
        self,
        pocket_emb: torch.Tensor,
        mol_emb: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> dict:
        loss_nce = self.infonce(pocket_emb, mol_emb, logit_scale)

        B = pocket_emb.size(0)
        sim = pocket_emb @ mol_emb.T
        mask = torch.eye(B, device=sim.device).bool()
        neg_sim = sim.masked_fill(mask, -float("inf"))
        hardest_neg_idx = neg_sim.argmax(dim=1)
        mol_neg = mol_emb[hardest_neg_idx]
        loss_triplet = self.triplet(pocket_emb, mol_emb, mol_neg)

        total = loss_nce + self.triplet_weight * loss_triplet
        return {
            "loss": total,
            "infonce": loss_nce.item(),
            "triplet": loss_triplet.item(),
        }
