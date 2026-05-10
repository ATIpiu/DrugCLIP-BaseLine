"""DrugCLIP — Virtual Screening Evaluation (UniMol API).

Badcase analyzer included for agent-driven optimization.
"""

import numpy as np
from typing import List, Dict, Tuple

import torch
from torch.amp import autocast

from ..model import DrugCLIP
from ..data import prepare_molecule, prepare_pocket


def compute_retrieval_metrics(
    sim_scores: np.ndarray,
    active_idx: int = 0,
) -> Dict[str, float]:
    """Compute virtual screening metrics for a single active + N decoys."""
    N = len(sim_scores)
    if N < 2:
        return dict(ef1=0.0, ef5=0.0, auroc=0.5, top1=0, top5=0, top10=0, mrr=0.0,
                    bedroc=0.0, rie=0.0, rank=0, total=N)

    order = np.argsort(-sim_scores)
    rank = int(np.where(order == active_idx)[0][0]) + 1

    k_top1 = max(1, int(np.ceil(N * 0.01)))
    ef1 = (1.0 / 0.01) if rank <= k_top1 else 0.0
    k_top5 = max(1, int(np.ceil(N * 0.05)))
    ef5 = (1.0 / 0.05) if rank <= k_top5 else 0.0

    labels = np.zeros(N, dtype=int)
    labels[active_idx] = 1
    s_min, s_max = sim_scores.min(), sim_scores.max()
    scores_norm = (sim_scores - s_min) / (s_max - s_min + 1e-8)
    desc_order = np.argsort(-scores_norm)
    labels_sorted = labels[desc_order]
    tp = np.cumsum(labels_sorted)
    fp = np.arange(1, N + 1) - tp
    tpr = tp / tp[-1] if tp[-1] > 0 else np.zeros(N)
    fpr = fp / fp[-1] if fp[-1] > 0 else np.zeros(N)
    auroc = float(np.trapezoid(tpr, fpr))

    top1 = 1.0 if rank <= 1 else 0.0
    top5 = 1.0 if rank <= min(5, N) else 0.0
    top10 = 1.0 if rank <= min(10, N) else 0.0
    mrr = 1.0 / rank

    alpha = 20.0
    ra = float(N)
    ri = float(rank)
    bedroc_num = np.exp(-alpha * ri / ra)
    bedroc_denom = (ra * (1 - np.exp(-alpha))) / (np.exp(alpha / ra) - 1)
    bedroc = float(bedroc_num / bedroc_denom) if bedroc_denom > 0 else 0.0
    rie = float((1.0 / ri) / (1.0 / ra)) if ri > 0 else 0.0

    return dict(ef1=ef1, ef5=ef5, auroc=auroc, top1=top1, top5=top5, top10=top10,
                mrr=mrr, bedroc=bedroc, rie=rie, rank=rank, total=N)


# ── Badcase Analyzer ──────────────────────────────────────────────

class BadcaseAnalyzer:
    """Identify worst-performing samples and extract structural features."""

    def __init__(self, top_k: int = 10):
        self.top_k = top_k

    def analyze(
        self,
        per_sample_metrics: List[Dict],
        pocket_features: List[Dict],
        mol_features: List[Dict],
    ) -> Dict:
        """Return structured badcase analysis.

        Args:
            per_sample_metrics: list of metric dicts per sample
            pocket_features: list of {"n_atoms": int, "mean_dist": float, ...}
            mol_features: list of {"n_atoms": int, "n_heavy": int}

        Returns:
            dict with overall stats + badcase list
        """
        n = len(per_sample_metrics)
        if n == 0:
            return {"status": "ok", "data": {"n_samples": 0, "badcases": []},
                    "summary": "No samples for badcase analysis"}

        # Rank by AUROC (or EF1% as tiebreaker)
        ranked = sorted(
            enumerate(per_sample_metrics),
            key=lambda x: (x[1].get("auroc", 0), x[1].get("ef1", 0)),
        )

        badcase_indices = [i for i, _ in ranked[:self.top_k]]
        badcases = []
        for idx in badcase_indices:
            metrics = per_sample_metrics[idx]
            pfeat = pocket_features[idx] if idx < len(pocket_features) else {}
            mfeat = mol_features[idx] if idx < len(mol_features) else {}

            hypothesis = self._generate_hypothesis(metrics, pfeat, mfeat)
            badcases.append(dict(
                sample_idx=idx,
                metrics=metrics,
                pocket_features=pfeat,
                mol_features=mfeat,
                hypothesis=hypothesis,
            ))

        # Aggregate stats
        all_auroc = [m["auroc"] for m in per_sample_metrics]
        return {
            "status": "ok",
            "data": {
                "n_samples": n,
                "badcases": badcases,
                "distribution": {
                    "auroc_mean": float(np.mean(all_auroc)),
                    "auroc_std": float(np.std(all_auroc)),
                    "auroc_p10": float(np.percentile(all_auroc, 10)),
                    "auroc_p50": float(np.percentile(all_auroc, 50)),
                    "ef1_mean": float(np.mean([m.get("ef1", 0) for m in per_sample_metrics])),
                },
            },
            "summary": (
                f"AUROC mean={np.mean(all_auroc):.3f}±{np.std(all_auroc):.3f}, "
                f"{len(badcases)} badcases analyzed"
            ),
        }

    def _generate_hypothesis(
        self, metrics: Dict, pfeat: Dict, mfeat: Dict
    ) -> str:
        parts = []
        p_size = pfeat.get("n_atoms", 0)
        m_size = mfeat.get("n_atoms", 0)
        if p_size > 200 and metrics.get("auroc", 0) < 0.6:
            parts.append("large pocket (>{}) may dilute binding signal".format(p_size))
        if m_size < 10 and metrics.get("auroc", 0) < 0.6:
            parts.append("small ligand (<10 atoms) insufficient for 3D matching")
        if p_size > 100 and m_size < 15:
            parts.append("size ratio pocket:ligand >10:1 may cause embedding mismatch")
        if not parts:
            parts.append("no specific pattern — likely random or low-quality conformer")
        return "; ".join(parts)


# ── Virtual Screening Evaluator (UniMol API) ──────────────────────

class VirtualScreeningEvaluator:
    """Evaluate DrugCLIP on virtual screening retrieval using UniMol API."""

    def __init__(
        self,
        model: DrugCLIP,
        max_pocket_atoms: int = 256,
        num_decoys: int = 999,
        device: str = "cuda",
        mixed_precision: bool = True,
    ):
        self.model = model
        self.max_pocket_atoms = max_pocket_atoms
        self.num_decoys = num_decoys
        self.device = device
        self.mixed_precision = mixed_precision
        self._decoy_pool = None
        self.analyzer = BadcaseAnalyzer(top_k=10)

    def set_decoy_pool(self, decoy_smiles: List[str]):
        """Pre-compute UniMol embeddings for decoy pool."""
        decoy_data = []
        for smi in decoy_smiles:
            mol_data = prepare_molecule(smi)
            if mol_data is not None:
                decoy_data.append(mol_data)
        self._decoy_pool = decoy_data

    def evaluate(
        self,
        pocket_coords_list: List[np.ndarray],
        pocket_elements_list: List[List[str]],
        active_smiles_list: List[str],
    ) -> Dict:
        """Evaluate on validation pocket-ligand pairs.

        Returns structured result dict with badcase analysis.
        """
        self.model.eval()
        n_samples = len(active_smiles_list)

        # Build decoy pool from active SMILES if not set
        if self._decoy_pool is None:
            decoy_data = []
            for smi in active_smiles_list:
                mol_data = prepare_molecule(smi)
                if mol_data is not None:
                    decoy_data.append(mol_data)
            self._decoy_pool = decoy_data

        decoy_pool = self._decoy_pool
        n_pool = len(decoy_pool)

        per_sample_metrics = []
        pocket_features_list = []
        mol_features_list = []

        for i in range(n_samples):
            # Prepare pocket
            pocket_data = prepare_pocket(
                pocket_coords_list[i], pocket_elements_list[i], self.max_pocket_atoms
            )
            pb = {
                k: torch.from_numpy(pocket_data[k]).unsqueeze(0)
                for k in ["tokens", "distances", "edge_types"]
            }

            # Prepare active molecule
            active_data = prepare_molecule(active_smiles_list[i])
            if active_data is None:
                continue

            # Select random decoys
            if n_pool > 1:
                decoy_indices = np.random.choice(
                    n_pool, size=min(self.num_decoys, n_pool - 1), replace=True
                )
                decoy_indices[decoy_indices == i] = (i + 1) % n_pool
                selected_decoys = [decoy_pool[j] for j in decoy_indices]
            else:
                selected_decoys = []

            # Build batch: 1 active + N decoys
            all_mol_data = [active_data] + selected_decoys

            with torch.no_grad():
                with autocast(device_type="cuda", enabled=self.mixed_precision):
                    # Encode pocket
                    pb_dev = {k: v.to(self.device) for k, v in pb.items()}
                    pocket_emb = self.model.encode_pocket(
                        pb_dev["tokens"], pb_dev["distances"], pb_dev["edge_types"]
                    )  # (1, D)

                    # Encode all molecules one by one (variable sizes)
                    mol_embs = []
                    for md in all_mol_data:
                        mb = {
                            k: torch.from_numpy(md[k]).unsqueeze(0).to(self.device)
                            for k in ["tokens", "distances", "edge_types"]
                        }
                        emb = self.model.encode_mol(
                            mb["tokens"], mb["distances"], mb["edge_types"]
                        )
                        mol_embs.append(emb)
                    mol_embs = torch.cat(mol_embs, dim=0)  # (1+K, D)

            sim = (pocket_emb @ mol_embs.T).squeeze(0).cpu().numpy()
            metrics = compute_retrieval_metrics(sim, active_idx=0)
            per_sample_metrics.append(metrics)

            # Collect features for badcase analysis
            pocket_features_list.append(dict(
                n_atoms=len(pocket_data["tokens"]),
                mean_dist=float(pocket_data["distances"].mean()) if pocket_data["distances"].size > 0 else 0,
            ))
            mol_features_list.append(dict(
                n_atoms=len(active_data["tokens"]) if active_data else 0,
            ))

        # Badcase analysis
        badcase_result = self.analyzer.analyze(
            per_sample_metrics, pocket_features_list, mol_features_list
        )

        # Aggregate metrics
        avg = {}
        if per_sample_metrics:
            for key in per_sample_metrics[0]:
                vals = [m[key] for m in per_sample_metrics]
                avg[key] = float(np.mean(vals))

        return {
            "status": "ok",
            "data": {
                "overall_metrics": avg,
                "per_sample_metrics": per_sample_metrics,
                "badcase_analysis": badcase_result["data"],
                "n_samples": n_samples,
            },
            "summary": (
                f"AUROC={avg.get('auroc', 0):.4f}, EF1%={avg.get('ef1', 0):.1f}, "
                f"badcases={len(badcase_result['data']['badcases'])}"
            ),
        }
