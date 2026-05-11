"""DrugCLIP — Training Loop."""

import signal
import sys
import time
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from tqdm import tqdm

from ..config import Config
from ..model.registry import get_model
from ..data import create_dataloader, CachedPDBbindDataset
from ..logger import OdysseyLogger
from .loss import CombinedLoss


class Trainer:
    """DrugCLIP contrastive learning trainer with structured metric tracking."""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.logger = OdysseyLogger(config.data.output_dir)

        # If pretrained checkpoint provided, inspect architecture before building model
        if config.model.pretrained_path:
            arch = self._inspect_checkpoint(config.model.pretrained_path)
            self._apply_arch(config, arch)

        # Use model registry — supports agent-generated models via config.model_name
        model_name = getattr(config.model, "model_name", None) or "drugclip"
        self.model = get_model(model_name, config.model).to(self.device)

        # Load pretrained weights if provided
        if config.model.pretrained_path:
            ckpt = torch.load(config.model.pretrained_path, map_location=self.device,
                              weights_only=False)
            ckpt_sd = self._extract_state_dict(ckpt)
            remapped = self._remap_state_dict(ckpt_sd, self.model)
            missing, unexpected = self.model.load_state_dict(remapped, strict=False)
            self.logger.log(f"Loaded pretrained: {config.model.pretrained_path}")
            self.logger.log(f"  Missing keys: {len(missing)}  Unexpected keys: {len(unexpected)}")
            if missing:
                # Only log keys that aren't in the extra-head/structural skip list
                significant = [k for k in missing if not any(
                    x in k for x in ['final_head_layer_norm',
                                     'cross_distance', 'holo_distance',
                                     'classification_head', 'fuse_project'])]
                if significant:
                    self.logger.log(f"  Significant missing: {significant}")

        # Apply freezing
        freeze_info = self._apply_freezing()

        # Auto-load atom dicts for pretrained models (compatible tokenization)
        self._mol_atom_dict = None
        self._pocket_atom_dict = None
        if config.model.pretrained_path:
            from pathlib import Path as _Path
            data_dir = _Path(__file__).resolve().parent.parent.parent / "data"
            mol_dict_path = data_dir / "dict_mol.txt"
            pocket_dict_path = data_dir / "dict_pkt.txt"
            if mol_dict_path.exists() and pocket_dict_path.exists():
                from ..data.utils import load_atom_dict
                self._mol_atom_dict = load_atom_dict(str(mol_dict_path))
                self._pocket_atom_dict = load_atom_dict(str(pocket_dict_path))
                # Validate: model's num_atom_types must be >= dict's num_types
                # to prevent GBF embedding index out-of-bounds
                mol_nt = self._mol_atom_dict["num_types"]
                pkt_nt = self._pocket_atom_dict["num_types"]
                if config.model.mol_atom_types < mol_nt:
                    self.logger.log(f"WARNING: mol_atom_types ({config.model.mol_atom_types}) < dict num_types ({mol_nt}), adjusting")
                    config.model.mol_atom_types = mol_nt
                if config.model.pocket_atom_types < pkt_nt:
                    self.logger.log(f"WARNING: pocket_atom_types ({config.model.pocket_atom_types}) < dict num_types ({pkt_nt}), adjusting")
                    config.model.pocket_atom_types = pkt_nt
                self.logger.log(f"Atom dicts loaded: mol={mol_nt} types, pocket={pkt_nt} types")

        self.criterion = CombinedLoss()
        self.optimizer = optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=config.train.new_lr if config.train.new_lr is not None else config.train.lr,
            weight_decay=config.train.weight_decay,
        )
        self.logger.log(
            f"Optimizer: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)} "
            f"trainable / {sum(p.numel() for p in self.model.parameters())} total params"
            f"{' (lr=' + str(config.train.new_lr) + ')' if config.train.new_lr else ''}"
        )

        self.scaler = GradScaler("cuda", enabled=config.train.mixed_precision)
        self.scheduler: Optional[optim.lr_scheduler.LRScheduler] = None
        self.current_epoch = 0
        self.best_loss = float("inf")
        self.epoch_metrics: List[dict] = []
        self._interrupted = False

    # ── Pretrained checkpoint helpers ──────────────────────────────────

    @staticmethod
    def _inspect_checkpoint(path: str) -> dict:
        """Read checkpoint and extract architecture parameters."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        # Determine format: unicore (has 'args') or odyssey (has 'epoch' + 'model_state_dict')
        is_unicore = "args" in ckpt
        sd = Trainer._extract_state_dict(ckpt)

        arch = {}
        if is_unicore:
            args = ckpt["args"]
            arch["mol_encoder_layers"] = getattr(args.mol, "encoder_layers", 15)
            arch["mol_encoder_embed_dim"] = getattr(args.mol, "encoder_embed_dim", 512)
            arch["mol_encoder_ffn_embed_dim"] = getattr(args.mol, "encoder_ffn_embed_dim", 2048)
            arch["mol_encoder_attention_heads"] = getattr(args.mol, "encoder_attention_heads", 64)
            arch["pocket_encoder_layers"] = getattr(args.pocket, "encoder_layers", 15)
            arch["pocket_encoder_embed_dim"] = getattr(args.pocket, "encoder_embed_dim", 512)
            arch["pocket_encoder_ffn_embed_dim"] = getattr(args.pocket, "encoder_ffn_embed_dim", 2048)
            arch["pocket_encoder_attention_heads"] = getattr(args.pocket, "encoder_attention_heads", 64)
            arch["mol_dropout"] = getattr(args.mol, "dropout", 0.1)
            arch["pocket_dropout"] = getattr(args.pocket, "dropout", 0.1)
        else:
            # Odyssey format — infer from state dict shapes
            pass  # fall through to shape inference below

        # Infer from state dict shapes (works for both formats, fills gaps)
        if "mol_model.embed_tokens.weight" in sd:
            arch["mol_atom_types"] = sd["mol_model.embed_tokens.weight"].shape[0]
        if "pocket_model.embed_tokens.weight" in sd:
            arch["pocket_atom_types"] = sd["pocket_model.embed_tokens.weight"].shape[0]

        # Count encoder layers from state dict
        mol_layers = set()
        pocket_layers = set()
        for k in sd:
            if k.startswith("mol_model.encoder.layers."):
                idx = int(k.split(".")[3])
                mol_layers.add(idx)
            elif k.startswith("pocket_model.encoder.layers."):
                idx = int(k.split(".")[3])
                pocket_layers.add(idx)
        if mol_layers:
            arch["mol_encoder_layers"] = max(mol_layers) + 1
        if pocket_layers:
            arch["pocket_encoder_layers"] = max(pocket_layers) + 1

        # Embed dim from fc1 weight shape (fc1: Linear(embed_dim, ffn_dim))
        if "mol_model.encoder.layers.0.fc1.weight" in sd:
            arch["mol_encoder_embed_dim"] = sd["mol_model.encoder.layers.0.fc1.weight"].shape[1]
        if "pocket_model.encoder.layers.0.fc1.weight" in sd:
            arch["pocket_encoder_embed_dim"] = sd["pocket_model.encoder.layers.0.fc1.weight"].shape[1]

        # Attention heads from gbf_proj output dim (gbf_proj maps to attention_heads)
        for prefix, key in [("mol", "mol_model"), ("pocket", "pocket_model")]:
            heads = None
            # Unicore: gbf_proj is NonLinearHead, output dim = linear2.shape[0]
            for gbf_key in sd:
                if gbf_key.startswith(f"{key}.gbf_proj.linear2.weight"):
                    heads = sd[gbf_key].shape[0]
                    break
                elif gbf_key.startswith(f"{key}.gbf_proj.weight"):
                    # Odyssey: gbf_proj is Linear(128, heads)
                    heads = sd[gbf_key].shape[0]
                    break
            if heads is not None:
                arch[f"{prefix}_encoder_attention_heads"] = heads

        # FFN dim from fc1
        if "mol_model.encoder.layers.0.fc1.weight" in sd:
            arch["mol_encoder_ffn_embed_dim"] = sd["mol_model.encoder.layers.0.fc1.weight"].shape[0]
        if "pocket_model.encoder.layers.0.fc1.weight" in sd:
            arch["pocket_encoder_ffn_embed_dim"] = sd["pocket_model.encoder.layers.0.fc1.weight"].shape[0]

        # Project dim
        for proj_prefix in ["mol_project.linear2", "pocket_project.linear2",
                            "mol_project.2", "pocket_project.2"]:
            if proj_prefix + ".weight" in sd:
                arch["project_dim"] = sd[proj_prefix + ".weight"].shape[0]
                break

        arch["is_unicore"] = is_unicore
        return arch

    @staticmethod
    def _extract_state_dict(ckpt: dict) -> dict:
        """Extract state dict from either unicore or odyssey checkpoint format."""
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            # unicore format: ckpt["model"] is the state dict
            # Check it's not a namespace
            return ckpt["model"]
        if "model_state_dict" in ckpt:
            # odyssey format
            return ckpt["model_state_dict"]
        # Maybe bare state dict
        return ckpt

    @staticmethod
    def _apply_arch(config: Config, arch: dict):
        """Apply detected architecture to config (before model construction)."""
        cfg = config.model
        cfg.mol_atom_types = arch.get("mol_atom_types", cfg.mol_atom_types)
        cfg.pocket_atom_types = arch.get("pocket_atom_types", cfg.pocket_atom_types)
        cfg.project_dim = arch.get("project_dim", cfg.project_dim)
        cfg.use_bos_pool = True  # pretrained models use [BOS] pooling

        for prefix, enc in [("mol", cfg.mol), ("pocket", cfg.pocket)]:
            enc.encoder_layers = arch.get(f"{prefix}_encoder_layers", enc.encoder_layers)
            enc.encoder_embed_dim = arch.get(f"{prefix}_encoder_embed_dim", enc.encoder_embed_dim)
            enc.encoder_ffn_embed_dim = arch.get(f"{prefix}_encoder_ffn_embed_dim", enc.encoder_ffn_embed_dim)
            enc.encoder_attention_heads = arch.get(f"{prefix}_encoder_attention_heads", enc.encoder_attention_heads)
            if f"{prefix}_dropout" in arch:
                enc.dropout = arch[f"{prefix}_dropout"]

    @staticmethod
    def _remap_state_dict(ckpt_sd: dict, model: nn.Module) -> dict:
        """Remap unicore-format state dict keys to match our model structure.

        Handles:
        - in_proj.weight/bias → in_proj_weight/in_proj_bias (MHA naming diff)
        - logit_scale [1] → [] (shape squeeze)
        - Skips extra heads not in our model (cross_distance, holo_distance, etc.)
        """
        model_keys = set(model.state_dict().keys())
        remapped = {}

        for ckpt_key, tensor in ckpt_sd.items():
            new_key = ckpt_key

            # in_proj.weight → in_proj_weight (PyTorch MHA naming difference)
            if ".in_proj.weight" in ckpt_key:
                new_key = ckpt_key.replace(".in_proj.weight", ".in_proj_weight")
            elif ".in_proj.bias" in ckpt_key:
                new_key = ckpt_key.replace(".in_proj.bias", ".in_proj_bias")

            if new_key in model_keys:
                model_shape = model.state_dict()[new_key].shape
                if tensor.shape != model_shape:
                    # Handle squeezable dims (e.g., logit_scale [1] → [])
                    if tensor.ndim == 1 and tensor.shape[0] == 1 and model_shape == ():
                        tensor = tensor.squeeze()
                    elif tensor.ndim == 0 and model_shape == (1,):
                        tensor = tensor.unsqueeze(0)
                    else:
                        continue  # shape mismatch — skip
                remapped[new_key] = tensor

        return remapped

    def _apply_freezing(self) -> dict:
        """Apply layer freezing based on config. Returns freeze summary."""
        cfg = self.config.train
        freeze_layers = set(cfg.freeze_encoder_layers)

        frozen_count = 0
        trainable_count = 0

        for name, param in self.model.named_parameters():
            should_freeze = False

            # Check encoder layer index
            for prefix in ["mol_model.encoder.layers.", "pocket_model.encoder.layers."]:
                if name.startswith(prefix):
                    layer_idx = int(name.split(".")[3])
                    if layer_idx in freeze_layers:
                        should_freeze = True
                    break

            # Embedding tables
            if cfg.freeze_embeddings:
                if "embed_tokens" in name:
                    should_freeze = True

            # GBF layers
            if cfg.freeze_gbf:
                if ".gbf." in name or ".gbf_proj." in name:
                    should_freeze = True

            # Projection heads
            if cfg.freeze_project:
                if "mol_project." in name or "pocket_project." in name:
                    should_freeze = True

            if should_freeze:
                param.requires_grad = False
                frozen_count += param.numel()
            else:
                trainable_count += param.numel()

        result = {"frozen": frozen_count, "trainable": trainable_count}
        if frozen_count > 0 or cfg.freeze_embeddings or cfg.freeze_gbf or cfg.freeze_project:
            self.logger.log(
                f"Freeze: {frozen_count:,} frozen / {trainable_count:,} trainable params"
            )
        return result

    def _build_scheduler(self, steps_per_epoch: int):
        cfg = self.config.train
        total_steps = cfg.epochs * steps_per_epoch
        warmup_steps = cfg.warmup_epochs * steps_per_epoch
        if cfg.lr_scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps - warmup_steps)
        elif cfg.lr_scheduler == "step":
            self.scheduler = StepLR(self.optimizer, step_size=steps_per_epoch * 30, gamma=0.5)
        elif cfg.lr_scheduler == "plateau":
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=10)

    def train(self, train_dir: str, val_dir: Optional[str] = None, max_samples: int = 0) -> dict:
        cfg = self.config.train

        self.logger.log_section("DrugCLIP Training")
        self.logger.log_config(self.config)

        train_loader = create_dataloader(
            train_dir, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
            shuffle=True, split="train", seed=self.config.seed, max_samples=max_samples,
            mol_atom_dict=self._mol_atom_dict, pocket_atom_dict=self._pocket_atom_dict,
        )
        self.logger.log(f"Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches/epoch")

        # Validation loader — use train/val split; always drop_last=False for eval
        val_loader = None
        if val_dir and Path(val_dir).exists():
            val_loader = create_dataloader(
                val_dir, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                shuffle=False, split="val", seed=self.config.seed, drop_last=False,
                mol_atom_dict=self._mol_atom_dict, pocket_atom_dict=self._pocket_atom_dict,
            )
            if len(val_loader.dataset) > 0:
                self.logger.log(f"Val: {len(val_loader.dataset)} samples")
            else:
                self.logger.log("Val dir has no valid samples, falling back to train/val split")
                val_loader = None
        if val_loader is None:
            val_loader = create_dataloader(
                train_dir, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                shuffle=False, split="val", seed=self.config.seed, max_samples=max_samples,
                drop_last=False,
                mol_atom_dict=self._mol_atom_dict, pocket_atom_dict=self._pocket_atom_dict,
            )
            if len(val_loader.dataset) > 0:
                self.logger.log(f"Val (from train split): {len(val_loader.dataset)} samples")
            else:
                self.logger.log("Val: no validation samples — skipping per-epoch eval")

        self._build_scheduler(len(train_loader))
        warmup_steps = cfg.warmup_epochs * len(train_loader)
        base_lr = cfg.lr
        self.epoch_metrics = []

        # Ctrl+C handler (only works in main thread)
        _interrupted_flag = [False]
        def _on_interrupt(sig, frame):
            _interrupted_flag[0] = True
        try:
            prev_handler = signal.signal(signal.SIGINT, _on_interrupt)
            prev_terminate = signal.signal(signal.SIGTERM, _on_interrupt)
        except ValueError:
            # Background thread — signal not supported, no handler needed
            prev_handler = prev_terminate = None

        try:
            for epoch in range(1, cfg.epochs + 1):
                if _interrupted_flag[0]:
                    break
                self.current_epoch = epoch
                self.model.train()

                epoch_loss = epoch_nce = epoch_trip = 0.0
                t0 = time.time()

                pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}", leave=False)
                for step, batch in enumerate(pbar):
                    if _interrupted_flag[0]:
                        break
                    global_step = (epoch - 1) * len(train_loader) + step

                    if global_step < warmup_steps:
                        lr = base_lr * (global_step + 1) / warmup_steps
                        for pg in self.optimizer.param_groups:
                            pg["lr"] = lr

                    loss_dict = self._train_step(batch)
                    epoch_loss += loss_dict["loss"]
                    epoch_nce += loss_dict["infonce"]
                    epoch_trip += loss_dict["triplet"]

                    if global_step >= warmup_steps and self.scheduler is not None:
                        self.scheduler.step()

                    pbar.set_postfix(loss=f"{loss_dict['loss']:.4f}", nce=f"{loss_dict['infonce']:.4f}")

            # End of epoch — collect structured metrics
            n_batches = len(train_loader)
            avg_loss = epoch_loss / n_batches
            dt = time.time() - t0
            current_lr = self.optimizer.param_groups[0]["lr"]

            epoch_entry = dict(
                epoch=epoch, train_loss=avg_loss,
                train_infonce=epoch_nce / n_batches,
                train_triplet=epoch_trip / n_batches,
                epoch_time_s=round(dt, 1), lr=current_lr,
                temperature=round(self.model.temperature.item(), 4),
            )

            # Validation — every epoch
            if val_loader is not None and len(val_loader.dataset) > 0:
                val_metrics = self._validate(val_loader)
                epoch_entry.update({f"val_{k}": v for k, v in val_metrics.items()})
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get("loss", avg_loss))

            self.epoch_metrics.append(epoch_entry)

            # Single concise log line per epoch — always visible in terminal
            val_str = ""
            if "val_ef1" in epoch_entry:
                val_str = (
                    f" | Val EF1: {epoch_entry['val_ef1']:.3f}"
                    f" AUROC: {epoch_entry['val_auroc']:.3f}"
                    f" MRR: {epoch_entry['val_mrr']:.3f}"
                    f" R@1: {epoch_entry['val_r1']:.3f}"
                    f" R@5: {epoch_entry['val_r5']:.3f}"
                )
            elif "val_loss" in epoch_entry:
                val_str = f" | Val Loss: {epoch_entry['val_loss']:.4f}"
            msg = (
                f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | "
                f"NCE: {epoch_nce/n_batches:.4f} | Trip: {epoch_trip/n_batches:.4f} | "
                f"Temp: {self.model.temperature:.4f} | Time: {dt:.1f}s{val_str}"
            )
            self.logger.log(msg)
            # Also print directly so terminal sees it even if logger buffers
            print(msg, flush=True)

            # Save checkpoints
            if epoch % cfg.save_interval == 0:
                self.save_checkpoint(is_best=False)
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.save_checkpoint(is_best=True)

        finally:
            if prev_handler is not None:
                signal.signal(signal.SIGINT, prev_handler)
            if prev_terminate is not None:
                signal.signal(signal.SIGTERM, prev_terminate)

        if _interrupted_flag[0]:
            self.save_checkpoint("interrupted")
            self.logger.log("Training interrupted — checkpoint saved, exiting")
            raise KeyboardInterrupt("Training stopped by user")

        self.logger.log(f"Training Complete | Best Loss: {self.best_loss:.4f}")
        self.save_checkpoint("final")

        return dict(
            best_loss=self.best_loss,
            best_epoch=min(
                (e for e in self.epoch_metrics if e["train_loss"] == self.best_loss),
                key=lambda e: e["epoch"], default={"epoch": self.current_epoch}
            ).get("epoch", self.current_epoch),
            epochs_trained=self.current_epoch,
            per_epoch_metrics=self.epoch_metrics,
        )

    def _train_step(self, batch: dict) -> dict:
        self.optimizer.zero_grad()
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

        with autocast(device_type="cuda", enabled=self.config.train.mixed_precision):
            mol_emb, pocket_emb = self.model(batch)
            losses = self.criterion(mol_emb, pocket_emb, self.model.logit_scale)

        self.scaler.scale(losses["loss"]).backward()

        if self.config.train.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        return {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in losses.items()}

    def _validate(self, val_loader) -> dict:
        """Validation: contrastive loss + retrieval metrics (EF1, AUROC, MRR)."""
        import numpy as np

        self.model.eval()
        total_loss = total_nce = total_trip = 0.0
        all_mol, all_pocket = [], []

        with torch.no_grad():
            with autocast(device_type="cuda", enabled=self.config.train.mixed_precision):
                for batch in tqdm(val_loader, desc="  Val  ", leave=False):
                    batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                    mol_emb, pocket_emb = self.model(batch)
                    losses = self.criterion(mol_emb, pocket_emb, self.model.logit_scale)
                    total_loss += losses["loss"].item()
                    total_nce += losses["infonce"]
                    total_trip += losses["triplet"]
                    all_mol.append(mol_emb)
                    all_pocket.append(pocket_emb)

        self.model.train()
        n = max(len(val_loader), 1)

        # Safety: if no batches yielded (e.g. dataset < batch_size), skip metrics
        if len(all_mol) == 0:
            return dict(loss=total_loss / n, infonce=total_nce / n, triplet=total_trip / n,
                        ef1=0.0, auroc=0.5, mrr=0.0, r1=0.0, r5=0.0, r10=0.0)

        # Retrieval metrics
        all_mol = torch.cat(all_mol, dim=0)  # (N, D)
        all_pocket = torch.cat(all_pocket, dim=0)  # (N, D)
        sim = (all_pocket @ all_mol.T).cpu().numpy()  # (N, N)
        N = sim.shape[0]

        ranks = []
        auroc_list = []
        ef1_list = []
        for i in range(N):
            scores = sim[i]
            order = np.argsort(-scores)
            rank = int(np.where(order == i)[0][0]) + 1
            ranks.append(rank)

            # EF1: is true ligand in top 1%?
            k = max(1, int(np.ceil(N * 0.01)))
            ef1_list.append(1.0 if rank <= k else 0.0)

            # AUROC per query
            labels = np.zeros(N, dtype=int)
            labels[i] = 1
            desc = np.argsort(-scores)
            labels_sorted = labels[desc]
            tp = np.cumsum(labels_sorted)
            fp = np.arange(1, N + 1) - tp
            tpr = tp / (tp[-1] + 1e-8)
            fpr = fp / (fp[-1] + 1e-8)
            auroc_list.append(float(np.trapezoid(tpr, fpr)))

        ranks = np.array(ranks)
        metrics = dict(
            loss=total_loss / n,
            infonce=total_nce / n,
            triplet=total_trip / n,
            ef1=float(np.mean(ef1_list)),
            auroc=float(np.mean(auroc_list)),
            mrr=float(np.mean(1.0 / ranks)),
            r1=float(np.mean(ranks <= 1)),
            r5=float(np.mean(ranks <= 5)),
            r10=float(np.mean(ranks <= 10)),
        )
        return metrics

    def save_checkpoint(self, tag: str = "final", is_best: bool = False):
        model_name = self.config.model.model_name
        model_dir = Path(self.config.data.output_dir) / "models" / model_name
        ckpt_dir = model_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save config as JSON
        import json, dataclasses
        config_dict = dataclasses.asdict(self.config)
        (model_dir / "config.json").write_text(
            json.dumps(config_dict, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8")

        # Copy model class file
        import shutil
        model_file = Path(__file__).parent.parent / "model" / f"model_{model_name}.py"
        if model_name != "drugclip" and model_file.exists():
            shutil.copy2(model_file, model_dir / f"model_{model_name}.py")

        # Save weights
        path = ckpt_dir / f"{tag}.pt"
        torch.save(dict(
            epoch=self.current_epoch, model_state_dict=self.model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            best_loss=self.best_loss,
        ), path)
        if is_best:
            shutil.copyfile(path, ckpt_dir / "best.pt")
        self.logger.log(f"Checkpoint: {model_dir}")
