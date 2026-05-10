"""DrugCLIP — Training Loop."""

import signal
import sys
import time
from pathlib import Path
from typing import Optional, List

import torch
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

        # Use model registry — supports agent-generated models via config.model_name
        model_name = getattr(config.model, "model_name", None) or "drugclip"
        self.model = get_model(model_name, config.model).to(self.device)
        self.criterion = CombinedLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.train.lr,
            weight_decay=config.train.weight_decay,
        )
        self.scaler = GradScaler("cuda", enabled=config.train.mixed_precision)
        self.scheduler: Optional[optim.lr_scheduler.LRScheduler] = None
        self.current_epoch = 0
        self.best_loss = float("inf")
        self.epoch_metrics: List[dict] = []
        self._interrupted = False

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
        )
        self.logger.log(f"Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches/epoch")

        # Validation loader — use train/val split; always drop_last=False for eval
        val_loader = None
        if val_dir and Path(val_dir).exists():
            val_loader = create_dataloader(
                val_dir, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                shuffle=False, split="val", seed=self.config.seed, drop_last=False,
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
