import json
import os
import random
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import UNet, CompositeLogitHuberTVLoss
from prep import InputDataset


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Seed all RNGs for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_worker_init_fn(seed: int):
    """Per-worker seed function for DataLoader workers."""
    def worker_init_fn(worker_id: int) -> None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    return worker_init_fn


# ---------------------------------------------------------------------------
# Integrated Gradients
# ---------------------------------------------------------------------------

def compute_ig_importance(
    model: torch.nn.Module,
    val_loader: DataLoader,
    predictor_names: Sequence[str],
    sample_batches: int = 8,
    steps: int = 32,
    save_dir: str = ".",
) -> None:
    """
    Integrated Gradients channel importance.

    Saves:
      {save_dir}/ig_channel_importance.npy

    Prints:
      Channels ordered by importance (descending).
    """
    model.eval()
    device = next(model.parameters()).device

    C_expected = len(predictor_names)
    total_importance = np.zeros((C_expected,), dtype=np.float64)
    n_used = 0

    for b_idx, (xb, yb) in enumerate(val_loader):
        if b_idx >= int(sample_batches):
            break

        xb = xb.to(device)
        yb = yb.to(device)

        if xb.dim() != 4:
            raise ValueError(f"Expected xb to have shape (B,C,H,W); got {tuple(xb.shape)}")
        if xb.shape[1] != C_expected:
            raise ValueError(
                f"predictor_names length ({C_expected}) != xb channels ({xb.shape[1]})."
            )

        if yb.dim() == 3:
            yb = yb.unsqueeze(1)
        mask = (yb != -1).float()  # (B,1,H,W)

        if mask.sum().item() <= 0:
            continue

        baseline = torch.zeros_like(xb)
        for i, name in enumerate(predictor_names):
            if name == "scf" or name.startswith("scf_lag"):
                baseline[:, i, :, :] = -1.0

        # IG: (x - x0) * E_alpha[dF/dx at x0 + alpha*(x-x0)]
        grads_sum = torch.zeros_like(xb)

        for s in range(1, int(steps) + 1):
            alpha = float(s) / float(steps)
            x_interp = baseline + alpha * (xb - baseline)
            x_interp.requires_grad_(True)

            model.zero_grad(set_to_none=True)

            y_logits = model(x_interp)
            if y_logits.dim() == 3:
                y_logits = y_logits.unsqueeze(1)

            y_prob = torch.sigmoid(y_logits)
            score = (y_prob * mask).sum() / mask.sum().clamp_min(1.0)

            score.backward()
            grads_sum += x_interp.grad.detach()

        avg_grads = grads_sum / float(steps)
        ig = (xb - baseline) * avg_grads  # (B,C,H,W)

        batch_importance = ig.abs().mean(dim=(0, 2, 3)).detach().cpu().numpy()
        total_importance += batch_importance.astype(np.float64)
        n_used += 1

    if n_used == 0:
        print("IG importance: no usable batches (no supervised pixels found). Nothing saved.")
        return

    mean_importance = (total_importance / float(n_used)).astype(np.float32)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(save_dir, "ig_channel_importance.npy")
    np.save(out_path, mean_importance)

    order = np.argsort(-mean_importance)
    print("\nIntegrated Gradients channel importance (descending):")
    for rank, i in enumerate(order, start=1):
        print(f"{rank:>2}. {predictor_names[i]}: {mean_importance[i]:.6e}")
    print(f"\nSaved: {out_path}\n")


# ---------------------------------------------------------------------------
# SCF bin weights
# ---------------------------------------------------------------------------

def compute_scf_bin_weights(train_ds, n_bins=10, eps=1e-6, clip=(0.25, 20.0), gamma=1.5):
    """
    Compute inverse-frequency weights for supervised SCF bins.
    Uses ONLY train_ds.targets (y != -1).
    """
    y = train_ds.targets[..., 0]  # (T,H,W)
    m = (y != -1.0) & np.isfinite(y)
    y_sup = np.clip(y[m], 0.0, 1.0)

    edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float32)
    bin_idx = np.digitize(y_sup, edges[1:-1], right=False)

    counts = np.bincount(bin_idx, minlength=n_bins).astype(np.int64)
    freq = counts / (counts.sum() + eps)

    weights = 1.0 / (freq + eps)
    weights /= weights.mean()

    if clip is not None:
        weights = np.clip(weights, clip[0], clip[1])

    return edges.astype(np.float32), weights.astype(np.float32), counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    with open("config.json", "r") as f:
        cfg = json.load(f)

    # --- Seed everything first, before any dataset/model construction -------
    seed = cfg["dataset"]["seed"]
    seed_everything(seed)

    # --- Config -------------------------------------------------------------
    data_path            = cfg["paths"]["data"]
    predictors           = cfg["dataset"]["predictors"]
    lag_days             = cfg["dataset"]["lag_days"]
    test_frac            = cfg["dataset"]["test_frac"]
    split_method         = cfg["dataset"]["split_method"]
    scf_quality_threshold= cfg["dataset"]["scf_quality_threshold"]
    cloud_min_blobs      = cfg["dataset"]["cloud_min_blobs"]
    cloud_max_blobs      = cfg["dataset"]["cloud_max_blobs"]
    cloud_min_radius     = cfg["dataset"]["cloud_min_radius"]
    cloud_max_radius     = cfg["dataset"]["cloud_max_radius"]
    persistence_days     = cfg["dataset"]["persistence_days"]
    repeat_edge          = cfg["dataset"]["repeat_edge"]
    scf_dropout_p        = cfg["dataset"]["scf_dropout_p"]

    batch_size           = int(cfg["training"]["batch_size"])
    num_epochs           = int(cfg["training"]["num_epochs"])
    early_stop_patience  = int(cfg["training"]["early_stop_patience"])
    grad_clip_max_norm   = float(cfg["training"]["grad_clip_max_norm"])
    use_amp              = bool(cfg["training"]["use_amp"])

    # --- Datasets -----------------------------------------------------------
    ds_kwargs = dict(
        test_frac=test_frac,
        split_method=split_method,
        seed=seed,
        predictors=predictors,
        lag_days=lag_days,
        repeat_edge=repeat_edge,
        scf_quality_threshold=scf_quality_threshold,
        cloud_min_blobs=cloud_min_blobs,
        cloud_max_blobs=cloud_max_blobs,
        cloud_min_radius=cloud_min_radius,
        cloud_max_radius=cloud_max_radius,
        persistence_days=persistence_days,
    )

    train_ds = InputDataset(data_path, split="train", scf_dropout_p=scf_dropout_p, **ds_kwargs)
    val_ds   = InputDataset(data_path, split="val",   scf_dropout_p=0.0,           **ds_kwargs)

    # --- DataLoaders (seeded workers) ---------------------------------------
    worker_init = make_worker_init_fn(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=worker_init,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=worker_init,
    )

    # --- Model --------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = UNet(in_channels=train_ds[0][0].shape[0]).to(device)

    # --- Loss ---------------------------------------------------------------
    edges, bin_w, counts = compute_scf_bin_weights(train_ds, n_bins=10)
    loss_fn = CompositeLogitHuberTVLoss(
        bin_edges=edges,
        bin_weights=bin_w,
        alpha=0.35,
        bias_penalty=3.0,
    ).to(device)

    # --- Optimizer ----------------------------------------------------------
    opt_cfg = cfg["optimizer"]
    if opt_cfg["name"].lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(opt_cfg["lr"]),
            weight_decay=float(opt_cfg["weight_decay"]),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {opt_cfg['name']}")

    # --- LR scheduler -------------------------------------------------------
    sched_cfg = cfg["scheduler"]
    if sched_cfg.get("use_scheduler", True):
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=sched_cfg.get("mode", "min"),
            patience=int(sched_cfg["patience"]),
            factor=float(sched_cfg["factor"]),
            min_lr=float(sched_cfg["min_lr"]),
        )
    else:
        lr_scheduler = None

    # --- AMP scaler ---------------------------------------------------------
    # use_amp is now actually wired up (was read but unused before)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # --- Training loop ------------------------------------------------------
    all_train_losses: List[float] = []
    all_val_losses:   List[float] = []
    best_val_loss = float("inf")
    min_delta     = 1e-4
    wait          = 0

    for epoch in range(num_epochs):

        # ---- train ---------------------------------------------------------
        model.train()
        train_losses: List[float] = []

        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch + 1} - Training"):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                predictions = model(xb)
                loss = loss_fn(predictions, yb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())

        # ---- validate ------------------------------------------------------
        model.eval()
        val_losses: List[float] = []

        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"Epoch {epoch + 1} - Validation"):
                xb, yb = xb.to(device), yb.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    predictions = model(xb)
                    loss = loss_fn(predictions, yb)
                val_losses.append(loss.item())

        mean_train_loss = float(np.mean(train_losses))
        mean_val_loss   = float(np.mean(val_losses))
        all_train_losses.append(mean_train_loss)
        all_val_losses.append(mean_val_loss)

        if lr_scheduler is not None:
            lr_scheduler.step(mean_val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        tqdm.write(
            f"Epoch {epoch + 1}: "
            f"Train Loss = {mean_train_loss:.4f}, "
            f"Val Loss = {mean_val_loss:.4f}, "
            f"LR = {current_lr:.2e}"
        )

        # ---- checkpoint ----------------------------------------------------
        if mean_val_loss < (best_val_loss - min_delta):
            torch.save(model.state_dict(), "model_best.pt")
            best_val_loss = mean_val_loss
            wait = 0
            tqdm.write(f"  ✓ New best model saved (val loss {best_val_loss:.4f})")
        else:
            wait += 1
            if wait >= early_stop_patience:
                tqdm.write("Early stopping triggered.")
                break

    # --- Integrated Gradients on best checkpoint ----------------------------
    model.load_state_dict(torch.load("model_best.pt", map_location=device))
    model.eval()
    compute_ig_importance(
        model=model,
        val_loader=val_loader,
        predictor_names=train_ds.channel_names,
        sample_batches=16,
        steps=64,
        save_dir="stats/train",
    )


if __name__ == "__main__":
    main()