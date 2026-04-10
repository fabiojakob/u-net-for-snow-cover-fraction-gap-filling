import argparse
import json
import os
import numpy as np
import torch
import xarray as xr
from tqdm import tqdm
from prep import InputDataset
from model import UNet


def enable_mc_dropout(model: torch.nn.Module) -> None:
    """Keep Dropout2d layers active during inference for MC Dropout."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout2d):
            m.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Single-pass inference (no MC dropout). Faster, for config comparison only.",
    )
    args = parser.parse_args()

    with open("config.json", "r") as f:
        cfg = json.load(f)

    infer_nc = cfg["paths"].get("infer_data", None)
    out_dir  = cfg["paths"].get("infer_out_dir")
    out_nc   = cfg["paths"].get("infer_output_nc")

    if args.quick:
        print("Running in QUICK mode (single forward pass, no MC dropout, no NetCDF output).")

    # ---- Inference dataset (full temporal range, different cloud seed) ----
    ds = InputDataset(
        infer_nc,
        split="train",
        test_frac=cfg["dataset"]["test_frac"],
        split_method="inference",
        seed=cfg["dataset"]["seed"] + 1000,  # different cloud mask from training
        predictors=cfg["dataset"]["predictors"],
        lag_days=cfg["dataset"]["lag_days"],
        repeat_edge=cfg["dataset"]["repeat_edge"],
        scf_quality_threshold=cfg["dataset"]["scf_quality_threshold"],
        cloud_min_blobs=cfg["dataset"]["cloud_min_blobs"],
        cloud_max_blobs=cfg["dataset"]["cloud_max_blobs"],
        cloud_min_radius=cfg["dataset"]["cloud_min_radius"],
        cloud_max_radius=cfg["dataset"]["cloud_max_radius"],
        persistence_days=cfg["dataset"]["persistence_days"],
        scf_dropout_p=0.0,
    )

    x = ds.inputs
    T, H, W, C_base = x.shape
    C_total = len(ds.channel_names)

    # ---- Load model ----
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = UNet(in_channels=C_total).to(device)
    ckpt_path = cfg["paths"].get("best_model_path")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    batch_size = int(cfg["training"]["batch_size"])

    # ---- Inference ----
    scf_pred = np.zeros((T, H, W), dtype=np.float32)

    if args.quick:
        # Single forward pass — no MC dropout, no uncertainty
        with torch.no_grad():
            for t0 in tqdm(range(0, T, batch_size), desc="Quick inference"):
                t1  = min(T, t0 + batch_size)
                xb  = torch.stack([ds[t][0] for t in range(t0, t1)]).to(device)
                logits = model(xb)
                if logits.dim() == 3:
                    logits = logits.unsqueeze(1)
                prob = torch.sigmoid(logits).squeeze(1).cpu().numpy().astype(np.float32)
                scf_pred[t0:t1] = prob

        scf_uncertainty = None

    else:
        # MC Dropout inference (Welford online mean/variance)
        enable_mc_dropout(model)
        mc_passes = int(cfg["inference"].get("mc_passes", 30))
        mean_acc = np.zeros((T, H, W), dtype=np.float32)
        M2_acc   = np.zeros((T, H, W), dtype=np.float32)

        for n in range(mc_passes):
            with torch.no_grad():
                for t0 in tqdm(range(0, T, batch_size), desc=f"MC pass {n+1}/{mc_passes}"):
                    t1  = min(T, t0 + batch_size)
                    xb  = torch.stack([ds[t][0] for t in range(t0, t1)]).to(device)
                    logits = model(xb)
                    if logits.dim() == 3:
                        logits = logits.unsqueeze(1)
                    prob = torch.sigmoid(logits).squeeze(1).cpu().numpy().astype(np.float32)
                    delta           = prob - mean_acc[t0:t1]
                    mean_acc[t0:t1] += delta / (n + 1)
                    delta2          = prob - mean_acc[t0:t1]
                    M2_acc[t0:t1]  += delta * delta2

        scf_pred        = np.round(mean_acc, 2)
        scf_uncertainty = np.sqrt(M2_acc / max(mc_passes - 1, 1)).astype(np.float32)

    # ---- Observations and evaluation mask ----
    ds_nc       = xr.open_dataset(infer_nc)
    scf_obs_raw = ds_nc["scfg"].values.astype(np.float32)
    scf_obs     = scf_obs_raw.copy()

    eval_mask     = ds.supervised
    scf_eval_true = ds.targets[..., 0].astype(np.float32)

    obs_vec  = scf_eval_true[eval_mask]
    pred_vec = scf_pred[eval_mask].copy()

    # ---- Overall metrics ----
    ss_res = np.sum((obs_vec - pred_vec) ** 2)
    ss_tot = np.sum((obs_vec - np.mean(obs_vec)) ** 2)
    overall_rmse = float(np.sqrt(np.mean((pred_vec - obs_vec) ** 2)))
    overall_mae  = float(np.mean(np.abs(pred_vec - obs_vec)))
    overall_bias = float(np.mean(pred_vec - obs_vec))
    overall_r2   = float(1.0 - ss_res / (ss_tot + 1e-10))

    mode_str = "quick (single-pass)" if args.quick else f"MC dropout ({mc_passes} passes)"
    print(f"\nInference complete  [{mode_str}]")
    print(f"  Eval pixels:          {obs_vec.shape[0]:,}")
    print(f"  Overall RMSE:         {overall_rmse:.4f}")
    print(f"  Overall MAE:          {overall_mae:.4f}")
    print(f"  Overall Bias:         {overall_bias:+.4f}")
    print(f"  Overall R²:           {overall_r2:.4f}")

    # ---- Save flat evaluation arrays ----
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "obs.npy"),             obs_vec)
    np.save(os.path.join(out_dir, "pred.npy"),            pred_vec)
    np.save(os.path.join(out_dir, "evaluation_mask.npy"), eval_mask.astype(bool))

    # ---- Per-time RMSE / MAE ----
    rmse_t = np.full((T,), np.nan, dtype=np.float32)
    mae_t  = np.full((T,), np.nan, dtype=np.float32)

    for t in range(T):
        m = eval_mask[t]
        if not np.any(m):
            continue
        o  = scf_eval_true[t][m]
        p  = scf_pred[t][m]
        ok = np.isfinite(o)
        if not np.any(ok):
            continue
        o, p = o[ok], p[ok]
        d = (p - o).astype(np.float32)
        rmse_t[t] = float(np.sqrt(np.mean(d * d)))
        mae_t[t]  = float(np.mean(np.abs(d)))

    np.save(os.path.join(out_dir, "rmse_per_time.npy"), rmse_t)
    np.save(os.path.join(out_dir, "mae_per_time.npy"),  mae_t)

    print(f"\nSaved to: {out_dir}")
    print(f"  obs.npy, pred.npy, evaluation_mask.npy")
    print(f"  rmse_per_time.npy, mae_per_time.npy")

    # ---- Write output NetCDF (full MC mode only) ----
    if not args.quick:
        scf_filled      = scf_obs.copy()
        missing         = ~np.isfinite(scf_filled)
        scf_filled[missing] = scf_pred[missing]

        scf_obs_o    = scf_obs.copy()
        scf_mean_o   = scf_pred.copy()
        scf_unc_o    = scf_uncertainty.copy()
        scf_filled_o = scf_filled.copy()

        scf_obs_o[ds.ocean]    = np.nan
        scf_mean_o[ds.ocean]   = np.nan
        scf_unc_o[ds.ocean]    = np.nan
        scf_filled_o[ds.ocean] = np.nan

        ds_out = xr.Dataset(
            data_vars=dict(
                scf_obs=(("time", "lat", "lon"),         scf_obs_o.astype(np.float32)),
                scf_mean=(("time", "lat", "lon"),        scf_mean_o.astype(np.float32)),
                scf_uncertainty=(("time", "lat", "lon"), scf_unc_o.astype(np.float32)),
                scf_filled=(("time", "lat", "lon"),      scf_filled_o.astype(np.float32)),
            ),
            coords=dict(
                time=ds_nc["time"].values,
                lat=ds_nc["lat"].values,
                lon=ds_nc["lon"].values,
            ),
            attrs=dict(mc_passes=mc_passes),
        )
        ds_out.to_netcdf(out_nc)
        print(f"\nWrote NetCDF: {out_nc}")


if __name__ == "__main__":
    main()