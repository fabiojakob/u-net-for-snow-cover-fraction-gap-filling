from torch.utils.data import Dataset
import xarray as xr
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import torch
from typing import List, Optional, Sequence, Tuple

def add_scf_quality(ds, k, window, threshold):
    ds = ds.copy()

    scf = ds["scfg"]
    swe = ds["swe"]

    scf_sentinel = scf.fillna(-1.0)
    known = scf_sentinel != -1.0

    # count observed pixels before filtering
    n_observed = int(known.values.sum())

    # add plausibility
    scf_clip = scf_sentinel.clip(0.0, 1.0)
    swe_sig = 1.0 / (1.0 + np.exp(-k * swe))
    plausibility = 1.0 - np.abs(scf_clip - swe_sig)
    plausibility = xr.where(known, plausibility, np.nan)

    ds["plausibility"] = plausibility

    # add trust
    scf_known = scf_sentinel.where(known)
    trust_slices = []
    for t in range(scf_known.sizes["time"]):
        scf_t = scf_known.isel(time=t)

        local_mean = scf_t.rolling(lat=window, lon=window, center=True, min_periods=1).mean()
        trust_t = (1.0 - np.abs(scf_t - local_mean)).clip(0.0, 1.0)

        trust_t = trust_t.where(np.isfinite(scf_t) & np.isfinite(local_mean))
        trust_slices.append(trust_t)

    trust = xr.concat(trust_slices, dim="time").assign_coords(time=scf_known["time"])

    ds["trust"] = trust

    # add quality
    has_plausibility = np.isfinite(ds["plausibility"])
    quality = xr.where(has_plausibility, ds["plausibility"], ds["trust"])
    quality = quality.where(np.isfinite(ds["plausibility"]) | np.isfinite(ds["trust"]))

    ds["quality"] = quality

    # count pixels that will be removed (observed but low quality)
    low_q = ds["quality"] < float(threshold)
    n_removed = int((known & low_q).values.sum())

    # remove low quality scf
    scf_clean = ds["scfg"]
    scf_clean = scf_clean.where(~low_q | scf_clean.isnull())
    ds["scfg"] = scf_clean

    # --- print quality filter statistics ---
    pct = 100.0 * n_removed / n_observed if n_observed > 0 else 0.0
    print(
        f"[SCF quality filter]  threshold={threshold}"
        f"  observed={n_observed:,}"
        f"  removed={n_removed:,}"
        f"  ({pct:.1f}%)"
        f"  retained={n_observed - n_removed:,}"
        f"  ({100.0 - pct:.1f}%)"
    )

    return ds

def add_cloud_mask(ds, min_blobs, max_blobs, min_radius, max_radius, persistence_days, seed):
    ds = ds.copy()
    nt = ds.sizes["time"]
    ny = ds.sizes["lat"]
    nx = ds.sizes["lon"]

    mask = np.zeros((nt, ny, nx), dtype=bool)
    yy, xx = np.ogrid[:ny, :nx]
    rng = np.random.default_rng(seed)

    base = None

    for t in range(nt):
        make_new = (base is None) or (persistence_days <= 1) or (t % persistence_days == 0)
        if make_new:
            n_blobs = rng.integers(min_blobs, max_blobs + 1)
            base = np.zeros((ny, nx), dtype=bool)

            for _ in range(n_blobs):
                cy = rng.integers(0, ny)
                cx = rng.integers(0, nx)
                r = rng.integers(min_radius, max_radius + 1)
                circle = (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2
                base[circle] = True

        day = base.copy()

        mask[t] = day

    ds["cloud_mask"] = xr.DataArray(
        mask,
        dims=("time", "lat", "lon"),
        coords={"time": ds.time, "lat": ds.lat, "lon": ds.lon},
    )

    return ds

def _remap_land_cover(lc):
    out = np.zeros_like(lc)

    #lc classes: https://maps.elie.ucl.ac.be/CCI/viewer/download/ESACCI-LC-Ph2-PUGv2_2.0.pdf
    forest   = [50, 60, 70, 80, 90, 100]
    shrub    = [110, 120, 122, 130, 140]
    cropland = [10, 11, 12, 20, 30, 40]
    sparse   = [150, 152, 200, 201]

    out[np.isin(lc, forest)] = 1
    out[np.isin(lc, shrub)] = 2
    out[np.isin(lc, cropland)] = 3
    out[np.isin(lc, sparse)] = 4
    out[lc == 190] = 5  # urban areas
    out[lc == 220] = 6  # permanent snow and ice
    out[lc == 210] = 7  # water bodies
    # flooded vegetation (160, 170, 180) intentionally left as 0 — eighth category

    return out

def lag_along_time(channel, k, repeat_edge):
    if k <= 0:
        return channel
    if repeat_edge:
        head = np.repeat(channel[:1], k, axis=0)
    else:
        head = np.zeros_like(channel[:k])
    return np.concatenate([head, channel[:-k]], axis=0)

def _channel_indices(predictors: Sequence[str], n_lc: int, lag_days: int):

    i = 0

    # --- base predictors
    if "swe" in predictors:
        i += 1
    if "t2m" in predictors:
        i += 1
    if "elev" in predictors:
        i += 1
    if "landcover" in predictors:
        i += n_lc

    # current SCF
    scf_idx = i if "scf" in predictors else None
    if "scf" in predictors:
        i += 1

    # known-mask
    km_idx = i if "known_mask" in predictors else None
    if "known_mask" in predictors:
        i += 1

    # melt proxy (derived, not lagged)
    if "swe_t2m_melt" in predictors:
        # swe_t2m_melt is always appended when both swe and t2m are present;
        # we don't need its index explicitly, so just advance past it.
        pass

    # NOTE: lag channels are no longer stored in self.inputs — they are computed
    # on-the-fly in __getitem__. scf_lag_idx is therefore always empty here, but
    # we keep the return signature identical for backward compatibility.
    scf_lag_idx: List[int] = []

    return scf_idx, km_idx, scf_lag_idx


class InputDataset(Dataset):
    def __init__(
        self,
        nc_path: str,
        split: str,
        test_frac: float,
        split_method: str,
        seed: int,
        predictors: tuple[str,...],
        lag_days: int,
        repeat_edge: bool,
        scf_quality_threshold: float,
        cloud_min_blobs: int,
        cloud_max_blobs: int,
        cloud_min_radius: int,
        cloud_max_radius: int,
        persistence_days: int,
        scf_dropout_p: float,
    ):
        self.predictors = tuple(predictors)
        self.lag_days = int(lag_days)
        self.repeat_edge = bool(repeat_edge)
        self.scf_dropout_p = float(scf_dropout_p)

        #Load and preprocess
        ds = xr.open_dataset(nc_path)

        ds = add_scf_quality(ds, k=4.0, window=10, threshold=scf_quality_threshold)

        ds = add_cloud_mask(ds,
                            cloud_min_blobs,
                            cloud_max_blobs,
                            cloud_min_radius,
                            cloud_max_radius,
                            persistence_days,
                            seed)

        # scf sentinel
        ds["scfg"] = ds["scfg"].fillna(-1.0)

        # create ocean mask
        ocean2d = np.isnan(ds["band_data"]).values
        T, H, W = ds["swe"].shape
        self.ocean = np.broadcast_to(ocean2d, (T, H, W))

        # fill remaining nans
        ds = ds.fillna(0.0)

        elev2d = ds["band_data"].values
        lc2d = ds["lccs_class"].values
        swe = ds["swe"].values
        t2m = ds["t2m"].values
        scf = ds["scfg"].values
        cloud = ds["cloud_mask"].values.astype(bool)
        time = ds["time"].values if "time" in ds.coords else None

        self.scf_known = (scf != -1.0) & (~self.ocean)

        # define supervised pixels
        self.supervised = cloud & self.scf_known

        # define target
        y = np.full((T, H, W, 1), -1.0, dtype=np.float32)
        y[self.supervised, 0] = scf[self.supervised].astype(np.float32)

        # hide supervised pixels with sentinel value for scf input channel
        scf_in = np.where(self.supervised, -1.0, scf).astype(np.float32)
        scf_ch = scf_in[..., None]

        # broadcast static fields to time
        elev3d = np.broadcast_to(elev2d, (T, H, W))
        lc3d = np.broadcast_to(lc2d, (T, H, W))

        # encode land cover
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        lc_coarse = _remap_land_cover(lc3d)
        lc_flat = lc_coarse.reshape(-1, 1)
        lc_1hot = enc.fit_transform(lc_flat).reshape(T, H, W, -1).astype(np.float32)

        n_lc = lc_1hot.shape[-1]
        self.cats = [enc.categories_[0]]

        # stack BASE predictors only (no lag copies)
        feats: list[np.ndarray] = []
        channel_names_base = []

        if "swe" in self.predictors:
            swe_ch = swe[..., None].astype(np.float32)
            feats.append(swe_ch)
            channel_names_base.append("swe")

        if "t2m" in self.predictors:
            t2m_ch = t2m[..., None].astype(np.float32)
            feats.append(t2m_ch)
            channel_names_base.append("t2m")

        if "elev" in self.predictors:
            feats.append(elev3d[..., None].astype(np.float32))
            channel_names_base.append("elev")

        if "landcover" in self.predictors:
            feats.append(lc_1hot)
            for c in self.cats[0]:
                channel_names_base.append(f"landcover_{int(c)}")

        if "scf" in self.predictors:
            feats.append(scf_ch.astype(np.float32))
            channel_names_base.append("scf")

        if "swe_t2m_melt" in self.predictors and "swe" in self.predictors and "t2m" in self.predictors:
            melt_proxy = (np.tanh(0.5 * t2m) + 1.0) / 2.0
            melt_ch = (swe * melt_proxy)[..., None].astype(np.float32)
            feats.append(melt_ch)
            channel_names_base.append("swe_t2m_melt")

        if not feats:
            raise ValueError("No predictors selected.")

        x = np.concatenate(feats, axis=-1)  # (T, H, W, C_base) — no lag copies

        # Build lag specs: list of (base_channel_idx, lag_k, channel_name)
        # These are used in __getitem__ to assemble lag channels on-the-fly.
        self._lag_specs: List[Tuple[int, int, str]] = []
        channel_names_lag: List[str] = []

        if self.lag_days > 0:
            for lag_k in range(1, self.lag_days + 1):
                if "swe" in self.predictors:
                    base_idx = channel_names_base.index("swe")
                    self._lag_specs.append((base_idx, lag_k, f"swe_lag{lag_k}"))
                    channel_names_lag.append(f"swe_lag{lag_k}")
                if "t2m" in self.predictors:
                    base_idx = channel_names_base.index("t2m")
                    self._lag_specs.append((base_idx, lag_k, f"t2m_lag{lag_k}"))
                    channel_names_lag.append(f"t2m_lag{lag_k}")
                if "scf" in self.predictors:
                    base_idx = channel_names_base.index("scf")
                    self._lag_specs.append((base_idx, lag_k, f"scf_lag{lag_k}"))
                    channel_names_lag.append(f"scf_lag{lag_k}")

        # Full channel name list (base + lag) — matches the tensor produced by __getitem__
        self.channel_names = channel_names_base + channel_names_lag

        # scf/known-mask dropout indices are into the BASE channels only
        self.scf_idx, self.km_idx, self.lag_idx = _channel_indices(
            self.predictors, n_lc, lag_days=0  # lag_days=0: lag channels not in inputs array
        )

        # scf_lag channel indices in the FULL output tensor (base + lag), used by
        # external callers (e.g. baselines.py) that reference channel_names
        _base_width = x.shape[-1]
        self.lag_idx = [
            _base_width + i
            for i, (_, _, name) in enumerate(self._lag_specs)
            if name.startswith("scf_lag")
        ]

        # train / validation split (on base array only — no lag copies in memory)
        if split_method == "random":
            all_idx = np.arange(T)
            tr_idx, va_idx = train_test_split(
                all_idx,
                test_size=float(test_frac),
                shuffle=True,
                random_state=seed,
            )
            self.inputs  = x[tr_idx]  if split == "train" else x[va_idx]
            self.targets = y[tr_idx]  if split == "train" else y[va_idx]
            self._orig_idx = tr_idx if split == "train" else va_idx
            self._full_x   = x

        elif split_method == "inference":
            self.inputs  = x
            self.targets = y
            self._orig_idx = np.arange(T)
            self._full_x   = x

        else:
            # chronological split
            m_tr = time < np.datetime64("2013-01-01")
            m_va = (time >= np.datetime64("2013-01-01")) & (time < np.datetime64("2013-12-31"))

            self.inputs  = x[m_tr] if split == "train" else x[m_va]
            self.targets = y[m_tr] if split == "train" else y[m_va]
            self._orig_idx = np.where(m_tr)[0] if split == "train" else np.where(m_va)[0]
            self._full_x   = x

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_base = torch.from_numpy(self.inputs[idx]).permute(2, 0, 1).float()
        y = torch.from_numpy(self.targets[idx]).permute(2, 0, 1).float()
    
        # Dropout: drop SCF (and known_mask if present) together
        drop_scf = (
            self.scf_idx is not None
            and np.random.rand() < self.scf_dropout_p
        )
        if drop_scf:
            x_base[self.scf_idx] = -1.0
            if self.km_idx is not None:
                x_base[self.km_idx] = 0.0
    
        if self._lag_specs:
            lag_slices: List[torch.Tensor] = []
            orig_t = int(self._orig_idx[idx])
    
            for base_ch_idx, lag_k, name in self._lag_specs:
                # Drop SCF lag channels when base SCF is dropped
                if name.startswith("scf_lag") and drop_scf:
                    H, W = x_base.shape[1], x_base.shape[2]
                    lag_slices.append(torch.full((1, H, W), -1.0, dtype=torch.float32))
                    continue
    
                src_t = orig_t - lag_k
                if src_t < 0:
                    if self.repeat_edge:
                        src_t = 0
                    else:
                        H, W = x_base.shape[1], x_base.shape[2]
                        fill = -1.0 if name.startswith("scf") else 0.0
                        lag_slices.append(torch.full((1, H, W), fill, dtype=torch.float32))
                        continue
    
                lag_frame = torch.from_numpy(
                    self._full_x[src_t, :, :, base_ch_idx].copy()
                ).float().unsqueeze(0)
                lag_slices.append(lag_frame)
    
            x = torch.cat([x_base] + lag_slices, dim=0)
        else:
            x = x_base
    
        return x, y