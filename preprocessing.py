"""
Preprocessing pipeline: combine yearly NetCDF files -> normalize -> pad to fullinput.nc

Expected input: one NetCDF file per year in DATA_DIR, named merged_scfv_{year}.nc,
containing the following variables harmonized to a common 0.05 degree grid:
    scfg       (time, lat, lon) float   - Snow Cover Fraction [0, 1], NaN for invalid/cloud
    swe        (time, lat, lon) float   - Snow Water Equivalent
    t2m        (time, lat, lon) float   - 2m air temperature
    band_data  (lat, lon)       float   - Elevation (SRTM)
    lccs_class (lat, lon)       int16   - ESA CCI land cover classification codes

Output:
    data/fullinput.nc      - Padded, normalized dataset ready for training and inference
    data/fullinput_norm.nc - Normalized but unpadded intermediate
    data/norm_params.json  - Per-variable normalization statistics (mean, std)
"""

import json
import sys
from typing import List, Dict

import numpy as np
import xarray as xr

# =========================
# ======= CONFIG ==========
# =========================

YEARS = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
         2010, 2011, 2012, 2013, 2014]

DATA_DIR = "data"

# Output files
OUT_COMBINED_NORM = "data/fullinput_norm.nc"
OUT_PADDED        = "data/fullinput.nc"
OUT_NORM_JSON     = "data/norm_params.json"

# Variables to z-score normalize
# Do NOT include scfg (already [0,1]), scf_flag (integer codes), or categoricals
NORM_VARS = ["band_data", "swe", "t2m"]

# Variables that are static (2D, no time dim)
STATIC_VARS = ["band_data", "lccs_class"]

# Integer-typed variables that must never be cast to float or NaN-padded
INT_VARS = {"scf_flag", "lccs_class"}

# Candidate SCF variable names to auto-detect
SCF_NAME_CANDIDATES = ["scfg", "scfv", "scf"]

# Pad to multiples of (for U-Net strides)
PAD_MULTIPLE = 16

# Optional dask chunks (None = disabled)
CHUNKS = None
# =========================


def find_scf_var(ds: xr.Dataset) -> str:
    for name in SCF_NAME_CANDIDATES:
        if name in ds:
            return name
    raise ValueError(f"Could not find SCF var in dataset. Tried {SCF_NAME_CANDIDATES}")


def sanitize_statics(ds: xr.Dataset, static_vars: List[str]) -> xr.Dataset:
    """
    Ensure static variables are 2D (lat, lon).
    Integer variables (lccs_class) are kept as-is; pad_to_multiple handles them.
    Float variables are cast to float32.
    """
    ds = ds.copy()
    for v in static_vars:
        if v not in ds:
            continue
        # Drop accidental time dim
        if "time" in ds[v].dims:
            ds[v] = ds[v].isel(time=0, drop=True)
        # Only cast to float if it is not an integer-flag variable
        if v not in INT_VARS and np.issubdtype(ds[v].dtype, np.integer):
            ds[v] = ds[v].astype("float32")
    return ds


def prep_scf(ds: xr.Dataset, scf_name: str) -> xr.Dataset:
    """
    Validate the SCF variable.
    CombineData already outputs scfg in [0,1] with NaNs for all non-valid pixels,
    so we only need to confirm the range and leave it unchanged.
    """
    ds = ds.copy()
    scf = ds[scf_name].astype("float32")

    mx = float(scf.max(skipna=True))
    if mx > 1.0 + 1e-3:
        # Shouldn't happen if CombineData ran correctly, but handle gracefully
        print(
            f"⚠️  {scf_name} max = {mx:.2f} > 1. Rescaling by /100. "
            "Check that CombineData normalised this variable.",
            file=sys.stderr,
        )
        scf = scf / 100.0

    # Mask anything outside [0, 1] (catches stray special codes if any slipped through)
    scf = scf.where((scf >= 0.0) & (scf <= 1.0))
    ds[scf_name] = scf
    return ds


def compute_norm_stats(ds: xr.Dataset, variables: List[str]) -> Dict[str, Dict[str, float]]:
    """Compute global mean/std over time+space for each variable (ignoring NaNs)."""
    params = {}
    for var in variables:
        if var not in ds:
            print(f"⚠️  Variable '{var}' not in dataset; skipping normalization.", file=sys.stderr)
            continue
        arr = ds[var].astype("float64").values
        mask = np.isfinite(arr)
        if not np.any(mask):
            print(f"⚠️  Variable '{var}' has no finite values; skipping.", file=sys.stderr)
            continue
        mean = float(arr[mask].mean())
        std  = float(arr[mask].std(ddof=0))
        if std == 0.0:
            print(f"⚠️  Variable '{var}' std == 0; forcing std=1 to avoid NaNs.", file=sys.stderr)
            std = 1.0
        params[var] = {"mean": mean, "std": std}
    return params


def apply_norm(ds: xr.Dataset, params: Dict[str, Dict[str, float]]) -> xr.Dataset:
    ds = ds.copy()
    for var, p in params.items():
        if var in ds:
            ds[var] = (ds[var] - p["mean"]) / p["std"]
            print(f"→ Normalized {var}: mean={p['mean']:.6f}, std={p['std']:.6f}")
    return ds


def pad_to_multiple(ds: xr.Dataset, multiple: int = 16) -> xr.Dataset:
    """
    Pad lat/lon to the nearest multiple of `multiple`.

    Float variables  → padded with NaN  (ocean/missing convention)
    Integer variables (scf_flag, lccs_class) → padded with -1  (no-data sentinel)
      This prevents prep.py's fillna(0) from turning padding rows into fake
      valid observations or fake 0% SCF values.
    """
    lat_n = ds.sizes["lat"]
    lon_n = ds.sizes["lon"]

    pad_lat = (multiple - lat_n % multiple) % multiple
    pad_lon = (multiple - lon_n % multiple) % multiple

    lat_before = pad_lat // 2
    lat_after  = pad_lat - lat_before
    lon_before = pad_lon // 2
    lon_after  = pad_lon - lon_before

    if pad_lat == 0 and pad_lon == 0:
        print("ℹ️  No padding needed; already aligned to the multiple.")
        return ds

    ds_pad = ds.copy()

    # Cast float-paddable variables to float32 (skip integer flag fields)
    for name, da in ds_pad.data_vars.items():
        if name in INT_VARS:
            continue
        if ("lat" in da.dims) and ("lon" in da.dims):
            if not np.issubdtype(da.dtype, np.floating):
                ds_pad[name] = da.astype("float32")

    # Pad everything with NaN first (xarray requirement for mixed dtypes)
    # Integer variables get temporarily NaN-padded as float, then restored below
    int_originals = {}
    for name in INT_VARS:
        if name in ds_pad:
            int_originals[name] = ds_pad[name].copy()
            ds_pad[name] = ds_pad[name].astype("float32")

    ds_pad = ds_pad.pad(
        lat=(lat_before, lat_after),
        lon=(lon_before, lon_after),
        constant_values=np.nan,
    )

    # Restore integer variables: fill NaN padding with -1, cast back to int16
    for name, orig in int_originals.items():
        ds_pad[name] = ds_pad[name].fillna(-1).astype(orig.dtype)
        # Carry over attributes
        ds_pad[name].attrs.update(orig.attrs)

    print(
        f"ℹ️  Padded: lat {lat_n}→{ds_pad.sizes['lat']} "
        f"(+{lat_before}|+{lat_after}), "
        f"lon {lon_n}→{ds_pad.sizes['lon']} "
        f"(+{lon_before}|+{lon_after})"
    )
    return ds_pad


def main():
    open_kwargs = {}
    if CHUNKS is not None:
        open_kwargs["chunks"] = CHUNKS

    datasets = []
    for year in YEARS:
        path = f"{DATA_DIR}/merged_scfv_{year}.nc"
        print(f"Loading {path} ...")
        ds = xr.open_dataset(path, **open_kwargs)
        ds = sanitize_statics(ds, STATIC_VARS)
        scf_name = find_scf_var(ds)
        ds = prep_scf(ds, scf_name)
        datasets.append(ds)

    # Verify all years share the same lat/lon grid
    lat0, lon0 = datasets[0]["lat"].values, datasets[0]["lon"].values
    for i, ds in enumerate(datasets[1:], start=1):
        if not (np.allclose(lat0, ds["lat"].values) and np.allclose(lon0, ds["lon"].values)):
            raise ValueError(f"Lat/Lon mismatch in year {YEARS[i]}")

    # Split dynamic vs static variables
    def split_ds(ds: xr.Dataset, static_vars: List[str]):
        dyn = ds.drop_vars([v for v in static_vars if v in ds], errors="ignore")
        sta = ds[[v for v in static_vars if v in ds]]
        return dyn, sta

    dyn_list, sta_list = [], []
    for ds in datasets:
        dyn, sta = split_ds(ds, STATIC_VARS)
        dyn_list.append(dyn)
        sta_list.append(sta)

    ds_dyn = xr.concat(dyn_list, dim="time").sortby("time")
    ds_all = xr.merge([ds_dyn, sta_list[0]], compat="override")

    # Ensure statics are still 2D after concat
    ds_all = sanitize_statics(ds_all, STATIC_VARS)

    # Mask SWE negatives (nodata) before normalization
    if "swe" in ds_all:
        ds_all["swe"] = ds_all["swe"].where(ds_all["swe"] >= 0)

    # Compute and apply normalization (scfg and scf_flag are intentionally excluded)
    norm_params = compute_norm_stats(ds_all, NORM_VARS)
    ds_all_norm = apply_norm(ds_all, norm_params)

    # Save combined normalised dataset
    ds_all_norm.to_netcdf(OUT_COMBINED_NORM)
    print(f"✅ Wrote combined+normalized dataset: {OUT_COMBINED_NORM}")

    # Save normalization params
    with open(OUT_NORM_JSON, "w") as fp:
        json.dump(norm_params, fp, indent=2)
    print(f"✅ Wrote normalization params: {OUT_NORM_JSON}")

    # Pad to PAD_MULTIPLE
    ds_padded = pad_to_multiple(ds_all_norm, multiple=PAD_MULTIPLE)
    ds_padded.to_netcdf(OUT_PADDED)
    print(f"✅ Wrote padded dataset: {OUT_PADDED}")

    print("\nℹ️  Notes:")
    print(f"- scfg is in [0,1] with NaN for invalid/cloud/water pixels.")
    print(f"- scf_flag carries original integer codes; padding pixels = -1.")
    print(f"- Static variables are 2D (lat, lon).")
    print(f"- Normalisation params saved to {OUT_NORM_JSON} for inference-time use.")


if __name__ == "__main__":
    main()
