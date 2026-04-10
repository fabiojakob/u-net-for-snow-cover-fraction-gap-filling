# U-Net for Snow Cover Fraction Gap-Filling

Code for the paper:

**"Applying U-Net for Estimating AVHRR-Based Snow Cover Fraction (ESA CCI Snow) During Cloud Cover and Polar Night in Scandinavia"**
Fabio Jakob, Christoph Neuhaus, Stefan Wunderle — University of Bern, 2026

---

## Overview

This repository provides a U-Net-based deep learning framework for reconstructing missing Snow Cover Fraction (SCF) values in the ESA CCI L3C AVHRR SCFV product, caused by cloud contamination and polar night conditions. The model is trained on Scandinavia (2000–2014) using physically meaningful auxiliary predictors: Snow Water Equivalent (SWE), near-surface air temperature, elevation, and land cover.

---

## Repository Structure

```
├── preprocessing.py     # Combine yearly NetCDFs, normalize, pad → fullinput.nc
├── prep.py              # Dataset class and SCF quality filtering (training + inference)
├── model.py             # U-Net architecture and composite loss function
├── training.py          # Model training with early stopping and Integrated Gradients
├── inf.py               # Inference with MC Dropout uncertainty estimation
├── config.json          # All hyperparameters and file paths
└── README.md
```

---

## Data

The following datasets are required and must be downloaded independently:

| Variable | Dataset | Source |
|----------|---------|--------|
| Snow Cover Fraction (SCF) | ESA CCI L3C SCFV AVHRR v4.0 | https://climate.esa.int/en/projects/snow/ |
| Snow Water Equivalent (SWE) | ESA CCI L3C SWE SSMIS DMSP v3.1 | https://climate.esa.int/en/projects/snow/ |
| Land Cover | ESA CCI LC L4 300m P1Y | https://climate.esa.int/en/projects/land-cover/ |
| 2m Temperature | ERA5-Land | https://cds.climate.copernicus.eu |
| Elevation | CGIAR-CSI SRTM 4.1 | http://srtm.csi.cgiar.org |

Ground station validation data:
- NVE (Norway): https://seklima.met.no/observations/
- FMI (Finland): https://en.ilmatieteenlaitos.fi/download-observations
- SMHI (Sweden): https://www.smhi.se/data/nederbord-och-fuktighet/sno/snowDepth

---

## Setup

```bash
pip install torch torchvision xarray numpy scikit-learn tqdm
```

A `requirements.txt` with exact versions will be added upon publication.

---

## Usage

### 1. Preprocess data
Prepare one NetCDF file per year containing all input variables harmonized to a common 0.05° grid, named `data/merged_scfv_{year}.nc`. Each file must contain:

| Variable | Dims | Description |
|----------|------|-------------|
| `scfg` | (time, lat, lon) | Snow Cover Fraction [0,1], NaN for cloud/invalid |
| `swe` | (time, lat, lon) | Snow Water Equivalent |
| `t2m` | (time, lat, lon) | 2m air temperature |
| `band_data` | (lat, lon) | Elevation (SRTM) |
| `lccs_class` | (lat, lon) | ESA CCI land cover codes (int16) |

Then run:
```bash
python preprocessing.py
```
This produces `data/fullinput.nc` (normalized and padded) and `data/norm_params.json` (normalization statistics).

### 2. Train
```bash
python training.py
```
Trains the U-Net on the year configured in `config.json` (default: 2012), validates on 2013, and saves the best checkpoint as `model_best.pt`. Integrated Gradients feature importance is computed after training.

### 3. Inference
Full MC Dropout inference over the complete 2000–2014 period:
```bash
python inf.py
```
Quick single-pass mode (faster, for testing):
```bash
python inf.py --quick
```
Output is saved as `scf_inference.nc` containing observed SCF, model predictions, uncertainty estimates, and the gap-filled product.

---

## Configuration

All parameters are controlled via `config.json`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `scf_quality_threshold` | 0.7 | SCF quality filter threshold (0.0 = no filtering) |
| `scf_dropout_p` | 0.1 | Probability of dropping entire SCF input channel during training |
| `lag_days` | 1 | Number of lag days for temporal context |
| `mc_passes` | 30 | Number of MC Dropout forward passes at inference |
| `persistence_days` | 10 | Days over which synthetic cloud masks persist |

Update `paths` in `config.json` to point to your local data files.

---

## Output

The gap-filled NetCDF (`scf_inference.nc`) contains four variables:

- `scf_obs` — original ESA CCI AVHRR observations
- `scf_mean` — U-Net posterior mean prediction (all pixels)
- `scf_uncertainty` — MC Dropout posterior standard deviation
- `scf_filled` — gap-filled product (observations where available, model predictions elsewhere)

---

## Citation

To be added upon publication.

---

## License

To be added upon publication.
