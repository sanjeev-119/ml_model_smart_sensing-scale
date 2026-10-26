"""
Read the INAOE plantar thermogram database (IEEE DataPort, DOI 10.21227/tm4t-9n15).

Layout, from the dataset page: one folder per subject (CG001_F, DM014_M, ...). Inside:
  CG001_F_L.csv / CG001_F_R.csv            whole-foot temperature matrices
  CG001_F_L.png / CG001_F_R.png            false-colour pictures (not used)
  a subfolder holding the four angiosome files, e.g. CG001_F_L_MPA.csv
Angiosomes are named LCA, LPA, MCA and MPA.

ASSUMPTIONS TO VERIFY with inspect_dataset.py before trusting any number:
  1. temperature CSVs are in degrees C
  2. each angiosome CSV has the SAME shape as the whole-foot CSV, with 0 or NaN
     outside the region (if it is cropped instead, load_foot raises an error)
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

REGIONS = ["lca", "lpa", "mca", "mpa"]
NAME_RE = re.compile(r"^(CG|DM)(\d{3})_([MF])_([LR])(?:_(LCA|LPA|MCA|MPA))?$")


def read_matrix(path):
    """Read a temperature CSV into a float array (tries comma, then whitespace)."""
    try:
        arr = pd.read_csv(path, header=None).to_numpy(dtype=float)
        if arr.shape[1] > 1:
            return arr
    except Exception:
        pass
    return pd.read_csv(path, header=None, sep=r"\s+").to_numpy(dtype=float)


def index_files(root):
    """Map file stem -> path for every CSV under root (works for any folder layout)."""
    return {p.stem: p for p in Path(root).rglob("*.csv")}


def parse_name(stem):
    m = NAME_RE.match(stem)
    if not m:
        return None
    group, num, sex, side, region = m.groups()
    return {
        "subject_id": f"{group}{num}",
        "group": group,
        "sex": sex,
        "side": side,
        "region": region.lower() if region else None,
    }


def foot_stems(index):
    """Stems of whole-foot files (no angiosome suffix), sorted for reproducibility."""
    out = []
    for stem in index:
        meta = parse_name(stem)
        if meta and meta["region"] is None:
            out.append(stem)
    return sorted(out)


def region_mask(arr):
    """Pixels inside a region. Background is assumed to be 0 or NaN."""
    return np.isfinite(arr) & (arr > 0)


def load_foot(index, stem):
    """Return (meta, temperature array in C, {region: boolean mask})."""
    meta = parse_name(stem)
    temp = read_matrix(index[stem])
    masks = {}
    for r in REGIONS:
        key = f"{stem}_{r.upper()}"
        if key not in index:
            raise KeyError(f"missing angiosome file for {key}")
        a = read_matrix(index[key])
        if a.shape != temp.shape:
            raise ValueError(
                f"{key}: shape {a.shape} differs from the whole-foot file {temp.shape}. "
                "The angiosome files look cropped, so the masks cannot be placed directly. "
                "Run inspect_dataset.py and check how the regions line up."
            )
        masks[r] = region_mask(a)
    return meta, temp, masks
