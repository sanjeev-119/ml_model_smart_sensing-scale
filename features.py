
"""
Feature extraction for one foot/image.

Zones:
LU = Left Upper
LL = Left Lower
RU = Right Upper
RL = Right Lower

Features per zone:
- avg: Average temperature
- max: Maximum temperature
- min: Minimum temperature
"""

import numpy as np

REGIONS = ["LU", "LL", "RU", "RL"]
STATS = ["avg", "max", "min"]


def _stats(values):
    if values.size == 0:
        return {stat: np.nan for stat in STATS}

    return {
        "avg": float(values.mean()),
        "max": float(values.max()),
        "min": float(values.min()),
    }


def foot_features(temp, masks):
    """
    Parameters
    ----------
    temp : 2D NumPy array
        Temperature values in degrees Celsius.
    masks : dict
        Boolean masks for LU, LL, RU, and RL.

    Returns
    -------
    feats : dict
        12 temperature features (3 per zone).
    sizes : dict
        Number of pixels in each zone.
    """
    temp = np.asarray(temp, dtype=float)

    if temp.ndim != 2:
        raise ValueError("temp must be a 2D temperature array.")

    feats = {}
    sizes = {}

    for region in REGIONS:
        if region not in masks:
            raise ValueError(f"Missing mask for zone: {region}")

        mask = np.asarray(masks[region], dtype=bool)

        if mask.shape != temp.shape:
            raise ValueError(
                f"Mask {region} has shape {mask.shape}, "
                f"but temperature array has shape {temp.shape}."
            )

        values = temp[mask]
        sizes[region] = int(values.size)

        for stat, value in _stats(values).items():
            feats[f"{region}_{stat}"] = value

    return feats, sizes
