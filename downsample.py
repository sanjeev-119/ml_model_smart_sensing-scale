"""
Area-average downsampling of a temperature array and its region masks.

Temperatures are averaged ONLY over foot pixels, so the cooler background never
leaks into the edge pixels. Masks are reduced by majority vote (a small pixel
belongs to a region if more than half of its source pixels did).

Two ways to choose the target size:
  reduce_resolution(temp, masks, (h, w))      whole image to h x w   (what the spec says)
  reduce_to_foot_rows(temp, masks, rows)      crop to the foot, then scale so the
                                              foot's long side is `rows` pixels
The second one mimics the real device, where the camera is placed so one foot fills
about 24 rows of the MLX90640. A whole-image 32x24 leaves the foot with far fewer pixels.
"""

import cv2
import numpy as np


def union(masks):
    out = None
    for m in masks.values():
        out = m.copy() if out is None else (out | m)
    return out


def reduce_resolution(temp, masks, out_hw):
    h, w = out_hw
    plantar = union(masks)
    f32 = np.float32
    num = cv2.resize(np.where(plantar, temp, 0).astype(f32), (w, h), interpolation=cv2.INTER_AREA)
    den = cv2.resize(plantar.astype(f32), (w, h), interpolation=cv2.INTER_AREA)
    small = np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan)
    masks_small = {
        k: cv2.resize(m.astype(f32), (w, h), interpolation=cv2.INTER_AREA) > 0.5
        for k, m in masks.items()
    }
    return small, masks_small


def foot_box(masks):
    plantar = union(masks)
    rows = np.flatnonzero(plantar.any(axis=1))
    cols = np.flatnonzero(plantar.any(axis=0))
    return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1


def reduce_to_foot_rows(temp, masks, foot_rows):
    r0, r1, c0, c1 = foot_box(masks)
    t = temp[r0:r1, c0:c1]
    m = {k: v[r0:r1, c0:c1] for k, v in masks.items()}
    scale = foot_rows / max(r1 - r0, c1 - c0)
    out_hw = (max(1, round((r1 - r0) * scale)), max(1, round((c1 - c0) * scale)))
    return reduce_resolution(t, m, out_hw)
