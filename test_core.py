import os
import sys

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from downsample import reduce_resolution, union
from features import foot_features


def fake_foot(h=240, w=320):
    """A rectangular 'foot' split into four disjoint regions."""
    masks = {r: np.zeros((h, w), dtype=bool) for r in ["lca", "lpa", "mca", "mpa"]}
    masks["lca"][20:120, 100:160] = True
    masks["mca"][20:120, 160:220] = True
    masks["lpa"][120:220, 100:160] = True
    masks["mpa"][120:220, 160:220] = True
    return np.full((h, w), 30.0), masks


def test_uniform_temperature_survives_downsampling():
    temp, masks = fake_foot()
    for hw in [(120, 160), (60, 80), (24, 32)]:
        small, m = reduce_resolution(temp, masks, hw)
        foot = union(m)
        assert foot.any()
        assert np.allclose(small[foot], 30.0, atol=0.05)


def test_downsampled_masks_are_disjoint():
    temp, masks = fake_foot()
    _, m = reduce_resolution(temp, masks, (24, 32))
    stacked = np.stack(list(m.values())).sum(axis=0)
    assert stacked.max() <= 1


def test_hot_patch_gives_expected_features():
    temp, masks = fake_foot(24, 32)
    temp = np.full((24, 32), 25.0)
    masks = {r: np.zeros((24, 32), dtype=bool) for r in ["lca", "lpa", "mca", "mpa"]}
    masks["mpa"][10:20, 12:20] = True
    temp[masks["mpa"]] = 31.0
    temp[12:14, 14:16] = 36.0                       # hot patch inside mpa
    for r in ["lca", "lpa", "mca"]:
        masks[r][2:5, 2 + 4 * ["lca", "lpa", "mca"].index(r): 5 + 4 * ["lca", "lpa", "mca"].index(r)] = True
        temp[masks[r]] = 31.0
    feats, sizes = foot_features(temp, masks)
    assert feats["mpa_max"] == 36.0
    assert feats["mpa_hse"] > 0
    assert feats["lca_hse"] == 0
    assert sizes["mpa"] == 80


def test_no_subject_in_both_train_and_test():
    rng = np.random.default_rng(0)
    groups = np.array([f"S{i:03d}" for i in range(167)])
    y = np.array([1] * 122 + [0] * 45)
    X = rng.normal(size=(167, 5))
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    for tr, te in cv.split(X, y, groups):
        assert not set(groups[tr]) & set(groups[te])
