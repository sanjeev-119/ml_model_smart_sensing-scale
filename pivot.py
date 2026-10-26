"""
Turn per-foot rows into per-subject rows.

For every feature f this gives:
  f_avg   = (left + right) / 2     overall level
  f_asym  = left - right           the clinically meaningful left/right difference
Subjects missing one foot are dropped and counted.
n_pixels is excluded from the model features: it mostly reflects foot size and
camera distance, which would be a shortcut rather than a thermal signal.
"""

import pandas as pd

META = ["subject_id", "side", "group", "sex", "resolution"]
EXCLUDE = ["n_pixels"]


def pivot_subjects(df):
    feats = [c for c in df.columns if c not in META + EXCLUDE]
    left = df[df["side"] == "L"].set_index("subject_id")
    right = df[df["side"] == "R"].set_index("subject_id")
    both = sorted(set(left.index) & set(right.index))
    dropped = df["subject_id"].nunique() - len(both)

    out = pd.DataFrame(index=pd.Index(both, name="subject_id"))
    out["group"] = left.loc[both, "group"]
    out["sex"] = left.loc[both, "sex"]
    for f in feats:
        out[f"{f}_avg"] = (left.loc[both, f] + right.loc[both, f]) / 2
        out[f"{f}_asym"] = left.loc[both, f] - right.loc[both, f]
    return out, dropped
