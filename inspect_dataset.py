"""
Run this FIRST, before anything else:
    python inspect_dataset.py /path/to/ThermoDataBase

It checks the counts against the paper and shows how the files are laid out, so the
assumptions in loader.py can be confirmed (units, mask shape, background value).
"""

import sys
from collections import Counter

import numpy as np

from loader import REGIONS, foot_stems, index_files, parse_name, read_matrix

root = sys.argv[1]
index = index_files(root)
stems = foot_stems(index)

print(f"CSV files found: {len(index)}   (expected 1670)")
print(f"Whole-foot files: {len(stems)}   (expected 334)")
subjects = {}
for s in stems:
    m = parse_name(s)
    subjects.setdefault(m["subject_id"], set()).add(m["side"])
print("Subjects per group:", dict(Counter(sid[:2] for sid in subjects)), "  (expected DM 122, CG 45)")
one_foot = [sid for sid, sides in subjects.items() if len(sides) < 2]
print("Subjects with only one foot:", len(one_foot), one_foot[:10])

missing = [f"{s}_{r.upper()}" for s in stems for r in REGIONS if f"{s}_{r.upper()}" not in index]
print("Missing angiosome files:", len(missing), missing[:5])

for sample in (stems[0], stems[len(stems) // 2], stems[-1]):
    print("\n---", sample, "---")
    temp = read_matrix(index[sample])
    finite = temp[np.isfinite(temp)]
    print(f"whole foot: shape {temp.shape}, min {finite.min():.2f}, max {finite.max():.2f}, "
          f"mean {finite.mean():.2f}, zeros {(temp == 0).sum()}, NaN {np.isnan(temp).sum()}")
    if not (15 < finite.mean() < 45):
        print("  WARNING: values do not look like degrees C. Find the conversion before going on.")
    union = np.zeros(temp.shape, dtype=bool)
    total = 0
    for r in REGIONS:
        key = f"{sample}_{r.upper()}"
        if key not in index:
            continue
        a = read_matrix(index[key])
        same = a.shape == temp.shape
        mask = np.isfinite(a) & (a > 0) if same else None
        n = int(mask.sum()) if same else a.size
        print(f"  {r.upper()}: shape {a.shape} {'(same as whole foot)' if same else '(DIFFERENT - cropped?)'}, "
              f"region pixels {n}")
        if same:
            total += n
            union |= mask
    if union.any():
        print(f"  disjoint regions: {total == int(union.sum())}   "
              f"foot covers {union.mean() * 100:.1f}% of the image")
