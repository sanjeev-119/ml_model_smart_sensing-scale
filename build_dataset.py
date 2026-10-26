"""
Build one feature CSV (one row per foot) and one region-size CSV.

Examples:
  python build_dataset.py --root /path/to/ThermoDataBase --mode native
  python build_dataset.py --root /path/to/ThermoDataBase --mode image --size 160x120
  python build_dataset.py --root /path/to/ThermoDataBase --mode image --size 80x60
  python build_dataset.py --root /path/to/ThermoDataBase --mode image --size 32x24
  python build_dataset.py --root /path/to/ThermoDataBase --mode foot --foot-rows 24

Outputs go to out/features_<label>.csv and out/region_sizes_<label>.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from downsample import reduce_resolution, reduce_to_foot_rows
from features import foot_features
from loader import foot_stems, index_files, load_foot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder holding the unzipped database")
    ap.add_argument("--mode", choices=["native", "image", "foot"], default="native")
    ap.add_argument("--size", default="32x24", help="image mode, WIDTHxHEIGHT, e.g. 160x120")
    ap.add_argument("--foot-rows", type=int, default=24, help="foot mode, pixels along the foot")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    if args.mode == "native":
        label = "native"
    elif args.mode == "image":
        label = args.size
    else:
        label = f"foot{args.foot_rows}"

    index = index_files(args.root)
    stems = foot_stems(index)
    print(f"Found {len(stems)} whole-foot files (expected 334)")

    rows, size_rows, skipped = [], [], 0
    for stem in stems:
        try:
            meta, temp, masks = load_foot(index, stem)
        except KeyError as e:
            print("skipping:", e)
            skipped += 1
            continue

        if args.mode == "image":
            w, h = (int(v) for v in args.size.lower().split("x"))
            temp, masks = reduce_resolution(temp, masks, (h, w))
        elif args.mode == "foot":
            temp, masks = reduce_to_foot_rows(temp, masks, args.foot_rows)

        feats, sizes = foot_features(temp, masks)
        base = {
            "subject_id": meta["subject_id"],
            "side": meta["side"],
            "group": meta["group"],
            "sex": meta["sex"],
            "resolution": label,
        }
        rows.append({**base, **feats})
        size_rows.append({**base, **sizes})

    df = pd.DataFrame(rows)
    sizes_df = pd.DataFrame(size_rows)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    df.to_csv(out / f"features_{label}.csv", index=False)
    sizes_df.to_csv(out / f"region_sizes_{label}.csv", index=False)

    subj = df.drop_duplicates("subject_id").groupby("group").size().to_dict()
    print(f"Wrote {len(df)} feet ({skipped} skipped). Subjects per group: {subj} (expected DM 122, CG 45)")
    print(f"NaN cells in features: {int(df.isna().sum().sum())}")
    small = int((sizes_df[["lca", "lpa", "mca", "mpa"]] < 10).sum().sum())
    print(f"Region measurements under 10 pixels: {small} (unreliable statistics)")


if __name__ == "__main__":
    main()
