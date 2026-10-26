"""
Evaluate three classifiers at every resolution.

  python evaluate.py --out out --seeds 0 1 2        ablation -> out/results.csv
  python evaluate.py --permute out/features_32x24.csv   shuffled-label sanity check

Protocol (as in the spec):
  - stratified group k-fold (k=5) on subject_id, sorted before splitting, fixed seeds
  - imputer and scaler are fitted inside each training fold (they sit in the pipeline)
  - hyperparameters are tuned by inner cross-validation, never on the test fold
  - metrics are computed per fold and then averaged, never pooled
Positive class = diabetic (DM). Headline metrics: balanced accuracy and AUC, because
122 vs 45 subjects means "always say diabetic" already scores about 73% plain accuracy.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pivot import pivot_subjects

MODELS = ["lr", "svm", "rf"]
METRICS = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "auc"]
ORDER = ["native", "160x120", "80x60", "32x24", "foot24"]


def make_model(name, seed):
    steps = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    if name == "lr":
        est = LogisticRegression(penalty="l2", max_iter=5000, class_weight="balanced")
        grid = {"clf__C": [0.01, 0.1, 1, 10]}
    elif name == "svm":
        est = SVC(kernel="rbf", class_weight="balanced")
        grid = {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01, 0.1]}
    elif name == "rf":
        est = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     random_state=seed, n_jobs=-1)
        grid = {"clf__max_depth": [3, 5, None]}
    else:
        raise ValueError(name)
    pipe = Pipeline(steps + [("clf", est)])
    inner = StratifiedKFold(3, shuffle=True, random_state=seed)
    return GridSearchCV(pipe, grid, cv=inner, scoring="balanced_accuracy")


def scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def run_cv(X, y, groups, model_name, seed, n_splits=5):
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for tr, te in cv.split(X, y, groups):
        assert not set(groups[tr]) & set(groups[te]), "subject appears in train and test"
        m = make_model(model_name, seed)
        m.fit(X[tr], y[tr])
        pred = m.predict(X[te])
        tn, fp, fn, tp = confusion_matrix(y[te], pred, labels=[0, 1]).ravel()
        rows.append({
            "accuracy": accuracy_score(y[te], pred),
            "balanced_accuracy": balanced_accuracy_score(y[te], pred),
            "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "auc": roc_auc_score(y[te], scores(m, X[te])),
        })
    return pd.DataFrame(rows)


def load_xy(csv_path):
    df = pd.read_csv(csv_path)
    sub, dropped = pivot_subjects(df)
    sub = sub.sort_index()                       # sorted subject ids -> reproducible splits
    X = sub.drop(columns=["group", "sex"]).to_numpy(dtype=float)
    y = (sub["group"] == "DM").astype(int).to_numpy()
    groups = sub.index.to_numpy()
    print(f"{os.path.basename(csv_path)}: {len(sub)} subjects "
          f"({int(y.sum())} DM, {int((1 - y).sum())} CG), {dropped} dropped for a missing foot, "
          f"{X.shape[1]} features, majority-class accuracy {max(y.mean(), 1 - y.mean()):.2f}")
    return X, y, groups


def label_of(path):
    return os.path.basename(path).replace("features_", "").replace(".csv", "")


def ablation(out_dir, seeds):
    paths = sorted(glob.glob(os.path.join(out_dir, "features_*.csv")),
                   key=lambda p: ORDER.index(label_of(p)) if label_of(p) in ORDER else 99)
    rows = []
    for path in paths:
        X, y, groups = load_xy(path)
        for name in MODELS:
            folds = pd.concat([run_cv(X, y, groups, name, s) for s in seeds], ignore_index=True)
            row = {"resolution": label_of(path), "model": name,
                   "n_subjects": len(y), "n_folds": len(folds)}
            for metric in METRICS:
                row[f"{metric}_mean"] = folds[metric].mean()
                row[f"{metric}_sd"] = folds[metric].std()
            rows.append(row)
            print(f"  {label_of(path):>8} {name:>3}  balanced acc "
                  f"{row['balanced_accuracy_mean']:.3f} +/- {row['balanced_accuracy_sd']:.3f}   "
                  f"AUC {row['auc_mean']:.3f}")
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    print("Wrote", os.path.join(out_dir, "results.csv"))


def permutation_check(csv_path, n_perm=10, seed=0):
    """Shuffle the labels (one row per subject, so each subject keeps one shuffled label)
    and re-run. Scores should fall to chance (about 0.5)."""
    X, y, groups = load_xy(csv_path)
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_perm):
        res = run_cv(X, rng.permutation(y), groups, "lr", seed=i)
        out.append(res[["balanced_accuracy", "auc"]].mean())
    out = pd.DataFrame(out)
    print(f"Shuffled labels: balanced accuracy {out['balanced_accuracy'].mean():.3f} "
          f"(sd {out['balanced_accuracy'].std():.3f}), AUC {out['auc'].mean():.3f}")
    if out["balanced_accuracy"].mean() > 0.58:
        print("WARNING: well above chance on shuffled labels. Suspect leakage in the split or features.")
    else:
        print("OK: shuffled-label scores sit at chance.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--permute", help="path to a features CSV to run the shuffled-label check on")
    args = ap.parse_args()
    if args.permute:
        permutation_check(args.permute)
    else:
        ablation(args.out, args.seeds)
