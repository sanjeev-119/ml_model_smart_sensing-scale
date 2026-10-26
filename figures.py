
"""Plot balanced accuracy for Native vs 32x24 resolution.

python figures.py
Reads out/results.csv and writes out/performance_vs_resolution.png
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ORDER = ["native", "32x24"]
NAMES = {
    "lr": "Logistic regression",
    "svm": "SVM (RBF)",
    "rf": "Random forest",
}

res = pd.read_csv("out/results.csv")

# Keep only Native and 32x24
res = res[res["resolution"].isin(ORDER)].copy()
levels = [r for r in ORDER if r in set(res["resolution"])]

if not levels:
    raise ValueError(
        "Neither 'native' nor '32x24' was found in out/results.csv. "
        "Check the resolution column."
    )

fig, ax = plt.subplots(figsize=(7, 4.5))

for i, (model, g) in enumerate(res.groupby("model")):
    g = g.set_index("resolution").reindex(levels)

    xs = [k + (i - 1) * 0.08 for k in range(len(levels))]

    ax.errorbar(
        xs,
        g["balanced_accuracy_mean"],
        yerr=g["balanced_accuracy_sd"],
        marker="o",
        capsize=3,
        label=NAMES.get(model, model),
    )

ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
ax.text(len(levels) - 1, 0.505, "chance",
        ha="right", color="gray", fontsize=9)

if "32x24" in levels:
    k = levels.index("32x24")
    ax.axvspan(k - 0.3, k + 0.3, color="orange", alpha=0.15)
    ax.text(k, 0.40, "MLX90640\n(whole image)",
            ha="center", fontsize=9)

ax.set_xticks(range(len(levels)))
ax.set_xticklabels(levels)
ax.set_xlabel("Resolution")
ax.set_ylabel("Balanced accuracy (mean +/- SD over folds)")
ax.set_ylim(0.35, 1.0)
ax.set_title("Native vs MLX90640 (32x24) resolution")
ax.legend(loc="upper right")

fig.tight_layout()
fig.savefig("out/performance_vs_resolution.png", dpi=200)
print("Wrote out/performance_vs_resolution.png")
