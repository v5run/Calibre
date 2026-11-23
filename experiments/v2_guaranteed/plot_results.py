"""Guarantee-validity chart from results.json (validate_pilot.py)."""
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

res = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else "results.json").read_text())
out = sys.argv[2] if len(sys.argv) > 2 else "v2_guarantee_validity.png"
T = [0.8, 0.9, 0.95]
SERIES = [("v2_expected", "Expected recall (ρ, nominal)", "#2a78d6", "o"),
          ("v2_guarantee", "CRC guarantee (on average)", "#eb6834", "s"),
          ("v2_highconf", "CRC high-confidence (δ = 0.1)", "#1baf7a", "^")]
NAMES = {"squad": "SQuAD", "scifact": "SciFact", "nfcorpus": "NFCorpus"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False})
fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), facecolor="#fcfcfb")
for j, (ds, name) in enumerate(NAMES.items()):
    r = res[ds]
    a, b = axes[0, j], axes[1, j]
    for ax in (a, b):
        ax.set_facecolor("#fcfcfb"); ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    a.plot([0.78, 0.97], [0.78, 0.97], color=MUTED, lw=1, ls="--", label="Target (y = x)")
    for i, (key, label, col, m) in enumerate(SERIES):
        xs = [t + (i - 1) * 0.004 for t in T]
        rec = [r[f"{key}_{t}"]["recall"] for t in T]
        lo = [r[f"{key}_{t}"]["recall"] - r[f"{key}_{t}"]["recall_min"] for t in T]
        a.errorbar(xs, rec, yerr=[lo, [0] * 3], color=col, marker=m, ms=7, lw=2, capsize=0,
                   mec="#fcfcfb", mew=1.5, label=label)
        b.plot(T, [r[f"{key}_{t}"]["mean_k"] for t in T], color=col, marker=m, ms=7, lw=2,
               mec="#fcfcfb", mew=1.5)
    b.axhline(r["top5"]["mean_k"], color=MUTED, lw=1, ls=":")
    b.text(0.78, r["top5"]["mean_k"] + 0.5, f"fixed top-5: recall {r['top5']['recall']:.2f}",
           ha="left", va="bottom", color=MUTED, fontsize=8.5)
    a.set_title(name, color=INK, fontsize=12, loc="left", fontweight="bold")
    a.set_xticks(T); b.set_xticks(T); a.set_xlim(0.775, 0.975); b.set_xlim(0.775, 0.975)
    a.set_ylim(0.72, 1.01); b.set_ylim(0, 21)
    b.set_xlabel("Recall target")
    if j == 0:
        a.set_ylabel("Realised recall on test queries"); b.set_ylabel("Passages sent per query")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 1.0))
fig.text(0.5, 0.005, "Markers: mean over 20 random train/calibration/test splits; whisker: worst split. "
         "Recall counts relevant passages in the top-20 candidates. BGE-small + BM25.\n"
         "High-confidence at 0.95 is infeasible with ~330 calibration queries, so it falls back to all 20 candidates.",
         ha="center", color=MUTED, fontsize=8.5)
fig.tight_layout(rect=(0, 0.04, 1, 0.95))
fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
print("wrote", out)
