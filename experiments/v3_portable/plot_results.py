"""v3 chart from results.json (run_shift.py): realised recall and passages after a shift."""
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

res = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else "results.json").read_text())
out = sys.argv[2] if len(sys.argv) > 2 else "v3_shift.png"
target = res["target_recall"]
ROWS = [("dataset_shift", t) for t in ("squad", "scifact", "nfcorpus")] + [("model_shift", t) for t in ("squad", "scifact", "nfcorpus")]
NAMES = {"squad": "SQuAD", "scifact": "SciFact", "nfcorpus": "NFCorpus"}
LABEL = lambda s, t: f"{NAMES[t]}: new {'dataset' if s == 'dataset_shift' else 'model (BGE→MiniLM)'}"
METHODS = [  # key, legend, colour, marker
    ("naive", "Naive transfer (0 labels)", "#eb6834", "X"),
    ("portable", "Standardised features (0 labels)", "#eda100", "D"),
    ("worst-source", "Worst-source cutoff (0 labels)", "#1baf7a", "^"),
    ("fewshot-crc-50", "Few-shot CRC (50 labels)", "#2a78d6", "o"),
    ("in-domain", "In-domain v2 (~500 labels)", "#52514e", "s"),
]
INK, MUTED, GRID, BG = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})
fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5.6), facecolor=BG, sharey=True, gridspec_kw={"width_ratios": [1.25, 1]})
ys = list(range(len(ROWS)))[::-1]
for ax in (a, b):
    ax.set_facecolor(BG); ax.grid(True, axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for y in ys:
        ax.axhline(y, color=GRID, lw=0.6, zorder=0)
a.axvline(target, color=INK, lw=1, ls="--"); a.text(target, len(ROWS) - 0.35, f" target {target:.0%}", color=INK, fontsize=9, va="bottom")
for i, (key, label, col, m) in enumerate(METHODS):
    off = (i - 2) * 0.12
    rec = [res[s][t][key]["recall"] for s, t in ROWS]
    lo = [res[s][t][key]["recall_min"] for s, t in ROWS]
    kk = [res[s][t][key]["mean_k"] for s, t in ROWS]
    yy = [y + off for y in ys]
    a.hlines(yy, lo, rec, color=col, lw=1.6, alpha=0.8)
    a.scatter(rec, yy, color=col, marker=m, s=60, edgecolor=BG, linewidth=1.2, zorder=3, label=label)
    b.scatter(kk, yy, color=col, marker=m, s=60, edgecolor=BG, linewidth=1.2, zorder=3)
a.set_yticks(ys); a.set_yticklabels([LABEL(s, t) for s, t in ROWS])
a.set_xlim(0.35, 1.01); a.set_xlabel("Realised recall on the new data (marker: mean of 10 splits; line: down to worst split)")
b.set_xlim(0, 21); b.set_xlabel("Passages sent per question")
b.axvline(5, color=MUTED, lw=1, ls=":"); b.text(5, len(ROWS) - 0.35, " top-5", color=MUTED, fontsize=9, va="bottom")
fig.legend(*a.get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(out, dpi=150, facecolor=BG)
print("wrote", out)
