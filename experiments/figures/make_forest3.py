# One chart: F1 change per model for rounds 1, 2 and 3, with 95 % intervals.
# Colour shows if the change holds after Holm correction over all 36 comparisons.
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

H = json.load(open("/home/claude/work/ext/out/holm.json"))["results"]
R3 = json.load(open("/home/claude/work/r3/final/out/round3_holdout.json"))["models"]
NAME = {15:"M15 CodeBERT",14:"M14 Hybrid",12:"M12 Trans-8H-2E",5:"M5 Trans-8H",0:"M0 LSTM",
        2:"M2 CNN",13:"M13 Trans-1H-2E",3:"M3 Trans-1H",8:"M8 Trans-16H",4:"M4 Trans-2H",
        7:"M7 Trans-8H+RoBERTa",6:"M6 Trans-8H+BERT"}
# Rows of one format: d, lo, hi, holm, raw. The agent report says no published flag changes in the family of 36.
row = {(x["round"], x["model"]): (x["d_f1"], x["ci_f1"][0], x["ci_f1"][1], x["sig_f1_holm"], x["sig_f1_raw"]) for x in H}
for k, x in R3.items():
    d = x["d_f1_vs_original"]; lo, hi = x["ci95"] if x.get("ci95") else (d, d)
    row[(3, int(k))] = (d, lo, hi, x.get("holm36_sig_f1", False), x.get("p_f1_two_sided", 1) < 0.05)
S = json.load(open("/home/claude/work/results/summary_all.json"))
f1 = {(x["round"], x["model"]): x["f1"] for x in S}   # [original, after the round]
F = {0: {int(k): x["f1_original"] for k, x in R3.items()}, 3: {int(k): x["f1_round3"] for k, x in R3.items()}}
for rnd in (1, 2): F[rnd] = {m: (f1[(rnd, m)][1] if (rnd, m) in f1 else F[0][m]) for m in NAME}
def ranks(d): return {m: i + 1 for i, m in enumerate(sorted(d, key=lambda k: -d[k]))}
RK = {k: ranks(v) for k, v in F.items()}
order = sorted(NAME, key=lambda m: row[(3, m)][0])

GREEN, RED, LIGHT, GREY = "#1a9850", "#d73027", "#9ecae1", "#b0b0b0"
def colour(d, holm, raw): return (GREEN if d > 0 else RED) if holm else (LIGHT if raw else GREY)
TITLE = {1: "Round 1 (MCC rule)", 2: "Round 2 (F1 rule)", 3: "Round 3 (one fixed recipe)"}
fig, axes = plt.subplots(1, 3, figsize=(16, 6.4), sharey=True, gridspec_kw={'wspace': 0.38})
for ax, rnd in zip(axes, (1, 2, 3)):
    ax.axvline(0, color="black", lw=1)
    for y, m in enumerate(order):
        if (rnd, m) not in row: ax.text(0.15, y, "not run", va="center", fontsize=9, color="#888"); continue
        d, lo, hi, holm, raw = row[(rnd, m)]; c = colour(d, holm, raw)
        ax.plot([lo, hi], [y, y], color=c, lw=3, solid_capstyle="round"); ax.plot(d, y, "o", color=c, ms=9, mec="white", mew=1)
        if holm or raw:
            pos = d > 0
            ax.text(hi + 0.25 if pos else d, y if pos else y + 0.18, f"{d:+.1f}", va="center" if pos else "bottom",
                    ha="left" if pos else "center", fontsize=11, fontweight="bold", color=c if holm else "#3b82b8")
    ax.set_title(TITLE[rnd], fontsize=13, pad=14); ax.set_xlabel("F1 change in points (95 % interval)", fontsize=11); ax.set_xlim(-7, 4)
    ax.grid(axis="x", color="#eee"); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels([NAME[m] for m in order], fontsize=11)
# Rank column after each round: F1 rank with original weights -> F1 rank after this round.
for ax, rnd in zip(axes, (1, 2, 3)):
    for y, m in enumerate(order):
        a, b = RK[0][m], RK[rnd][m]; hot = b < a - 1
        ax.text(5.3, y, f"{a} \u2192 {b}" if a != b else f"{a}", va="center", fontsize=11,
                color=GREEN if hot else "#333", fontweight="bold" if hot else "normal")
    ax.text(5.3, len(order) - 0.2, "F1 rank", fontsize=11, fontweight="bold", va="bottom")
h = [Line2D([], [], color=GREEN, marker="o", lw=3, label="Real gain (holds after strict correction)"),
     Line2D([], [], color=RED, marker="o", lw=3, label="Real loss (holds after strict correction)"),
     Line2D([], [], color=LIGHT, marker="o", lw=3, label="Only without correction"),
     Line2D([], [], color=GREY, marker="o", lw=3, label="No clear change")]
fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=10.5)
fig.subplots_adjust(left=0.13, right=0.925, top=0.93, bottom=0.17)
fig.savefig("/home/claude/work/fig/fig0_gains3.png", dpi=200, facecolor="white")
for m in order[::-1]: print(NAME[m], [RK[k][m] for k in (0, 1, 2, 3)])
