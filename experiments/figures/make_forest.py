# Make one chart: F1 change per model, round 1 and round 2, with 95 % intervals.
# Colour shows if the change holds after Holm correction.
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = json.load(open("/home/claude/work/ext/out/holm.json"))["results"]
S = json.load(open("/home/claude/work/results/summary_all.json"))
NAME = {15:"M15 CodeBERT",14:"M14 Hybrid",12:"M12 Trans-8H-2E",5:"M5 Trans-8H",0:"M0 LSTM",
        2:"M2 CNN",13:"M13 Trans-1H-2E",3:"M3 Trans-1H",8:"M8 Trans-16H",4:"M4 Trans-2H",
        7:"M7 Trans-8H+RoBERTa",6:"M6 Trans-8H+BERT"}
r = {(x["round"], x["model"]): x for x in H}
f1 = {(x["round"], x["model"]): x["f1"] for x in S}

# Rank by F1: original, and after round 2.
orig = {m: f1[(1, m)][0] for m in NAME}
fin = {m: f1[(2, m)][1] if (2, m) in f1 else f1[(1, m)][0] for m in NAME}
def ranks(d):
    return {m: i + 1 for i, m in enumerate(sorted(d, key=lambda k: -d[k]))}
ro, rf = ranks(orig), ranks(fin)

# Order rows by round 2 change (largest gain on top).
def d2(m): return r[(2, m)]["d_f1"] if (2, m) in r else 0.0
order = sorted(NAME, key=d2)

GREEN, RED, LIGHT, GREY = "#1a9850", "#d73027", "#9ecae1", "#b0b0b0"
def colour(x):
    if x["sig_f1_holm"]: return GREEN if x["d_f1"] > 0 else RED
    if x["sig_f1_raw"]: return LIGHT
    return GREY

fig, axes = plt.subplots(1, 2, figsize=(11, 6.2), sharey=True)
for ax, rnd in zip(axes, (1, 2)):
    ax.axvline(0, color="black", lw=1)
    for y, m in enumerate(order):
        x = r.get((rnd, m))
        if x is None:
            ax.text(0.15, y, "not run", va="center", fontsize=9, color="#888")
            continue
        c = colour(x); lo, hi = x["ci_f1"]
        ax.plot([lo, hi], [y, y], color=c, lw=3, solid_capstyle="round")
        ax.plot(x["d_f1"], y, "o", color=c, ms=9, mec="white", mew=1)
        if x["sig_f1_holm"]:
            pos = x["d_f1"] > 0
            ax.text(hi + 0.25 if pos else x["d_f1"], y if pos else y + 0.18, f"{x['d_f1']:+.1f}",
                    va="center" if pos else "bottom", ha="left" if pos else "center",
                    fontsize=11, fontweight="bold", color=c)
    ax.set_title(f"Round {rnd}" + (" (MCC rule)" if rnd == 1 else " (F1 rule, final)"), fontsize=13)
    ax.set_xlabel("F1 change in points (95 % interval)", fontsize=11)
    ax.set_xlim(-7, 4)
    ax.grid(axis="x", color="#eee"); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[0].set_yticks(range(len(order)))
axes[0].set_yticklabels([NAME[m] for m in order], fontsize=11)
# Rank column at the right.
for y, m in enumerate(order):
    a, b = ro[m], rf[m]
    txt = f"{a} → {b}" if a != b else f"{a}"
    axes[1].text(4.6, y, txt, va="center", fontsize=11,
                 color=GREEN if b < a - 1 else "#333", fontweight="bold" if b < a - 1 else "normal")
axes[1].text(4.6, len(order) - 0.2, "F1 rank", fontsize=11, fontweight="bold", va="bottom")
from matplotlib.lines import Line2D
h = [Line2D([], [], color=GREEN, marker="o", lw=3, label="Real gain (holds after strict correction)"),
     Line2D([], [], color=RED, marker="o", lw=3, label="Real loss (holds after strict correction)"),
     Line2D([], [], color=LIGHT, marker="o", lw=3, label="Only without correction"),
     Line2D([], [], color=GREY, marker="o", lw=3, label="No clear change")]
fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=10.5)
fig.tight_layout(rect=(0, 0.09, 0.93, 1))
for ext in ("png", "svg"):
    fig.savefig(f"/home/claude/work/fig/fig0_gains.{ext}", dpi=200, facecolor="white")
for m in order[::-1]: print(NAME[m], ro[m], rf[m], d2(m))
