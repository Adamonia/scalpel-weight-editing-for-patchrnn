# Chart: F1 before and after the zero edit for 30 retrained models, and the gain per model.
import csv, collections, statistics as st, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = list(csv.DictReader(open("results.csv")))
ORDER = ["M3", "M4", "M13", "M5", "M8", "M12"]
LAB = {"M3": "M3\n1 head\n1 layer", "M4": "M4\n2 heads\n1 layer", "M13": "M13\n1 head\n2 layers",
       "M5": "M5\n8 heads\n1 layer", "M8": "M8\n16 heads\n1 layer", "M12": "M12\n8 heads\n2 layers"}
by = collections.defaultdict(list)
for r in R: by[r["arch"]].append(r)
BLUE, ORANGE = "#2a6fbb", "#e08a1e"
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 5))
for i, m in enumerate(ORDER):
    g = [float(r["gain_f1"]) for r in by[m]]
    a.plot([i + (j - 2) * 0.07 for j in range(len(g))], sorted(g), "o", color=ORANGE, ms=8, mec="white")
    a.plot([i - 0.28, i + 0.28], [st.mean(g)] * 2, color="black", lw=2)
    a.text(i, max(g) + 0.12, "%+.2f" % st.mean(g), ha="center", fontsize=10, fontweight="bold")
    f0 = [float(r["f1"]) for r in by[m]]; f1 = [float(r["zero_f1"]) for r in by[m]]
    for x, y in zip(f0, f1): b.plot([i - 0.15, i + 0.15], [x, y], color="#bbb", lw=1, zorder=1)
    b.plot([i - 0.15] * 5, f0, "o", color=BLUE, ms=7, mec="white", label="Original" if i == 0 else None)
    b.plot([i + 0.15] * 5, f1, "o", color=ORANGE, ms=7, mec="white", label="After the zero edit" if i == 0 else None)
a.axhline(0, color="black", lw=1); a.set_ylim(-0.2, 3.1); a.set_xlim(-0.6, 5.8)
a.set_title("F1 gain of the zero edit, 5 seeds per architecture", fontsize=12); a.set_ylabel("F1 change in points")
b.set_title("F1 before and after", fontsize=12); b.set_ylabel("F1 on the test set"); b.legend(frameon=False, loc="lower right")
for ax in (a, b):
    ax.set_xticks(range(6)); ax.set_xticklabels([LAB[m] for m in ORDER], fontsize=9)
    ax.grid(axis="y", color="#eee"); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig("fig7_seeds.png", dpi=200, facecolor="white")
