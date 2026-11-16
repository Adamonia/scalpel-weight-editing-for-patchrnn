# Charts for the 50 retrained models: gain grid, round 3, and the two causal tests.
import csv, collections, itertools, statistics as st, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = list(csv.DictReader(open("results.csv"))); R3 = list(csv.DictReader(open("round3_results.csv"))); M = list(csv.DictReader(open("mechanism.csv")))
HL = {(int(r["heads"]), int(r["layers"])): r["arch"] for r in R}
H = [1, 2, 4, 8, 16]; BLUE, ORANGE, GREEN, GREY = "#2a6fbb", "#e08a1e", "#1a9850", "#999999"
def vals(rows, arch, key): return [float(r[key]) for r in rows if r["arch"] == arch]
# --- own check of the trend test (architecture level, exact permutations inside each layer) ---
mean = {a: st.mean(vals(R, a, "gain_f1")) for a in set(HL.values())}
def slope(order):
    s = []
    for li, L in enumerate((1, 2)):
        y = [mean[HL[(h, L)]] for h in order[li]]; s.append(np.polyfit(np.log2(H), y, 1)[0])
    return np.mean(s)
obs = slope((H, H)); perms = list(itertools.permutations(H))
cnt = sum(abs(slope((a, b))) >= abs(obs) - 1e-12 for a in perms for b in perms)
print("own check: slope %.4f per doubling, exact p %.5f (%d of %d)" % (obs, cnt / len(perms) ** 2, cnt, len(perms) ** 2))
g = [float(r["gain_f1"]) for r in R]; print("runs", len(g), "gain>0", sum(x > 0 for x in g), "min %.2f mean %.2f" % (min(g), st.mean(g)))
d = [float(r["d_f1_R_vs_E1"]) for r in R3]; d0 = [float(r["d_f1_R_vs_E0"]) for r in R3]
print("round3 vs zero: mean %.3f improved %d | vs original mean %.3f improved %d" % (st.mean(d), sum(x > 0 for x in d), st.mean(d0), sum(x > 0 for x in d0)))
# --- figure A: gain by heads, two lines for layers, with seed dots ---
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6))
for L, col, lab in ((1, ORANGE, "1 encoder layer"), (2, BLUE, "2 encoder layers")):
    ms = []
    for i, h in enumerate(H):
        v = vals(R, HL[(h, L)], "gain_f1"); ms.append(st.mean(v))
        a.plot([i + (-0.08 if L == 1 else 0.08)] * len(v), v, "o", color=col, ms=5, alpha=0.35, mec="none")
    a.plot(range(5), ms, "-o", color=col, lw=2.5, ms=8, mec="white", label=lab)
a.axhline(0, color="black", lw=1); a.set_xticks(range(5)); a.set_xticklabels(H); a.set_xlabel("Attention heads"); a.set_ylabel("F1 gain of the zero edit (points)")
a.set_title("More heads, less gain (50 models)", fontsize=12); a.legend(frameon=False)
# --- figure B: original, zero edit, round 3 per architecture ---
for L, off in ((1, 0), (2, 5.6)):
    for i, h in enumerate(H):
        ar = HL[(h, L)]; x = off + i
        for key, col, dx in (("f1_E0", GREY, -0.22), ("f1_E1", ORANGE, 0), ("f1_R", GREEN, 0.22)):
            v = vals(R3, ar, key); b.plot([x + dx] * len(v), v, "o", color=col, ms=4, alpha=0.3, mec="none"); b.plot(x + dx, st.mean(v), "s", color=col, ms=8, mec="white")
b.set_xticks([0, 1, 2, 3, 4, 5.6, 6.6, 7.6, 8.6, 9.6]); b.set_xticklabels(H + H); b.set_xlabel("Attention heads   (left: 1 layer, right: 2 layers)"); b.set_ylabel("F1 on the test set")
from matplotlib.lines import Line2D
b.legend(handles=[Line2D([], [], marker="s", ls="", color=c, label=l) for c, l in ((GREY, "Original"), (ORANGE, "Zero edit"), (GREEN, "Round 3 recipe"))], frameon=False, loc="lower right")
b.set_title("F1 before and after the two edits", fontsize=12)
for ax in (a, b):
    ax.grid(axis="y", color="#eee"); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig("fig8_grid50.png", dpi=200, facecolor="white")
# --- figure C: causal tests ---
NS = "nsr_margin"
def mnsr(arch, pred): 
    v = [float(r[NS]) for r in M if r["arch"] == arch and pred(r)]; return st.mean(v) if v else None
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4))
cols = {1: "#d73027", 2: "#fc8d59", 4: "#999999", 8: "#74a9cf", 16: "#2a6fbb"}
T = [0.5, 1, 2, 4]
for h in H:
    y = [mnsr(HL[(h, 1)], lambda r, t=t: r["group"] == "temp_%g" % t) for t in T]
    a.plot(range(4), y, "-o", color=cols[h], lw=2.2, ms=7, mec="white", label="%d head%s" % (h, "" if h == 1 else "s"))
a.set_xticks(range(4)); a.set_xticklabels(["0.5\nflatter", "1\noriginal", "2", "4\nsharper"]); a.set_xlabel("Attention temperature factor"); a.set_ylabel("Noise-to-signal of the margin")
a.set_title("Sharper attention lets more noise through\n(one-layer models)", fontsize=12); a.legend(frameon=False, fontsize=9)
for ar, col in (("M5", "#74a9cf"), ("M8", "#2a6fbb"), ("M12", "#1a9850"), ("H16L2", "#00441b")):
    hh = int([r["heads"] for r in R if r["arch"] == ar][0]); ks = [k for k in (1, 2, 4, 8) if k < hh]
    y = [mnsr(ar, lambda r, k=k: r["group"] == "ablate_k%d_rescale" % k) for k in ks]
    y.append(mnsr(ar, lambda r: r["setting"] == "temp_1")); xs = [np.log2(k) for k in ks] + [np.log2(hh)]
    lay = [r["layers"] for r in R if r["arch"] == ar][0]
    b.plot(xs, y, "-o", color=col, lw=2.2, ms=7, mec="white", label="%s (%d heads, %s layer%s)" % (ar, hh, lay, "" if lay == "1" else "s"))
b.axhline(mnsr("M3", lambda r: r["setting"] == "temp_1"), color="#d73027", ls="--", lw=1.5); b.text(4.0, mnsr("M3", lambda r: r["setting"] == "temp_1") + 0.004, "M3 (1 head)", color="#d73027", ha="right", fontsize=9)
b.set_xticks(range(5)); b.set_xticklabels([1, 2, 4, 8, 16]); b.set_xlabel("Heads left active (others switched off, output rescaled)"); b.set_ylabel("Noise-to-signal of the margin")
b.set_title("Fewer active heads, more noise", fontsize=12); b.legend(frameon=False, fontsize=9)
for ax in (a, b):
    ax.grid(axis="y", color="#eee"); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig("fig9_mechanism.png", dpi=200, facecolor="white")
