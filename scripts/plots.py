"""Ablation plots + paired statistics for the report.

  python scripts/plots.py            # reads results/*.csv, writes results/fig_ablation.pdf/.png and results/stats.txt
"""
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize import summarize  # noqa: E402

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
CFG = ["control", "cf33", "cf66", "cf100"]
X = [0, 33, 66, 100]


def rows(path):
    return list(csv.DictReader(open(path)))


def main(d="results"):
    s = summarize(f"{d}/ablation.csv")
    R = rows(f"{d}/ablation.csv")
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.4, 1.75), gridspec_kw={"wspace": 0.6})
    for key, col, mk, lab in (("convergence", BLUE, "o", "Converged"), ("accurate_convergence", ORANGE, "s", "Accurate")):
        m = [s[c][key][0] * 100 for c in CFG]
        lo = [(s[c][key][0] - s[c][key][1]) * 100 for c in CFG]
        hi = [(s[c][key][2] - s[c][key][0]) * 100 for c in CFG]
        a1.errorbar(X, m, yerr=[lo, hi], color=col, marker=mk, ms=3.5, lw=1.5, capsize=2, label=lab)
    a1.set_xticks(X); a1.set_xlabel("CF agents in panel (%)"); a1.set_ylabel("% of 500 trials")
    a1.set_ylim(40, 100); a1.grid(axis="y", color=GRID, lw=0.6); a1.set_axisbelow(True)
    a1.legend(frameon=False, fontsize=6.5, loc="upper left", handlelength=1.5)
    live = defaultdict(list); sb = defaultdict(list)
    for r in R:
        live[r["config"]].append(int(r["live_tokens"])); sb[r["config"]].append(int(r["sandbox_tokens"]))
    L = [sum(live[c]) / len(live[c]) / 1000 for c in CFG]
    S = [sum(sb[c]) / len(sb[c]) / 1000 for c in CFG]
    a2.bar(X, L, width=22, color=BLUE, label="Live debate")
    a2.bar(X, S, width=22, bottom=[l + 0.25 for l in L], color=ORANGE, label="Sandbox")
    a2.set_ylim(0, 37); a2.set_xticks(X); a2.set_xlabel("CF agents in panel (%)"); a2.set_ylabel("k tokens / trial")
    a2.grid(axis="y", color=GRID, lw=0.6); a2.set_axisbelow(True)
    a2.legend(frameon=False, fontsize=6.5, loc="upper left", handlelength=1.2)
    for ax in (a1, a2):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.savefig(f"{d}/fig_ablation.pdf", bbox_inches="tight"); fig.savefig(f"{d}/fig_ablation.png", dpi=200, bbox_inches="tight")

    # ---- paired statistics ----
    by = defaultdict(dict)
    for r in R:
        by[r["trial"]][r["config"]] = r
    out = []
    for c in ("cf33", "cf66", "cf100"):
        gain = sum(1 for t in by.values() if t["control"]["outcome"] != "consensus" and t[c]["outcome"] == "consensus")
        loss = sum(1 for t in by.values() if t["control"]["outcome"] == "consensus" and t[c]["outcome"] != "consensus")
        wrong = sum(1 for t in by.values() if t["control"]["outcome"] != "consensus" and t[c]["outcome"] == "consensus" and t[c]["correct"] == "0")
        flip = sum(1 for t in by.values() if t["control"]["correct"] == "1" and t[c]["correct"] == "0")
        # exact two-sided sign test on discordant pairs
        from math import comb
        n, k = gain + loss, min(gain, loss)
        p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
        out.append(f"{c}: rescued {gain}, lost {loss}, sign-test p={p:.2e}; rescued-but-wrong {wrong}; correct->not {flip}")
    role = defaultdict(lambda: [0, 0])
    for r in R:
        if r["config"] == "cf33":
            role[r["cf_agents"]][0] += r["outcome"] == "consensus"; role[r["cf_agents"]][1] += 1
    out.append("cf33 by CF role: " + ", ".join(f"{k} {v[0] / v[1]:.3f} (n={v[1]})" for k, v in sorted(role.items())))
    for c in CFG + ["control+"]:
        out.append(f"{c}: live tokens {sum(live[c]) / len(live[c]):.0f}, sandbox {sum(sb[c]) / len(sb[c]):.0f}")
    for f in ("0.4", "0.6", "1.0"):
        p = f"{d}/fidelity_{f}.csv"
        if os.path.exists(p):
            fs = summarize(p)
            out.append(f"fidelity {f}: " + ", ".join(f"{c} conv {fs[c]['convergence'][0]:.3f} acc {fs[c]['accurate_convergence'][0]:.3f}" for c in fs))
    open(f"{d}/stats.txt", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main(*sys.argv[1:])
