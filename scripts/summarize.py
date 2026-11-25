"""Summarise an ablation CSV: per-config metrics with 95% bootstrap CIs (stdlib only)."""
import csv
import random
import sys
from collections import defaultdict


def ci(xs, stat=lambda v: sum(v) / len(v), n=1000, seed=0):
    rng = random.Random(seed)
    bs = sorted(stat([rng.choice(xs) for _ in xs]) for _ in range(n))
    return stat(xs), bs[int(0.025 * n)], bs[int(0.975 * n)]


def summarize(path):
    rows = list(csv.DictReader(open(path)))
    by = defaultdict(list)
    for r in rows:
        by[r["config"]].append(r)
    out = {}
    for cfg, rs in by.items():
        conv = [int(r["outcome"] == "consensus") for r in rs]
        acc = [int(r["correct"]) for r in rs]
        ultra = [int(r["intelligibility"] == "Ultra-Strong") for r in rs]
        strong = [int(r["intelligibility"] == "Strong") for r in rs]
        tok = [int(r["total_tokens"]) for r in rs]
        sb = [int(r["sandbox_tokens"]) for r in rs]
        out[cfg] = {"n": len(rs), "convergence": ci(conv), "accurate_convergence": ci(acc),
                    "precision": sum(acc) / max(1, sum(conv)), "ultra": sum(ultra) / len(rs),
                    "strong": sum(strong) / len(rs), "failed": 1 - (sum(ultra) + sum(strong)) / len(rs),
                    "impasse": sum(r["outcome"] == "impasse" for r in rs) / len(rs),
                    "maxit": sum(r["outcome"] == "max_iterations" for r in rs) / len(rs),
                    "noelig": sum(r["outcome"] == "no_eligible_agents" for r in rs) / len(rs),
                    "tokens": ci(tok), "sandbox_share": sum(sb) / max(1, sum(tok)),
                    "tokens_per_accurate": sum(tok) / max(1, sum(acc)),
                    "iterations": sum(int(r["iterations"]) for r in rs) / len(rs),
                    "deltas": sum(int(r["deltas_applied"]) for r in rs) / len(rs)}
    return out


if __name__ == "__main__":
    for cfg, s in summarize(sys.argv[1] if len(sys.argv) > 1 else "results/ablation.csv").items():
        c, a, t = s["convergence"], s["accurate_convergence"], s["tokens"]
        print(f"{cfg:9} conv {c[0]:.3f} [{c[1]:.3f},{c[2]:.3f}]  acc {a[0]:.3f} [{a[1]:.3f},{a[2]:.3f}]  "
              f"prec {s['precision']:.3f}  ultra {s['ultra']:.3f} strong {s['strong']:.3f}  "
              f"imp {s['impasse']:.3f} maxit {s['maxit']:.3f} noel {s['noelig']:.3f}  "
              f"tok {t[0]:.0f}  sb {s['sandbox_share']:.2f}  tok/acc {s['tokens_per_accurate']:.0f}  "
              f"it {s['iterations']:.1f}  deltas {s['deltas']:.2f}")
