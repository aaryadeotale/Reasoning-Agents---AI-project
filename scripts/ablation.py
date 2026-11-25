"""Milestone 4 ablation: vary the share of counterfactual (CF) agents in a 3-agent panel.

  python scripts/ablation.py                       # 500 trials x 5 configs -> results/ablation.csv
  python scripts/ablation.py --trials 50 --fidelity 0.6 --out results/pilot.csv

Configs (paired design: trial i uses the same task and hidden agent profiles in every config):
  control   0% CF agents (all standard PEX agents)
  control+  0% CF, but the impasse threshold is raised by the same grace a CF delta would get
            (checks that gains are not just "more messages")
  cf33      1 of 3 agents CF (random member)      cf66  2 of 3      cf100  all 3
"""
import argparse
import csv
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pxp_blackboard.blackboard import Blackboard
from pxp_blackboard.counterfactual import CounterfactualSandbox
from pxp_blackboard.scheduler import Scheduler, SchedulerConfig
from pxp_blackboard.sim import make_panel, make_task

CONFIGS = {"control": 0, "control+": 0, "cf33": 1, "cf66": 2, "cf100": 3}
FIELDS = ["config", "trial", "domain", "cf_agents", "outcome", "intelligibility", "correct",
          "author_correct", "majority_correct", "iterations", "live_tokens", "sandbox_tokens",
          "total_tokens", "reflections", "deltas_applied", "seconds"]


def run_trial(config: str, trial: int, args) -> dict:
    rng = random.Random(10_000 + trial)           # same task + profiles for every config
    task = make_task(rng, trial)
    n_cf = CONFIGS[config]
    order = random.Random(20_000 + trial).sample(range(3), 3)
    mask = [i in order[:n_cf] for i in range(3)]
    agents = make_panel(task, rng, cf_mask=mask, competence=args.competence, fidelity=args.fidelity, seed=trial)
    beliefs = [a.p.belief for a in agents]
    majority = max(set(beliefs), key=beliefs.count) if len(set(beliefs)) < 3 else beliefs[0]
    threshold = args.impasse + (4 if config == "control+" else 0)
    bb = Blackboard(task.problem, [a.agent_id for a in agents], args.max_messages)
    sch = Scheduler(bb, agents, SchedulerConfig(max_concurrent=1, impasse_threshold=threshold,
                                                max_reflections=args.reflections),
                    sandbox=CounterfactualSandbox(impasse_threshold=args.impasse, seed=trial) if n_cf else None)
    t0 = time.perf_counter()
    r = sch.run()
    live = sum(m.meta.get("prompt_tokens", 0) + m.meta.get("completion_tokens", 0) for m in bb.history())
    correct = r.outcome.value == "consensus" and bb.head.meta.get("answer") == task.gold
    return {"config": config, "trial": trial, "domain": task.domain,
            "cf_agents": "+".join(a.agent_id for a, m in zip(agents, mask) if m) or "-",
            "outcome": r.outcome.value, "intelligibility": r.intelligibility, "correct": int(correct),
            "author_correct": int(beliefs[0] == task.gold), "majority_correct": int(majority == task.gold),
            "iterations": r.iterations, "live_tokens": live, "sandbox_tokens": r.sandbox_tokens,
            "total_tokens": live + r.sandbox_tokens, "reflections": r.reflections,
            "deltas_applied": r.deltas_applied, "seconds": round(time.perf_counter() - t0, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=500)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--fidelity", type=float, default=0.8, help="accuracy of the sandbox's model of peers")
    ap.add_argument("--competence", type=float, default=0.7, help="P(an agent's initial belief is correct)")
    ap.add_argument("--impasse", type=int, default=4)
    ap.add_argument("--reflections", type=int, default=1, help="reflections per CF agent per session")
    ap.add_argument("--max-messages", type=int, default=30)
    ap.add_argument("--out", default="results/ablation.csv")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for cfg in args.configs.split(","):
            for t in range(args.trials):
                w.writerow(run_trial(cfg, t, args))
            print(f"{cfg}: done", flush=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
