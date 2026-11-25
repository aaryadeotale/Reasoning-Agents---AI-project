# Team Epsilon: 3-minute video script

Target length: **2:50** (about 400 spoken words). Four speakers, about 40 seconds each. Record the screen at 1080p; the dashboard runs with `python scripts/serve.py`.

Before recording:
- Open `http://127.0.0.1:8000/?scenario=deadlock&speed=2` in one tab and `http://127.0.0.1:8000/?scenario=counterfactual&speed=1` in another (don't press Start yet).
- Have `results/fig_ablation.png` and the results table from the report ready to show full screen.

---

## 0:00–0:35 · Shriya: the problem and PXP
**Screen:** title slide (project title, team name, four names), then the dashboard with the legend visible.

> "Hi, we're Team Epsilon. Our project is an intelligible blackboard architecture with counterfactual agents.
> When LLM agents pass messages down a chain, errors compound, and small models often argue themselves into deadlock.
> We put every agent on one shared blackboard. Each message has a prediction, an explanation, and a tag from the PXP intelligibility protocol: RATIFY, REVISE, REFUTE or REJECT. One validator checks every move, so a malformed or out-of-date message can never reach the board."

## 0:35–1:15 · Saee and Priyanshu: board, scheduler, live dashboard
**Screen:** start the **Deadlock** scenario at 2×. Point at the nodes appearing, the negative-streak meter, then the red DEADLOCK banner.

> **Saee:** "The blackboard is an append-only, thread-safe log. Every agent sees the same state, and we keep the full history, which you can scrub through here."
> **Priyanshu:** "An event-driven scheduler decides who acts next. Authors only act to answer objections, and it stops on consensus, on an impasse after four objections in a row, or on a message limit. Here two reviewers keep objecting, the streak hits four, and the session deadlocks. Without counterfactual agents, about half our panels end like this."

## 1:15–2:05 · Aarya: counterfactual agents
**Screen:** start **Counterfactual rescue** at 1×. Let it run until the purple sandbox panel appears; hover the best (green) timeline; then show the green Consensus · Ultra-Strong banner.

> "Our main addition is the counterfactual agent. Here the analyst answers only the latest objection, so each fix undoes the last one.
> When the board deadlocks, the analyst pauses, copies the history into a private sandbox, and asks: if I had said something different earlier, would we agree now? It tries several alternative messages, simulates how the others would respond, and scores each timeline.
> The best one answers both objections at once. It gets posted to the live board as a REVISE, both reviewers ratify, and the session ends in Ultra-Strong consensus."

## 2:05–2:45 · Shriya: results
**Screen:** `fig_ablation.png` full screen, then the results table.

> "We ran a paired ablation of 500 simulated three-agent panels per setting, changing the share of counterfactual agents.
> Convergence rises from 51 percent with none to 87 percent with all three, and accurate answers from 47 to 78 percent. Counterfactual agents never broke a panel that already worked. Simply allowing more messages changed nothing.
> The cost is about three and a half times the tokens, and some rescued sessions agree on a wrong answer.
> These are simulated agents, not a 7B model, and running real models on the benchmark datasets is our next step."

## 2:45–2:55 · Close
**Screen:** GitHub URL and team names.

> "The code, tests and the full ablation are on our public GitHub. Thank you!"

---

**Tips:** read the lines naturally rather than word for word; cut the *Saee* lines first if you run long. Check the final file is under 3:00 and under 1 GB.
