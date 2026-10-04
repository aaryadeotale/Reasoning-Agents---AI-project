"""Run: python demo.py -- REJECT -> REVISE -> RATIFY using only the four 4R tags."""
from pxp_blackboard import Blackboard, Draft, ScriptedAgent, Scheduler, SchedulerConfig, Tag


def analyst(s):
    h = s["head"]
    if h["agent_id"] == "Analyst" and "excluding" not in h["explanation"]:
        return Draft(
            "Analyst", Tag.REVISE, h["seq"],
            "Revenue fell 4%",
            "Q3 total was below Q2, excluding the one-off asset sale",
        )
    return None


def skeptic(s):
    h = s["head"]
    if "excluding" in h["explanation"]:
        return Draft("Skeptic", Tag.RATIFY, h["seq"])
    return Draft("Skeptic", Tag.REJECT, h["seq"], None, "Explanation ignores the one-off asset sale")


def auditor(s):
    return Draft("Auditor", Tag.RATIFY, s["head"]["seq"])


agents = [
    ScriptedAgent("Analyst", analyst, 5),
    ScriptedAgent("Skeptic", skeptic),
    ScriptedAgent("Auditor", auditor),
]

board = Blackboard(
    "Did revenue fall in Q3?",
    [a.agent_id for a in agents],
    initial_prediction="Revenue fell 4%",
    initial_explanation="Q3 total was below Q2",
    initial_agent_id="Analyst",
)

result = Scheduler(board, agents, SchedulerConfig(max_concurrent=1)).run()

for m in board.history():
    print(
        f"#{m.seq} {m.agent_id:8} {m.tag.value:8} -> {m.target_id} "
        f"{m.prediction or ''} | {m.explanation or ''}"
    )
print(result.to_dict())
