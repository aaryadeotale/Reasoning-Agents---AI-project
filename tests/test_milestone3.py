"""Milestone 3: counterfactual sandbox and its scheduler hook."""
from pxp_blackboard.blackboard import Blackboard
from pxp_blackboard.counterfactual import CounterfactualSandbox, consensus_distance
from pxp_blackboard.prompts import PromptLibrary
from pxp_blackboard.scheduler import Outcome, Scheduler, SchedulerConfig
from pxp_blackboard.server.sessions import AgentSpec, Session, SessionRequest
from pxp_blackboard.sim import SimAgent, make_panel, make_task

LIB = PromptLibrary.load()


def _run(cf: bool, **kw):
    s = Session(SessionRequest(scenario="counterfactual", **kw), LIB)
    if not cf:
        s.scheduler._cf_budget = {}
    s.start()
    s.join(20)
    return s


def test_standard_panel_deadlocks():
    s = _run(cf=False)
    assert s.result.outcome is Outcome.IMPASSE and s.result.reflections == 0


def test_counterfactual_agent_breaks_the_deadlock():
    s = _run(cf=True)
    r = s.result
    assert r.outcome is Outcome.CONSENSUS and r.intelligibility == "Ultra-Strong"
    assert r.reflections == 1 and r.deltas_applied == 1 and r.sandbox_tokens > 0
    delta = [m for m in s.board.history() if m.meta.get("counterfactual")]
    assert len(delta) == 1 and delta[0].tag.value == "REVISE"


def test_sandbox_never_touches_the_live_board():
    s = _run(cf=False)
    before = s.board.snapshot()
    sb = CounterfactualSandbox()
    ref = sb.reflect(before, s.scheduler.agents["Analyst"], s.scheduler.agents, s.scheduler.policy)
    assert ref.timelines and ref.best.outcome == "consensus"
    assert s.board.snapshot() == before


def test_reflection_events_are_streamed():
    s = _run(cf=True)
    types = [e["type"] for e in s.hub.events()]
    assert "counterfactual_started" in types and "counterfactual_finished" in types
    fin = [e for e in s.hub.events() if e["type"] == "counterfactual_finished"][0]
    assert fin["applied"] and fin["reflection"]["best"]["outcome"] == "consensus"


def test_budget_limits_reflections():
    s = _run(cf=True, max_reflections=0)
    assert s.result.outcome is Outcome.IMPASSE and s.result.reflections == 0


def test_consensus_distance():
    bb = Blackboard("q", ["A", "B", "C"])
    assert consensus_distance(bb.snapshot()) == 3


def test_sim_counterfactual_improves_convergence_on_average():
    import random
    wins = {False: 0, True: 0}
    for cf in (False, True):
        for seed in range(40):
            rng = random.Random(seed)
            task = make_task(rng, seed)
            agents = make_panel(task, rng, cf_mask=[cf, cf, cf], seed=seed)
            bb = Blackboard(task.problem, [a.agent_id for a in agents], 30)
            sch = Scheduler(bb, agents, SchedulerConfig(max_concurrent=1, max_reflections=1),
                            sandbox=CounterfactualSandbox() if cf else None)
            wins[cf] += sch.run().outcome is Outcome.CONSENSUS
    assert wins[True] > wins[False]
