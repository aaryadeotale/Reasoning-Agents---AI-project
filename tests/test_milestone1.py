import threading
import time

import pytest

from pxp_blackboard import (Blackboard, BoardClosedError, Draft, Outcome, ProtocolError,
                            ScriptedAgent, Scheduler, SchedulerConfig, StaleTargetError, Tag)


def board(n=3, **kw):
    return Blackboard("What is 6*7?", [chr(65 + i) for i in range(n)], **kw)


def propose(b, aid="A", pred="42"):
    return b.post(Draft(aid, Tag.PROPOSE, None, pred, "6*7=42"))


# ---------------- validator / state machine ----------------
def test_validator_rejects_malformed_drafts():
    b = board()
    with pytest.raises(ProtocolError):
        b.post(Draft("A", Tag.RATIFY, 1))                           # no head yet
    with pytest.raises(ProtocolError):
        b.post(Draft("A", Tag.PROPOSE, None, "42", ""))             # missing explanation
    with pytest.raises(ProtocolError):
        b.post(Draft("Z", Tag.PROPOSE, None, "42", "why"))          # unknown agent
    propose(b)
    with pytest.raises(ProtocolError):
        b.post(Draft("A", Tag.RATIFY, 1))                           # self-ratify
    with pytest.raises(ProtocolError):
        b.post(Draft("B", Tag.RATIFY, 1, "43"))                     # RATIFY with content
    with pytest.raises(ProtocolError):
        b.post(Draft("B", Tag.REJECT, 1))                           # REJECT w/o explanation
    with pytest.raises(ProtocolError):
        b.post(Draft("B", Tag.REFUTE, 1, "43", "no"))               # REFUTE with prediction
    with pytest.raises(ProtocolError):
        b.post(Draft("A", Tag.REVISE, 1, "42", "6*7=42"))           # no-op revise
    with pytest.raises(StaleTargetError):
        b.post(Draft("B", Tag.RATIFY, 99))                          # stale target
    assert b.version == 1  # nothing invalid was appended


def test_ratify_and_reject_transitions():
    b = board()
    propose(b)
    b.post(Draft("B", Tag.REJECT, 1, None, "arithmetic is wrong"))
    assert b.negative_streak == 1 and not b.is_consensus()
    b.post(Draft("A", Tag.REVISE, 1, "42", "6*7 = 6*6 + 6 = 42"))   # new head, stances reset
    assert b.head.seq == 3 and b.negative_streak == 1               # REVISE is transparent
    b.post(Draft("B", Tag.RATIFY, 3))
    assert b.negative_streak == 0 and not b.is_consensus()
    b.post(Draft("C", Tag.RATIFY, 3))
    assert b.is_consensus()


def test_board_closes_at_limit():
    b = board(max_messages=1)
    propose(b)
    with pytest.raises(BoardClosedError):
        b.post(Draft("B", Tag.RATIFY, 1))


def test_persistence_roundtrip(tmp_path):
    b = board()
    propose(b)
    b.post(Draft("B", Tag.RATIFY, 1))
    b.post(Draft("C", Tag.RATIFY, 1))
    p = str(tmp_path / "board.json")
    b.save(p)
    b2 = Blackboard.load(p)
    assert b2.snapshot() == b.snapshot() and b2.is_consensus()


# ---------------- thread safety ----------------
def test_concurrent_ratifies_keep_log_consistent():
    b = Blackboard("q", [f"a{i}" for i in range(21)], max_messages=100)
    b.post(Draft("a0", Tag.PROPOSE, None, "x", "y"))
    ts = [threading.Thread(target=b.post, args=(Draft(f"a{i}", Tag.RATIFY, 1),)) for i in range(1, 21)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert [m.seq for m in b.history()] == list(range(1, 22))
    assert b.is_consensus()


def test_racing_revisions_only_one_wins():
    b = board(n=11)
    propose(b)
    wins, stale = [], []

    def go(i):
        try:
            b.post(Draft(chr(65 + i), Tag.REVISE, 1, f"p{i}", f"e{i}")); wins.append(i)
        except StaleTargetError:
            stale.append(i)
    ts = [threading.Thread(target=go, args=(i,)) for i in range(1, 11)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert len(wins) == 1 and len(stale) == 9 and b.version == 2


# ---------------- scheduler ----------------
def ratifier(aid, delay=0.0):
    def fn(s):
        time.sleep(delay)
        return Draft(aid, Tag.RATIFY, s["head"]["seq"]) if s["head"] else None
    return ScriptedAgent(aid, fn)


def proposer_revising(aid="A"):
    def fn(s):
        h = s["head"]
        if h is None:
            return Draft(aid, Tag.PROPOSE, None, "answer-1", "first attempt")
        if h["agent_id"] == aid and h["prediction"] == "answer-1":
            return Draft(aid, Tag.REVISE, h["seq"], "answer-2", "fixed after objection")
        return None
    return ScriptedAgent(aid, fn, priority=5)


def run(agents, n=None, cfg=None, **bkw):
    b = Blackboard("q", [a.agent_id for a in agents], **bkw)
    return b, Scheduler(b, agents, cfg or SchedulerConfig(max_concurrent=1)).run()


def test_straight_ratify_is_strong_consensus():
    _, r = run([proposer_revising(), ratifier("B"), ratifier("C")])
    assert r.outcome is Outcome.CONSENSUS and r.intelligibility == "Strong"
    assert r.tag_counts == {"PROPOSE": 1, "RATIFY": 2} and r.final_prediction == "answer-1"


def rejecting_once(aid):
    def fn(s):
        h = s["head"]
        if h["prediction"] == "answer-1":
            return Draft(aid, Tag.REJECT, h["seq"], None, "answer-1 is wrong")
        return Draft(aid, Tag.RATIFY, h["seq"])
    return ScriptedAgent(aid, fn)


def test_reject_then_revise_is_ultra_strong():
    _, r = run([proposer_revising(), rejecting_once("B"), ratifier("C")])
    assert r.outcome is Outcome.CONSENSUS and r.intelligibility == "Ultra-Strong"
    assert r.final_prediction == "answer-2" and r.tag_counts["REJECT"] == 1 and r.tag_counts["REVISE"] == 1


def test_impasse_on_reject_loop():
    n = [0]

    def reviser(s):
        h = s["head"]
        if h is None:
            return Draft("A", Tag.PROPOSE, None, "v0", "e")
        n[0] += 1
        return Draft("A", Tag.REVISE, h["seq"], f"v{n[0]}", "e")
    rej = ScriptedAgent("B", lambda s: Draft("B", Tag.REJECT, s["head"]["seq"], None, "no"))
    ref = ScriptedAgent("C", lambda s: Draft("C", Tag.REFUTE, s["head"]["seq"], None, "stuck"))
    _, r = run([ScriptedAgent("A", reviser, priority=5), rej, ref],
               cfg=SchedulerConfig(max_concurrent=1, impasse_threshold=4), max_messages=60)
    assert r.outcome is Outcome.IMPASSE and r.intelligibility == "Failed"


def test_max_iterations_when_agents_only_counter_revise():
    def mk(aid):
        def fn(s):
            h = s["head"]
            if h is None:
                return Draft(aid, Tag.PROPOSE, None, "v0", "e")
            if h["agent_id"] != aid:
                return Draft(aid, Tag.REVISE, h["seq"], f"v{s['version']}", "e")
        return ScriptedAgent(aid, fn)
    _, r = run([mk("A"), mk("B")], max_messages=10)
    assert r.outcome is Outcome.MAX_ITERATIONS and r.iterations == 10


def test_no_eligible_agents():
    def a(s):
        return Draft("A", Tag.PROPOSE, None, "x", "y") if s["head"] is None else None
    b = ScriptedAgent("B", lambda s: Draft("B", Tag.REFUTE, s["head"]["seq"], None, "blocked"))
    _, r = run([ScriptedAgent("A", a), b])
    assert r.outcome is Outcome.NO_ELIGIBLE_AGENTS


def test_misbehaving_agent_is_benched_not_fatal():
    bad = ScriptedAgent("B", lambda s: Draft("B", Tag.RATIFY, 999))    # always stale-free garbage? -> stale
    broken = ScriptedAgent("C", lambda s: (_ for _ in ()).throw(RuntimeError("boom")))
    _, r = run([proposer_revising(), bad, broken], cfg=SchedulerConfig(max_concurrent=1, max_protocol_errors=2))
    assert r.outcome is Outcome.NO_ELIGIBLE_AGENTS and any("boom" in e for e in r.protocol_errors)


def test_concurrent_run_reaches_consensus_with_valid_log():
    agents = [proposer_revising(), ratifier("B", 0.02), ratifier("C", 0.02), rejecting_once("D")]
    b, r = run(agents, cfg=SchedulerConfig(max_concurrent=3))
    assert r.outcome is Outcome.CONSENSUS and r.intelligibility == "Ultra-Strong"
    assert [m.seq for m in b.history()] == list(range(1, r.iterations + 1))
    assert Blackboard.from_dict(b.snapshot()).is_consensus()  # log replays cleanly
