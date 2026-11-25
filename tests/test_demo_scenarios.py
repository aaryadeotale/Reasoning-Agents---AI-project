"""Scripted demo scenarios, latency simulation and the extra server endpoints."""
import time

import pytest
from fastapi.testclient import TestClient

from pxp_blackboard.llm.backends import MockBackend
from pxp_blackboard.llm.mock import SCENARIOS
from pxp_blackboard.prompts import PromptLibrary
from pxp_blackboard.server.app import create_app
from pxp_blackboard.server.sessions import BackendSpec, Session, SessionRequest

LIB = PromptLibrary.load()
EXPECTED = {"easy": ("consensus", "Strong"), "revenue": ("consensus", "Ultra-Strong"),
            "diagnosis": ("consensus", "Ultra-Strong"), "deadlock": ("impasse", "Failed"),
            "counterfactual": ("consensus", "Ultra-Strong")}


@pytest.mark.parametrize("flaky", [False, True])
@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_every_scenario_ends_as_scripted(name, flaky):
    s = Session(SessionRequest(scenario=name, backend=BackendSpec(flaky=flaky)), LIB).start()
    s.join(20)
    outcome, intel = EXPECTED[name]
    assert s.result.outcome.value == outcome and s.result.intelligibility == intel
    assert not s.result.protocol_errors


def test_deadlock_scenario_has_negative_streak_at_threshold():
    s = Session(SessionRequest(scenario="deadlock"), LIB).start()
    s.join(20)
    assert s.board.negative_streak >= 4


def test_mock_backend_latency_is_jittered():
    b = MockBackend(lambda m: "x", latency_s=0.05, jitter=0.5)
    times = []
    for _ in range(6):
        t = time.perf_counter()
        b.generate([{"role": "user", "content": "hi"}])
        times.append(time.perf_counter() - t)
    assert min(times) >= 0.024 and max(times) < 0.5 and len({round(x, 3) for x in times}) > 1


def test_speed_makes_session_take_real_time():
    t = time.perf_counter()
    Session(SessionRequest(scenario="easy", backend=BackendSpec(speed=20)), LIB).start().join(20)
    assert time.perf_counter() - t > 0.15  # instant (speed=0) runs take milliseconds


def test_scenarios_endpoint_and_session_started_config():
    with TestClient(create_app()) as c:
        sc = c.get("/api/scenarios").json()
        assert {s["id"] for s in sc} == set(SCENARIOS) and all(s["agents"] for s in sc)
        sid = c.post("/api/sessions", json={"scenario": "deadlock"}).json()["session_id"]
        with c.websocket_connect(f"/ws/{sid}") as ws:
            first = ws.receive_json()
        assert first["type"] == "session_started"
        assert first["scenario"]["id"] == "deadlock" and first["limits"]["impasse_threshold"] == 4
        assert first["agents"][0]["title"] and first["problem"] == SCENARIOS["deadlock"]["problem"]
        assert c.post("/api/sessions", json={"scenario": "nope"}).status_code == 400
