import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

from pxp_blackboard import (Blackboard, Draft, FormatError, LLMAgent, Outcome, PromptLibrary, Scheduler,
                            SchedulerConfig, Tag, TokenLedger, parse_response)
from pxp_blackboard.llm import MockBackend, OpenAICompatBackend
from pxp_blackboard.llm.mock import demo_responder
from pxp_blackboard.prompts import allowed_tags, build_messages, extract_state
from pxp_blackboard.server.app import create_app

LIB = PromptLibrary.load()
GEN = LIB.get("generalist")


def snap_with_head():
    b = Blackboard("What is 6*7?", ["A", "B", "C"])
    b.post(Draft("A", Tag.PROPOSE, None, "42", "six times seven equals forty two"))
    return b.snapshot()


def parse(text, agent="B", snap=None):
    snap = snap or snap_with_head()
    return parse_response(text, agent_id=agent, snapshot=snap,
                          allowed=allowed_tags(snap, agent), persona=GEN)


# ---------------- prompt library & builder ----------------
def test_library_has_domain_personas_with_constraints():
    assert {"clinician", "data_analyst", "skeptic", "pharmacologist"} <= set(LIB.ids())
    for pid in LIB.ids():
        p = LIB.get(pid)
        assert p.system and p.explanation_min_words < p.explanation_max_words
    with pytest.raises(KeyError):
        LIB.get("nope")


def test_allowed_tags_follow_validator_rules():
    empty = Blackboard("q", ["A", "B"]).snapshot()
    assert allowed_tags(empty, "A") == [Tag.PROPOSE]
    s = snap_with_head()
    assert allowed_tags(s, "A") == [Tag.REVISE]                      # author
    assert Tag.RATIFY in allowed_tags(s, "B") and Tag.PROPOSE not in allowed_tags(s, "B")


def test_prompt_contains_problem_schema_and_state():
    msgs = build_messages(snap_with_head(), "B", LIB.get("skeptic"))
    assert msgs[0]["role"] == "system" and "critical" in msgs[0]["content"]
    assert "What is 6*7?" in msgs[1]["content"] and "ALLOWED TAGS NOW" in msgs[1]["content"]
    st = extract_state(msgs)
    assert st["head"]["seq"] == 1 and "REJECT" in st["allowed_tags"]


# ---------------- parser ----------------
def test_parser_accepts_fences_chatter_and_case():
    d = parse('Here you go:\n```json\n{"tag": "reject", "prediction": null, '
              '"explanation": "that arithmetic looks wrong to me"}\n```')
    assert d.tag is Tag.REJECT and d.target_id == 1 and d.agent_id == "B"


def test_parser_strips_content_from_ratify_and_refute():
    d = parse('{"tag":"RATIFY","prediction":"42","explanation":"I agree with everything"}')
    assert d.prediction is None and d.explanation is None
    d = parse('{"tag":"REFUTE","prediction":"41","explanation":"I cannot verify this claim"}')
    assert d.prediction is None and d.explanation


@pytest.mark.parametrize("text", [
    "no json here",
    '{"tag": "MAYBE"}',
    '{"tag": "PROPOSE", "prediction": "x", "explanation": "five whole words are here"}',   # not allowed now
    '{"tag": "REJECT", "prediction": null, "explanation": "bad"}',                          # too short
    '{"tag": "REVISE", "prediction": "42", "explanation": "six times seven equals forty two"}',  # no-op
    '{"foo": 1}',
])
def test_parser_rejects_bad_output(text):
    with pytest.raises(FormatError):
        parse(text)


def test_parser_truncates_overlong_explanation():
    long = " ".join(["word"] * 500)
    d = parse(json.dumps({"tag": "REJECT", "prediction": None, "explanation": long}))
    assert len(d.explanation.split()) == GEN.explanation_max_words


# ---------------- LLMAgent repair loop ----------------
def scripted_backend(replies):
    it = iter(replies)
    return MockBackend(lambda msgs: next(it))


GOOD = '{"tag":"RATIFY","prediction":null,"explanation":null}'


def test_agent_repairs_and_feeds_error_back():
    seen = []
    replies = iter(["garbage", GOOD])

    def responder(msgs):
        seen.append(msgs)
        return next(replies)
    led = TokenLedger()
    agent = LLMAgent("B", GEN, MockBackend(responder), ledger=led)
    d = agent.act(snap_with_head())
    assert d.tag is Tag.RATIFY and d.meta["attempts"] == 2
    assert "invalid" in seen[1][-1]["content"] and seen[1][-2]["content"] == "garbage"
    assert led.format_stats()["repaired"] == 1 and led.totals()["calls"] == 2
    assert d.meta["prompt_tokens"] == led.totals()["prompt_tokens"]   # failed attempts are billed


def test_agent_gives_up_after_max_attempts():
    led = TokenLedger()
    agent = LLMAgent("B", GEN, scripted_backend(["x", "y", "z"]), ledger=led, max_attempts=3)
    assert agent.act(snap_with_head()) is None
    assert led.format_stats()["failed"] == 1 and led.totals()["calls"] == 3


# ---------------- OpenAI-compatible backend over real HTTP ----------------
class _Handler(BaseHTTPRequestHandler):
    requests, reject_json_mode = [], False

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append((self.path, self.headers.get("Authorization"), body))
        if "response_format" in body and type(self).reject_json_mode:
            self.send_response(400); self.end_headers(); self.wfile.write(b"{}"); return
        out = json.dumps({"choices": [{"message": {"content": GOOD}}],
                          "usage": {"prompt_tokens": 11, "completion_tokens": 7}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    _Handler.requests, _Handler.reject_json_mode = [], False
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def test_openai_backend_request_shape_and_usage(server):
    be = OpenAICompatBackend(server, "qwen2.5:7b", api_key="k")
    r = be.generate([{"role": "user", "content": "hi"}], max_tokens=50)
    path, auth, body = _Handler.requests[0]
    assert path == "/v1/chat/completions" and auth == "Bearer k"
    assert body["model"] == "qwen2.5:7b" and body["max_tokens"] == 50 and body["response_format"]
    assert (r.text, r.prompt_tokens, r.completion_tokens) == (GOOD, 11, 7)


def test_openai_backend_falls_back_when_json_mode_unsupported(server):
    _Handler.reject_json_mode = True
    be = OpenAICompatBackend(server, "m")
    assert be.generate([{"role": "user", "content": "hi"}]).text == GOOD
    assert len(_Handler.requests) == 2 and "response_format" not in _Handler.requests[1][2]
    be.generate([{"role": "user", "content": "hi"}])                 # remembered: no more probing
    assert len(_Handler.requests) == 3 and "response_format" not in _Handler.requests[2][2]


def test_agent_works_through_http_backend(server):
    agent = LLMAgent("B", GEN, OpenAICompatBackend(server, "m"))
    assert agent.act(snap_with_head()).tag is Tag.RATIFY


# ---------------- end-to-end: scheduler + LLM agents ----------------
def panel(flaky, led):
    roles = [("Analyst", "data_analyst", "analyst", 5), ("Skeptic", "skeptic", "skeptic", 1),
             ("Auditor", "generalist", "auditor", 1)]
    return [LLMAgent(n, LIB.get(p), MockBackend(demo_responder(r, flaky)), priority=pr, ledger=led)
            for n, p, r, pr in roles]


@pytest.mark.parametrize("flaky", [False, True])
def test_end_to_end_panel_reaches_ultra_strong_consensus(flaky):
    led = TokenLedger()
    agents = panel(flaky, led)
    events = []
    board = Blackboard("Did revenue fall in Q3?", [a.agent_id for a in agents])
    r = Scheduler(board, agents, SchedulerConfig(max_concurrent=2), listener=events.append).run()
    assert r.outcome is Outcome.CONSENSUS and r.intelligibility == "Ultra-Strong"
    assert r.final_prediction == "Revenue fell 4%" and not r.protocol_errors
    fs = led.format_stats()
    assert fs["compliance_rate"] == 1.0
    assert (fs["repaired"] > 0) if flaky else (fs["first_try_rate"] == 1.0)
    assert {e["type"] for e in events} == {"agent_started", "agent_finished"}


# ---------------- server ----------------
TERMINAL = {"session_finished", "session_error"}


def collect(ws):
    out = []
    while True:
        ev = ws.receive_json()
        out.append(ev)
        if ev["type"] in TERMINAL:
            return out


def test_server_streams_full_session_over_websocket():
    with TestClient(create_app()) as c:
        assert c.get("/api/health").json()["ok"] and "PXP" in c.get("/").text
        sid = c.post("/api/sessions", json={"backend": {"type": "mock-demo", "flaky": True}}).json()["session_id"]
        with c.websocket_connect(f"/ws/{sid}") as ws:
            evs = collect(ws)
        assert evs[0]["type"] == "session_started" and evs[-1]["type"] == "session_finished"
        msgs = [e for e in evs if e["type"] == "message"]
        assert [m["message"]["seq"] for m in msgs] == list(range(1, len(msgs) + 1))
        assert [e["id"] for e in evs] == list(range(len(evs)))          # gap-free ordering
        fin = evs[-1]
        assert fin["result"]["outcome"] == "consensus" and fin["result"]["intelligibility"] == "Ultra-Strong"
        assert fin["tokens"]["total_tokens"] > 0 and fin["format"]["compliance_rate"] == 1.0
        assert any(e["type"] == "agent_started" for e in evs)
        # late joiner gets an identical replay; REST agrees
        with c.websocket_connect(f"/ws/{sid}") as ws2:
            assert [e["id"] for e in collect(ws2)] == [e["id"] for e in evs]
        assert len(c.get(f"/api/sessions/{sid}/events").json()) == len(evs)
        assert c.get(f"/api/sessions/{sid}").json()["status"] == "finished"


def test_server_validation_errors():
    with TestClient(create_app()) as c:
        assert c.post("/api/sessions", json={"agents": [{"id": "X", "persona": "nope"}, {"id": "Y"}]}).status_code == 400
        assert c.post("/api/sessions", json={"backend": {"type": "openai"}}).status_code == 400
        assert c.get("/api/sessions/missing").status_code == 404
