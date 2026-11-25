"""Run one panel session from the command line.

  python scripts/run_session.py                                   # offline demo (mock model)
  python scripts/run_session.py --flaky                           # + malformed first replies (shows repair)
  python scripts/run_session.py --backend openai --base-url http://localhost:11434/v1 \
        --model qwen2.5:7b --problem "..." --agents "Doc:clinician:5,Critic:skeptic,Pharm:pharmacologist"
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pxp_blackboard.prompts import PromptLibrary
from pxp_blackboard.server.sessions import AgentSpec, BackendSpec, Session, SessionRequest

ap = argparse.ArgumentParser()
ap.add_argument("--problem", default="Did revenue fall in Q3?")
ap.add_argument("--backend", choices=["mock-demo", "openai"], default="mock-demo")
ap.add_argument("--base-url"); ap.add_argument("--model"); ap.add_argument("--api-key")
ap.add_argument("--flaky", action="store_true")
ap.add_argument("--agents", help="comma list of id:persona[:priority]")
ap.add_argument("--max-messages", type=int, default=30)
ap.add_argument("--max-concurrent", type=int, default=2)
ap.add_argument("--save", help="write the final board JSON here")
a = ap.parse_args()

agents = None
if a.agents:
    agents = []
    for item in a.agents.split(","):
        p = item.split(":")
        agents.append(AgentSpec(id=p[0], persona=p[1] if len(p) > 1 else "generalist",
                                priority=float(p[2]) if len(p) > 2 else 1.0))
req = SessionRequest(problem=a.problem, agents=agents, max_messages=a.max_messages,
                     max_concurrent=a.max_concurrent,
                     backend=BackendSpec(type=a.backend, base_url=a.base_url, model=a.model,
                                         api_key=a.api_key, flaky=a.flaky))
s = Session(req, PromptLibrary.load()).start()
s.join()
for m in s.board.history():
    print(f"#{m.seq} {m.agent_id:10} {m.tag.value:8} -> {m.target_id}  {m.prediction or ''} | {m.explanation or ''}")
fin = [e for e in s.hub.events() if e["type"] in ("session_finished", "session_error")][-1]
print(json.dumps({k: fin[k] for k in fin if k in ("result", "tokens", "format", "error")}, indent=2))
if a.save:
    s.board.save(a.save)
