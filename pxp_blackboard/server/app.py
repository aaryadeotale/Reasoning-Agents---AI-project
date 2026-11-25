"""FastAPI app: REST to start/inspect sessions, WebSocket to stream them live."""
from __future__ import annotations

import asyncio
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from ..llm.mock import SCENARIOS
from ..prompts import PromptLibrary
from .hub import TERMINAL
from .sessions import Session, SessionRequest

STATIC = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(title="PXP Blackboard")
    # lets index.html work even when opened straight from disk (file:// pages send Origin: null)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    library = PromptLibrary.load()
    sessions: dict[str, Session] = {}
    lock = threading.Lock()

    def get(sid: str) -> Session:
        with lock:
            if sid not in sessions:
                raise HTTPException(404, "unknown session")
            return sessions[sid]

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    def health():
        return {"ok": True, "personas": library.ids()}

    @app.get("/api/scenarios")
    def scenarios():
        return [{"id": k, "title": v["title"], "blurb": v["blurb"], "problem": v["problem"],
                 "agents": [{"id": a[0], "persona": a[1], "role": a[2], "priority": a[3]} for a in v["agents"]]}
                for k, v in SCENARIOS.items()]

    @app.post("/api/sessions")
    def start(req: SessionRequest):
        try:
            s = Session(req, library)
        except (KeyError, ValueError) as e:
            raise HTTPException(400, str(e.args[0] if e.args else e))
        with lock:
            sessions[s.id] = s
        s.start()
        return {"session_id": s.id, "ws_url": f"/ws/{s.id}"}

    @app.get("/api/sessions")
    def list_sessions():
        with lock:
            return [s.summary() for s in sessions.values()]

    @app.get("/api/sessions/{sid}")
    def status(sid: str):
        return get(sid).summary()

    @app.get("/api/sessions/{sid}/events")
    def events(sid: str):
        return get(sid).hub.events()

    @app.websocket("/ws/{sid}")
    async def stream(ws: WebSocket, sid: str):
        await ws.accept()
        with lock:
            session = sessions.get(sid)
        if session is None:
            await ws.close(code=4404)
            return
        sub, backlog = session.hub.subscribe(asyncio.get_running_loop())
        try:
            for ev in backlog:
                await ws.send_json(ev)
            done = any(ev["type"] in TERMINAL for ev in backlog)
            while not done:
                ev = await sub.queue.get()
                await ws.send_json(ev)
                done = ev["type"] in TERMINAL
            await ws.close()
        except WebSocketDisconnect:
            pass
        finally:
            session.hub.unsubscribe(sub)

    return app


app = create_app()
