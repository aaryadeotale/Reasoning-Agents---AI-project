"""LLM-backed agent: prompt -> model -> parse -> (repair loop) -> Draft."""
from __future__ import annotations

import dataclasses
import json
from typing import Optional

from .ledger import TokenLedger
from .llm.backends import LLMBackend
from .models import Draft
from .parsing import FormatError, extract_json, parse_response
from .prompts import Persona, allowed_tags, build_counterfactual_messages, build_messages


class LLMAgent:
    def __init__(self, agent_id: str, persona: Persona, backend: LLMBackend, *, priority: float = 1.0,
                 ledger: Optional[TokenLedger] = None, max_attempts: int = 3, max_tokens: int = 512,
                 temperature: float = 0.2, history_window: int = 8, counterfactual: bool = False):
        self.agent_id, self.persona, self.backend = agent_id, persona, backend
        self.priority, self.ledger = priority, ledger or TokenLedger()
        self.max_attempts, self.max_tokens = max_attempts, max_tokens
        self.temperature, self.history_window = temperature, history_window
        self.counterfactual = counterfactual
        self.lessons: list[str] = []   # appended to the system prompt after a reflection

    # ---- Milestone 3 hooks -------------------------------------------------
    def fork(self) -> "LLMAgent":
        """Sandbox copy: same model and persona, separate ledger, same lessons."""
        twin = LLMAgent(self.agent_id, self.persona, self.backend, priority=self.priority,
                        ledger=TokenLedger(), max_attempts=self.max_attempts, max_tokens=self.max_tokens,
                        temperature=self.temperature, history_window=self.history_window)
        twin.lessons = list(self.lessons)
        return twin

    def learn(self, alternative: dict) -> None:
        if alternative.get("explanation"):
            lesson = (f"A counterfactual replay showed the panel accepts this reasoning: "
                      f"{alternative['explanation']!r}. Keep every point it covers when you revise.")
            if lesson not in self.lessons:
                self.lessons.append(lesson)

    def propose_alternatives(self, snapshot: dict, own_msg: dict, objections: list, k: int):
        """Ask the model what it could have posted instead of `own_msg`. Returns (alternatives, tokens)."""
        messages = build_counterfactual_messages(snapshot, self.agent_id, self.persona, own_msg, objections, k)
        resp = self.backend.generate(messages, max_tokens=self.max_tokens, temperature=self.temperature)
        self.ledger.record(self.agent_id, resp)
        tokens = resp.prompt_tokens + resp.completion_tokens
        try:
            data = extract_json(resp.text, key="alternatives")
            alts = data.get("alternatives", []) if isinstance(data, dict) else []
        except (FormatError, ValueError, json.JSONDecodeError):
            return [], tokens
        out = []
        for a in alts:
            if isinstance(a, dict) and a.get("tag") in {"PROPOSE", "REVISE", "RATIFY", "REJECT", "REFUTE"}:
                out.append({"tag": a["tag"], "prediction": a.get("prediction") or None,
                            "explanation": a.get("explanation") or None})
        return out[:k], tokens

    def act(self, snapshot: dict) -> Optional[Draft]:
        allowed = allowed_tags(snapshot, self.agent_id)
        if not allowed:
            return None
        messages = build_messages(snapshot, self.agent_id, self.persona, history_window=self.history_window)
        if self.lessons:
            messages[0] = {"role": "system", "content": messages[0]["content"] + "\n\nLESSONS:\n- "
                           + "\n- ".join(self.lessons)}
        ptok = ctok = 0
        latency = 0.0
        for attempt in range(1, self.max_attempts + 1):
            resp = self.backend.generate(messages, max_tokens=self.max_tokens, temperature=self.temperature)
            self.ledger.record(self.agent_id, resp)  # failed attempts cost tokens too
            ptok, ctok, latency = ptok + resp.prompt_tokens, ctok + resp.completion_tokens, latency + resp.latency_s
            try:
                draft = parse_response(resp.text, agent_id=self.agent_id, snapshot=snapshot,
                                       allowed=allowed, persona=self.persona)
            except FormatError as e:
                messages = messages + [
                    {"role": "assistant", "content": resp.text},
                    {"role": "user", "content": f"Your reply was invalid: {e}. "
                                                f"Reply again with ONLY the corrected JSON object."}]
                continue
            self.ledger.note_outcome(self.agent_id, attempt, True)
            meta = {"persona": self.persona.id, "model": resp.model, "attempts": attempt,
                    "prompt_tokens": ptok, "completion_tokens": ctok, "latency_s": round(latency, 4)}
            return dataclasses.replace(draft, meta=meta)
        self.ledger.note_outcome(self.agent_id, self.max_attempts, False)
        return None  # give up on this board version; scheduler moves on
