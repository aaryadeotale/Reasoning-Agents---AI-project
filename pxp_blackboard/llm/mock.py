"""Scripted 'models' that speak the real JSON protocol, for tests and offline demos.

Each scenario is a small story: who the agents are, what the author proposes, what the
skeptic objects to, and how the author revises. Roles: analyst (author), skeptic, auditor.
"""
from __future__ import annotations

import itertools
import json

from ..prompts import extract_cf_state, extract_state

BAD_FIRST_ATTEMPTS = [
    "Sure! I think that looks fine to me.",                          # no JSON at all
    '{"tag": "MAYBE", "prediction": null, "explanation": null}',     # illegal tag
    '```json\n{"tag": "RATIFY", "prediction": null\n```',            # truncated JSON
]

# Simulated "thinking" time per role, in seconds at speed 1x (jittered by +-JITTER).
ROLE_LATENCY = {"analyst": 1.9, "skeptic": 2.5, "auditor": 1.6}
JITTER = 0.4

SCENARIOS: dict[str, dict] = {
    "easy": {
        "title": "Easy consensus",
        "blurb": "Everyone agrees straight away. Strong intelligibility, no objections.",
        "problem": "Is 91 a prime number?",
        "agents": [("Analyst", "data_analyst", "analyst", 5), ("Skeptic", "skeptic", "skeptic", 1),
                   ("Auditor", "generalist", "auditor", 1)],
        "propose": ("No, 91 is not prime", "91 = 7 x 13, so it has divisors other than 1 and itself"),
        "skeptic": "ratify", "auditor": "ratify",
    },
    "revenue": {
        "title": "Revenue question (Ultra-Strong)",
        "blurb": "Skeptic rejects, analyst revises, panel ratifies: a productive REVISE loop.",
        "problem": "Did revenue fall in Q3?",
        "agents": [("Analyst", "data_analyst", "analyst", 5), ("Skeptic", "skeptic", "skeptic", 1),
                   ("Auditor", "generalist", "auditor", 1)],
        "propose": ("Revenue fell 4%", "Q3 total revenue was below the Q2 total"),
        "objection": "The explanation ignores the one-off asset sale",
        "marker": "excluding",
        "revised": ("Revenue fell 4%", "Q3 total was below Q2, excluding the one-off asset sale"),
        "skeptic": "reject_until_fixed", "auditor": "ratify",
    },
    "diagnosis": {
        "title": "Clinical diagnosis (Ultra-Strong)",
        "blurb": "Clinician proposes, skeptic and pharmacologist object, evidence is added.",
        "problem": "A 54-year-old has crushing chest pain radiating to the left arm. Most likely diagnosis?",
        "agents": [("Clinician", "clinician", "analyst", 5), ("Skeptic", "skeptic", "skeptic", 1),
                   ("Pharmacist", "pharmacologist", "auditor", 1)],
        "propose": ("Acute myocardial infarction", "Classic presentation of crushing chest pain radiating to the arm"),
        "objection": "No objective evidence is cited, an ECG and troponin are needed to rule out other causes",
        "marker": "troponin",
        "revised": ("Acute myocardial infarction (STEMI)",
                    "Pain pattern plus ST elevation on ECG and raised troponin supports STEMI over angina or aortic dissection"),
        "refute": "I cannot confirm this without lab evidence and the aspirin/anticoagulant history",
        "skeptic": "reject_until_fixed", "auditor": "refute_until_fixed",
    },
    "deadlock": {
        "title": "Deadlock (impasse alert)",
        "blurb": "Two reviewers never accept any revision: the negative streak hits the limit.",
        "problem": "Should the company restate last year's earnings?",
        "agents": [("Analyst", "data_analyst", "analyst", 5), ("Skeptic", "skeptic", "skeptic", 1),
                   ("Auditor", "generalist", "auditor", 1)],
        "propose": ("Yes, restate the earnings", "The revenue recognition error is material to the annual total"),
        "objection": "The error is immaterial and a restatement would damage trust for no gain",
        "refute": "Materiality cannot be established from the figures provided",
        "revisions": [
            ("Yes, restate the earnings", "The error exceeds the 5% materiality threshold once deferred revenue is included"),
            ("Restate only the affected quarters", "A partial restatement limits disruption while correcting the material error"),
            ("Restate with a prominent disclosure", "Full restatement plus disclosure addresses regulators and investors together"),
        ],
        "skeptic": "reject_always", "auditor": "refute_always",
    },
    "counterfactual": {
        "title": "Counterfactual rescue (Milestone 3)",
        "blurb": "Each revision answers only the latest objection, so the panel loops into a deadlock. "
                 "The counterfactual Analyst replays history, finds an explanation that answers both, and breaks it.",
        "problem": "Should the company restate last year's earnings?",
        "agents": [("Analyst", "data_analyst", "analyst", 5, True), ("Skeptic", "skeptic", "skeptic", 1),
                   ("Auditor", "generalist", "auditor", 1)],
        "propose": ("Yes, restate the earnings", "The revenue recognition error is material to the annual total"),
        "objection": "Materiality is not shown: the deferred revenue effect is missing",
        "refute": "I cannot sign off without a disclosure plan for investors",
        "markers": ["deferred revenue", "disclosure"],
        "revisions": [  # myopic: each revision answers only the most recent objection
            ("Yes, restate the earnings", "With a disclosure note to investors, the error is material"),
            ("Yes, restate the earnings", "Including deferred revenue, the error exceeds the 5% materiality threshold"),
        ],
        "cf_alternatives": [
            ("Yes, restate the earnings",
             "Including deferred revenue the error exceeds the 5% threshold, and a disclosure note explains the restatement to investors"),
            ("Restate only the affected quarters", "A partial restatement limits disruption"),
        ],
        "skeptic": "reject_until_all", "auditor": "refute_until_all",
    },
}


def _reply(tag, prediction=None, explanation=None) -> str:
    return json.dumps({"tag": tag, "prediction": prediction, "explanation": explanation})


def demo_responder(role: str, flaky: bool = False, scenario: str = "revenue"):
    """role: analyst | skeptic | auditor. flaky => every other first attempt is malformed."""
    sc = SCENARIOS[scenario]
    counter = itertools.count()
    revisions = itertools.count()

    def fixed(head) -> bool:
        text = head["explanation"] or ""
        if "markers" in sc:
            return all(mk in text for mk in sc["markers"])
        return sc.get("marker") is not None and sc["marker"] in text

    def respond(messages) -> str:
        cf = extract_cf_state(messages)
        if cf is not None:  # counterfactual reflection prompt
            alts = [{"tag": "REVISE" if "REVISE" in cf["allowed_tags"] else "PROPOSE", "prediction": p, "explanation": e}
                    for p, e in sc.get("cf_alternatives", [])]
            return json.dumps({"alternatives": alts[: cf["k"]]})
        if flaky and len(messages) == 2:  # first attempt of a decision
            n = next(counter)
            if n % 2 == 0:
                return BAD_FIRST_ATTEMPTS[(n // 2) % len(BAD_FIRST_ATTEMPTS)]
        st = extract_state(messages)
        head, allowed = st["head"], st["allowed_tags"]
        if head is None:
            return _reply("PROPOSE", *sc["propose"])
        if role == "analyst" and "REVISE" in allowed:
            if "revisions" in sc:  # a fresh (never accepted) revision each time
                return _reply("REVISE", *sc["revisions"][next(revisions) % len(sc["revisions"])])
            return _reply("REVISE", *sc["revised"])
        if role == "skeptic":
            mode = sc["skeptic"]
            if mode == "reject_always" or (mode in ("reject_until_fixed", "reject_until_all") and not fixed(head)):
                return _reply("REJECT", None, sc["objection"])
        if role == "auditor":
            mode = sc["auditor"]
            if mode == "refute_always" or (mode in ("refute_until_fixed", "refute_until_all") and not fixed(head)):
                return _reply("REFUTE", None, sc["refute"])
        return _reply("RATIFY")

    return respond


def role_for_persona(persona_id: str) -> str:
    return {"data_analyst": "analyst", "clinician": "analyst", "skeptic": "skeptic"}.get(persona_id, "auditor")
