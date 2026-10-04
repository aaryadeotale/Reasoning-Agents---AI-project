# Milestone 1 — PXP Blackboard and 4R Event-Driven Scheduler

## 1. Purpose

Milestone 1 implements the **central blackboard and event-driven control layer** of the project.

The implementation is intentionally limited to the four PXP action tags specified in the project proposal:

- **RATIFY** — accept the current hypothesis and explanation.
- **REVISE** — replace the current hypothesis/explanation with an updated version.
- **REFUTE** — challenge the current hypothesis with a reasoned objection.
- **REJECT** — reject the current hypothesis/explanation with a reasoned objection.

The project proposal describes Milestone 1 as initializing the central blackboard repository and verifying that the event-driven scheduler can process basic `RATIFY` and `REJECT` state changes. The code below implements that core functionality, together with validation, concurrency protection, persistence, scheduling, and terminal-state detection.

---

## 2. What the system does

The system maintains one shared **blackboard** containing the current problem, the current hypothesis, agent stances, and a history of agent actions.

Agents do not directly communicate with each other. Instead:

```text
                    ┌─────────────────┐
                    │     Problem     │
                    └────────┬────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │  Initial Hypothesis   │
                 │ prediction + reason   │
                 └───────────┬───────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │   Blackboard   │
                    │  shared state  │
                    └───────┬────────┘
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
          Agent A        Agent B        Agent C
             │              │              │
             └─────── 4R actions ──────────┘
                            │
                            ▼
                    ┌────────────────┐
                    │    Scheduler   │
                    │ priority/event │
                    │     control    │
                    └───────┬────────┘
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
             Consensus              Impasse /
                                   Max limit /
                                   No agents
```

Every agent action is validated before it can modify the blackboard.

The four tags required by this Milestone 1 implementation are exactly:

```text
RATIFY
REVISE
REFUTE
REJECT
```

An agent must act on an existing hypothesis. Therefore, the blackboard constructor receives an **initial system-seeded hypothesis** when the board is created:

```python
board = Blackboard(
    "Did revenue fall in Q3?",
    ["Analyst", "Skeptic", "Auditor"],
    initial_prediction="Revenue fell 4%",
    initial_explanation="Q3 total was below Q2",
    initial_agent_id="Analyst",
)
```

Internally the seed has sequence number `0`. The first actual agent action has sequence number `1` and targets `0`.

For example:

```text
Initial hypothesis (seq 0)
        │
        ├── REJECT (seq 1, target 0)
        │
        └── REVISE (seq 2, target 0)
                    │
                    ▼
             New hypothesis
               (head = 2)
                    │
             RATIFY/REJECT/etc.
             must target 2
```

This target mechanism prevents agents from acting on an outdated hypothesis.

---

# 3. The four PXP tags

## RATIFY

An agent agrees with the current hypothesis and explanation.

Example:

```python
Draft("B", Tag.RATIFY, 2)
```

A RATIFY:

- must specify the current `target_id`;
- cannot contain a new prediction;
- cannot contain a new explanation;
- cannot be issued by the current hypothesis author.

A RATIFY resets the consecutive negative-action streak.

---

## REVISE

An agent changes the current hypothesis.

Example:

```python
Draft(
    "A",
    Tag.REVISE,
    0,
    "Revenue fell 4%",
    "Q3 total was below Q2 after accounting for the one-off sale",
)
```

A REVISE:

- must target the current hypothesis;
- must contain a non-empty prediction;
- must contain a non-empty explanation;
- cannot be identical to the current prediction and explanation.

After a successful REVISE, that message becomes the new **blackboard head**.

Its sequence number becomes the target that subsequent actions must use.

---

## REFUTE

An agent challenges the current hypothesis and provides an explanation for the challenge.

Example:

```python
Draft(
    "C",
    Tag.REFUTE,
    2,
    None,
    "The calculation ignores the one-off asset sale",
)
```

A REFUTE:

- must target the current hypothesis;
- must contain an explanation;
- cannot contain a replacement prediction;
- cannot be issued by the current hypothesis author.

REFUTE contributes to the negative-action streak.

---

## REJECT

An agent rejects the current hypothesis/explanation and gives a reason.

Example:

```python
Draft(
    "B",
    Tag.REJECT,
    2,
    None,
    "The explanation does not account for the one-off asset sale",
)
```

A REJECT:

- must target the current hypothesis;
- must contain an explanation;
- cannot be issued by the current hypothesis author.

REJECT contributes to the negative-action streak.

---

# 4. File-by-file explanation

## `models.py`

Defines the data structures used throughout the project.

### `Tag`

The protocol enum containing exactly four values:

```python
Tag.RATIFY
Tag.REVISE
Tag.REFUTE
Tag.REJECT
```

### `Draft`

Represents an action **before** it is accepted by the blackboard.

Important fields:

```text
agent_id
 tag
target_id
prediction
explanation
meta
```

A Draft is what an agent wants to submit.

### `Message`

Represents an action **after** it has been accepted by the blackboard.

It receives a sequence number and timestamp and becomes part of the immutable action history.

---

# 5. `validator.py`

This is the protocol/state-machine validation layer.

Before any action reaches the blackboard state, `validate()` checks it.

It verifies:

1. The agent is registered.
2. The tag is one of the four valid tags.
3. A current hypothesis exists.
4. A `target_id` is supplied.
5. The target matches the current blackboard head.
6. REVISE contains a prediction and explanation.
7. REVISE actually changes something.
8. RATIFY contains no new prediction/explanation.
9. REFUTE contains an explanation but no replacement prediction.
10. REJECT contains an explanation.
11. The current hypothesis author cannot RATIFY/REFUTE/REJECT their own hypothesis.

If validation fails, the action is not written to the board.

The main errors are:

```text
ProtocolError       → malformed/invalid action
StaleTargetError    → action targets an old hypothesis
BoardClosedError    → board cannot accept more actions
```

---

# 6. `blackboard.py`

This is the **central shared state store**.

It is thread-safe because all state-changing and state-reading operations are protected using:

```python
threading.RLock()
```

The blackboard stores:

```text
problem
registered agents
initial hypothesis
current head
agent stances
negative-action streak
message history
closed/open state
subscribers
```

### `post()`

The most important function.

The flow is:

```text
Agent creates Draft
       ↓
Blackboard.post()
       ↓
Check board is open
       ↓
Validate Draft
       ↓
Assign sequence number
       ↓
Create Message
       ↓
Append to history
       ↓
Update blackboard state
       ↓
Notify subscribers
```

### Head

The `head` represents the hypothesis currently being evaluated.

Initially:

```text
head.seq = 0
```

If an agent performs REVISE at target `0`, the new message becomes the head:

```text
head.seq = 1
```

All future actions must therefore target `1`.

This is how the implementation prevents stale concurrent updates.

---

# 7. Thread safety

The blackboard uses a re-entrant lock so multiple threads can safely attempt to write at the same time.

For example, suppose ten agents simultaneously attempt:

```text
REVISE → target 0
```

Only one can acquire the valid current state and successfully revise it.

After the first REVISE succeeds, the head changes from:

```text
0 → 1
```

The remaining actions still targeting `0` are rejected as stale.

Therefore:

```text
1 successful revision
9 stale revisions
```

The same locking mechanism ensures that concurrent RATIFY actions receive unique, ordered sequence numbers.

---

# 8. Consensus detection

The board reaches consensus when every registered agent has RATIFY status for the current hypothesis, except the agent who authored that current hypothesis.

Example:

```text
Current hypothesis: seq 2
Author: Analyst

Analyst   → author of hypothesis
Skeptic   → RATIFY
Auditor   → RATIFY

             ↓
          CONSENSUS
```

A REJECT or REFUTE prevents consensus until the state changes and the relevant agents RATIFY the current hypothesis.

---

# 9. Negative-action streak and impasse

The blackboard maintains a counter called:

```python
negative_streak
```

It increases for:

```text
REFUTE
REJECT
```

It resets when a RATIFY is received.

A REVISE does not reset the negative streak because a revision alone does not demonstrate agreement.

The scheduler uses this counter to detect a possible deadlock/impasse.

For example, with:

```python
impasse_threshold = 4
```

four consecutive REFUTE/REJECT actions can terminate the run as an impasse.

---

# 10. `agents.py`

This file defines the agent interface and a simple scripted implementation.

## `Agent`

A protocol describing what an agent needs:

```text
agent_id
priority
act(snapshot)
```

The `act()` function receives a snapshot of the blackboard and returns either:

```text
Draft
```

or:

```text
None
```

## `ScriptedAgent`

Used for Milestone 1 testing and demonstration.

It allows us to simulate agent behaviour without requiring an LLM.

This is important: **Milestone 1 does not yet contain a real LLM.**

The later project milestone is responsible for connecting local 7B models and prompt libraries.

---

# 11. `scheduler.py`

The scheduler is the event-driven control layer.

Its job is to decide **which agent should act next** and to manage concurrent agent execution.

The scheduler uses:

```python
ThreadPoolExecutor
```

so multiple agents can execute concurrently when `max_concurrent > 1`.

### Scheduler flow

```text
Read blackboard snapshot
        ↓
Calculate eligible agents
        ↓
Apply priority policy
        ↓
Select agents
        ↓
Run agent.act(snapshot)
        ↓
Receive Draft
        ↓
Submit Draft to Blackboard
        ↓
Blackboard validates it
        ↓
State changes
        ↓
Scheduler receives event
        ↓
Repeat
```

The scheduler subscribes to blackboard updates and maintains an event queue.

---

# 12. Priority policy

`default_policy()` determines whether an agent is eligible to act and assigns a priority score.

The basic rules are:

- Agents that have not responded to the current hypothesis can become eligible.
- The author of the current hypothesis can receive a priority boost when another agent has challenged it.
- Agents that have already declined for the current board version are not repeatedly dispatched for that same version.

The author boost is:

```python
AUTHOR_BOOST = 10.0
```

This allows a revised/challenged hypothesis author to respond when necessary.

---

# 13. Scheduler terminal states

The scheduler can stop for four reasons.

### `CONSENSUS`

All required agents have RATIFIED the current hypothesis.

### `IMPASSE`

The negative-action streak reaches the configured threshold.

### `MAX_ITERATIONS`

The blackboard reaches its maximum message capacity.

### `NO_ELIGIBLE_AGENTS`

No agent can currently produce another valid action.

The scheduler returns these through the `Outcome` enum.

---

# 14. Intelligibility classification

Milestone 1 includes a simple classification of the resulting interaction.

### Strong

A run reaches consensus without requiring a productive REVISE after a negative action.

Example:

```text
RATIFY
RATIFY
CONSENSUS
```

### Ultra-Strong

A run reaches consensus where a REVISE occurs after a negative action such as REJECT/REFUTE.

Example:

```text
REJECT
REVISE
RATIFY
RATIFY
CONSENSUS
```

### Failed

The run terminates without consensus.

Example:

```text
REJECT
REFUTE
REJECT
REFUTE
IMPASSE
```

This classification is a **Milestone 1 implementation metric**, not an LLM explanation-quality evaluator.

---

# 15. Persistence

The blackboard can be saved to JSON:

```python
board.save("board.json")
```

and reconstructed with:

```python
board2 = Blackboard.load("board.json")
```

The saved state contains the problem, agents, initial hypothesis, current head, stances, negative streak, and action history.

This allows the blackboard state to survive outside the running Python process.

---

# 16. `pxp_blackboard.py`

This is the public API/entry point.

It re-exports the important classes so users and tests can simply write:

```python
from pxp_blackboard import (
    Blackboard,
    Draft,
    Tag,
    Scheduler,
    ScriptedAgent,
    Outcome,
)
```

This file also fixes the package/import problem encountered when running pytest directly from the project directory.

---

# 17. `demo.py`

The demo provides a complete Milestone 1 example.

It creates three scripted agents:

```text
Analyst
Skeptic
Auditor
```

The intended interaction is:

```text
Initial hypothesis
       ↓
Skeptic → REJECT
       ↓
Analyst → REVISE
       ↓
Skeptic → RATIFY
       ↓
Auditor → RATIFY
       ↓
CONSENSUS
```

The exact execution order is controlled by the scheduler.

Run it with:

```powershell
python demo.py
```

---

# 18. Tests

The project contains tests for:

### Protocol/state-machine behaviour

- exactly four tags exist;
- malformed actions are rejected;
- RATIFY and REJECT transitions work;
- stale targets are rejected;
- invalid self-actions are rejected.

### Blackboard behaviour

- message limits;
- persistence round-trip;
- consensus detection;
- negative streak handling.

### Thread safety

- concurrent RATIFY operations;
- competing concurrent REVISE operations;
- stale revision detection.

### Scheduler behaviour

- straight consensus;
- REJECT → REVISE → RATIFY flow;
- impasse detection;
- maximum iteration termination;
- no eligible agents;
- misbehaving agents being benched;
- concurrent scheduler execution.

Run the complete test suite with:

```powershell
python -m pytest -q
```

Expected result for the supplied Milestone 1 tests:

```text
14 passed
```

---

# 19. What is and is not implemented in Milestone 1

## Implemented

- Thread-safe central blackboard
- Four-tag PXP state machine
- RATIFY
- REVISE
- REFUTE
- REJECT
- Initial system-seeded hypothesis
- Target/version validation
- Stale-update protection
- Agent registration
- Agent priority scheduling
- Event-driven scheduler
- Concurrent agent execution
- Consensus detection
- Negative-action/impasse detection
- Maximum message limit
- Persistence to JSON
- Blackboard event subscriptions
- Scripted-agent demonstration
- Automated tests

## Not implemented yet

Milestone 1 intentionally does **not** implement the later project components:

- real local 7B LLM inference;
- LLM prompt libraries;
- retrospective counterfactual reasoning;
- counterfactual credit assignment;
- MCP tool calling;
- A2A communication;
- WebSocket visualization;
- React/D3 frontend;
- benchmark ingestion for KramaBench/MSCoRe/MedAgentBench;
- token-cost analytics;
- large-scale ablation experiments.

Those belong to later stages of the overall project.

---

# 20. Minimal example

```python
from pxp_blackboard import Blackboard, Draft, Tag

board = Blackboard(
    "What is 6*7?",
    ["A", "B", "C"],
    initial_prediction="42",
    initial_explanation="6*7=42",
)

# Agent B rejects the initial hypothesis.
board.post(
    Draft(
        "B",
        Tag.REJECT,
        0,
        explanation="The proposed arithmetic is incorrect",
    )
)

# Agent A revises it.
board.post(
    Draft(
        "A",
        Tag.REVISE,
        0,
        prediction="42",
        explanation="6*7 = 6*6 + 6 = 42",
    )
)

# Other agents ratify the revised hypothesis.
board.post(Draft("B", Tag.RATIFY, 2))
board.post(Draft("C", Tag.RATIFY, 2))

print(board.is_consensus())
# True
```

The key concept is that **the blackboard is the single shared source of truth**. Agents generate actions, the validator decides whether those actions are legal, the blackboard commits valid actions, and the scheduler controls which agents get to act next.
