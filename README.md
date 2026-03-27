---
title: SupportEnv
emoji: 🎫
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
tags:
  - openenv
  - customer-support
  - nlp
  - rl-environment
  - agent-evaluation
pinned: false
---

# SupportEnv — AI Customer Support Triage Environment

An **OpenEnv-compliant** reinforcement-learning environment where agents learn to resolve real-world SaaS customer support tickets.

## Why This Domain?

Every software company processes thousands of support tickets daily. Getting the right ticket to the right team with the right priority — and generating quality, empathetic responses — directly impacts revenue and customer satisfaction. SupportEnv provides a realistic, grounded benchmark for evaluating and training support-automation agents.

---

## Environment Overview

**Domain**: SaaS customer support triage
**Ticket types**: billing disputes, technical failures, account changes, feature requests, compliance requests
**Dataset**: 15 pre-defined tickets (5 per task) with fully specified ground truth — entirely deterministic, no LLM judge needed

---

## Observation Space

```
Observation
├── task_id           str         "task1" | "task2" | "task3"
├── task_description  str         Human-readable task description
├── episode_id        str         UUID, used in subsequent /step calls
├── ticket
│   ├── ticket_id         str
│   ├── subject           str
│   ├── body              str
│   ├── customer_tier     str     "free" | "pro" | "enterprise"
│   ├── account_age_days  int
│   ├── previous_tickets  int
│   └── attachments       list[str]
├── thread_history    list[dict]  Prior agent actions this episode
├── available_actions list[str]   Valid action_type values right now
├── step_number       int
├── max_steps         int
└── hint              str | null  Optional step-level guidance
```

---

## Action Space

All actions share `action_type` (required). Only fields relevant to the chosen type need to be populated.

| action_type | Key fields | Used in |
|-------------|-----------|---------|
| `classify` | `category`, `priority` | Task 1 |
| `extract` | `extracted_entities` (dict), `required_actions` (list) | Task 2 |
| `respond` | `response_text`, `resolution_steps` | Task 3 |
| `resolve` | `response_text`, `resolution_steps` | Task 3 |
| `escalate` | `escalation_team`, `escalation_reason` | Task 3 |
| `submit` | — | All tasks |

**Category options**: `billing` · `technical` · `account` · `feature_request` · `complaint` · `general`
**Priority options**: `low` · `medium` · `high` · `critical`

---

## Tasks

### Task 1 — Ticket Classification (Easy)
**max_steps**: 3
**Objective**: Assign the correct `category` and `priority` to the ticket.

| Criterion | Weight |
|-----------|--------|
| Category correct | 0.50 |
| Priority correct | 0.40 |
| Efficiency (≤ 2 classify attempts) | 0.10 |

Partial credit is awarded for priorities that are one level off, and for category substring matches.

---

### Task 2 — Information Extraction (Medium)
**max_steps**: 5
**Objective**: Extract all key entities (IDs, names, amounts, dates, domains) and identify the complete list of required actions.

| Criterion | Weight |
|-----------|--------|
| Entity coverage | 0.60 |
| Action coverage | 0.30 |
| No hallucination (≤ 2 extra keys) | 0.10 |

Partial credit for values that substring-match or partially overlap the ground truth.

---

### Task 3 — Resolution Generation (Hard)
**max_steps**: 8
**Objective**: Write a professional customer-facing response and an ordered list of resolution steps that fully address the ticket.

| Criterion | Weight |
|-----------|--------|
| Required keyword coverage | 0.30 |
| Resolution step coverage | 0.30 |
| Tone compliance (apology / urgency / timeline) | 0.25 |
| Response length ≥ minimum word count | 0.10 |
| No empty/trivial steps | 0.05 |

---

## Reward Function

The reward function provides **dense** signal throughout the episode:

```
step_cost        = -0.02    per step (encourages efficiency)
submit_bonus     = +0.05    for explicitly calling submit
max_step_penalty = -0.10    if episode ends by exhausting max_steps
grader_bonus     = grader_score × 1.0  (0.0–1.0, added at episode end)
```

This gives agents useful gradient signal even before the episode ends.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Start a new episode |
| `POST` | `/step` | Submit an action |
| `GET` | `/state` | Get current episode state |
| `GET` | `/tasks` | List all tasks + action schema |
| `POST` | `/grader` | Grade a finished episode |
| `POST` | `/baseline` | Run the heuristic baseline |
| `GET` | `/health` | Liveness check |
| `GET` | `/docs` | Interactive Swagger UI |

### Quick example

```python
import httpx

BASE = "http://localhost:7860"

# 1. Start episode
obs = httpx.post(f"{BASE}/reset", json={"task_id": "task1"}).json()
episode_id = obs["episode_id"]

# 2. Classify the ticket
result = httpx.post(f"{BASE}/step", json={
    "episode_id": episode_id,
    "action": {
        "action_type": "classify",
        "category": "billing",
        "priority": "high"
    }
}).json()

# 3. Submit
result = httpx.post(f"{BASE}/step", json={
    "episode_id": episode_id,
    "action": {"action_type": "submit"}
}).json()

# 4. Get grader score
score = httpx.post(f"{BASE}/grader", json={"episode_id": episode_id}).json()
print(score["score"])  # e.g. 0.9
```

---

## Setup & Installation

### Local (Python)

```bash
git clone <repo-url>
cd supportenv
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t supportenv .
docker run -p 7860:7860 supportenv
```

The server will be available at `http://localhost:7860`.
Interactive docs at `http://localhost:7860/docs`.

---

## Baseline Script

### Heuristic baseline (no API key required)

```bash
# Single ticket
python baseline.py --mode heuristic

# All tickets, averaged
python baseline.py --mode heuristic --all-tickets
```

### LLM baseline (requires OpenAI API key)

```bash
export OPENAI_API_KEY="sk-..."
python baseline.py --mode llm --model gpt-4o-mini
python baseline.py --mode llm --model gpt-4o-mini --all-tickets
```

---

## Baseline Scores (Heuristic — fully reproducible, no API key)

### Single ticket (ticket_index=0)

| Task | Score |
|------|-------|
| task1 — Classification (easy) | 1.0000 |
| task2 — Extraction (medium) | 0.8000 |
| task3 — Generation (hard) | 1.0000 |
| **Average** | **0.9333** |

### Average across all 5 tickets per task

| Task | Average Score | Score Range |
|------|---------------|-------------|
| task1 — Classification (easy) | 0.8600 | 0.30 – 1.00 |
| task2 — Extraction (medium) | 0.5614 | 0.39 – 0.80 |
| task3 — Generation (hard) | 0.9895 | 0.96 – 1.00 |
| **Overall average** | **0.8036** | |

The lower task2 score (0.56 average) demonstrates that the extraction task genuinely challenges pattern-based agents — realistic LLMs that understand context score significantly higher.

> Scores are fully reproducible — run `python baseline.py --all-tickets` to verify.

---

## Project Structure

```
supportenv/
├── app.py            FastAPI server (all endpoints)
├── environment.py    Episode lifecycle: reset / step / state / grade
├── graders.py        Deterministic graders for all 3 tasks
├── data.py           15 pre-defined tickets + ground truth
├── models.py         Pydantic typed models
├── baseline.py       Heuristic + LLM baseline inference scripts
├── openenv.yaml      OpenEnv spec metadata
├── Dockerfile        Container (Hugging Face Spaces compatible)
├── requirements.txt
└── README.md
```

---

## Evaluation & Reproducibility

- All 15 tickets are **static** — no randomisation between runs
- Graders are **purely rule-based** — no LLM judge, identical results every run
- The heuristic baseline requires **no external API** — scores are fully reproducible

---

## License

MIT