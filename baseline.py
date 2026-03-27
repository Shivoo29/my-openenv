"""
SupportEnv — FastAPI server

Endpoints
---------
POST   /reset          Create a new episode
POST   /step           Advance the episode
GET    /state          Current episode state
GET    /tasks          List tasks and action schema
POST   /grader         Grade a finished episode
POST   /baseline       Run the built-in baseline agent on all tasks
GET    /health         Liveness check
GET    /               Info / spec link
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import environment as env
from data import TASK_META
from models import (
    Action,
    BaselineResult,
    GraderResponse,
    Observation,
    State,
    StepResult,
    TaskInfo,
)

app = FastAPI(
    title="SupportEnv",
    description=(
        "An OpenEnv-compliant customer-support triage environment. "
        "Agents learn to classify, extract information from, and resolve "
        "real-world SaaS support tickets."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response shapes for endpoints not covered by models.py
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str
    ticket_index: Optional[int] = None


class StepRequest(BaseModel):
    episode_id: str
    action: Action


class GraderRequest(BaseModel):
    episode_id: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["meta"])
def root():
    return {
        "name": "SupportEnv",
        "version": "1.0.0",
        "description": "OpenEnv customer-support ticket triage environment",
        "openenv_spec": "https://github.com/openenv/openenv",
        "tasks": list(TASK_META.keys()),
        "endpoints": {
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state?episode_id=...",
            "tasks": "GET /tasks",
            "grader": "POST /grader",
            "baseline": "POST /baseline",
            "health": "GET /health",
            "docs": "GET /docs",
        },
    }


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Core OpenEnv endpoints
# ---------------------------------------------------------------------------

@app.post("/reset", response_model=Observation, tags=["openenv"])
def reset(request: ResetRequest) -> Observation:
    """
    Start a new episode.

    - **task_id**: `task1` | `task2` | `task3`
    - **ticket_index**: 0-indexed ticket to use (optional; default 0)
    """
    try:
        return env.reset(request.task_id, request.ticket_index)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResult, tags=["openenv"])
def step(request: StepRequest) -> StepResult:
    """
    Submit an action and advance the episode.

    The `action` object must include `action_type` and the fields relevant
    to that action type (see GET /tasks for the schema).
    """
    try:
        return env.step(request.episode_id, request.action)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Episode '{request.episode_id}' not found. Call POST /reset first.",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=State, tags=["openenv"])
def state(episode_id: str = Query(..., description="Episode UUID from POST /reset")) -> State:
    """Return the current state of an episode."""
    try:
        return env.state(episode_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Episode '{episode_id}' not found.",
        )


# ---------------------------------------------------------------------------
# /tasks — task listing + action schema
# ---------------------------------------------------------------------------

# JSON Schema for the Action model (subset used in each task)
_BASE_ACTION_SCHEMA = {
    "type": "object",
    "required": ["action_type"],
    "properties": {
        "action_type": {
            "type": "string",
            "description": "One of the available_actions listed in the Observation",
        },
    },
}

_ACTION_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "task1": {
        **_BASE_ACTION_SCHEMA,
        "description": "classify action: set category + priority; then submit",
        "properties": {
            **_BASE_ACTION_SCHEMA["properties"],
            "category": {
                "type": "string",
                "enum": [
                    "billing", "technical", "account",
                    "feature_request", "complaint", "general",
                ],
            },
            "priority": {
                "type": "string",
                "enum": ["low", "medium", "high", "critical"],
            },
        },
    },
    "task2": {
        **_BASE_ACTION_SCHEMA,
        "description": "extract action: populate extracted_entities + required_actions; then submit",
        "properties": {
            **_BASE_ACTION_SCHEMA["properties"],
            "extracted_entities": {
                "type": "object",
                "additionalProperties": True,
                "description": "Key-value pairs extracted from the ticket text",
            },
            "required_actions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of action identifiers (snake_case) needed to close the ticket",
            },
        },
    },
    "task3": {
        **_BASE_ACTION_SCHEMA,
        "description": (
            "respond or resolve action: write response_text + resolution_steps; "
            "optionally escalate; then submit"
        ),
        "properties": {
            **_BASE_ACTION_SCHEMA["properties"],
            "response_text": {
                "type": "string",
                "description": "Full professional response to send to the customer",
            },
            "resolution_steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Ordered steps for support staff to resolve the ticket",
            },
            "escalation_team": {
                "type": "string",
                "enum": ["billing_team", "engineering", "account_management", "legal"],
            },
            "escalation_reason": {"type": "string"},
        },
    },
}


@app.get("/tasks", response_model=List[TaskInfo], tags=["openenv"])
def list_tasks() -> List[TaskInfo]:
    """Return metadata and action schema for all tasks."""
    result = []
    for task_id, meta in TASK_META.items():
        result.append(
            TaskInfo(
                task_id=task_id,
                name=meta["name"],
                description=meta["description"],
                difficulty=meta["difficulty"],
                max_steps=meta["max_steps"],
                action_schema=_ACTION_SCHEMAS[task_id],
            )
        )
    return result


# ---------------------------------------------------------------------------
# /grader — grade a finished episode
# ---------------------------------------------------------------------------

@app.post("/grader", response_model=GraderResponse, tags=["openenv"])
def grader(request: GraderRequest) -> GraderResponse:
    """
    Grade a finished episode.

    The episode must have reached `done=True` (either via a `submit` action
    or by exhausting `max_steps`).
    """
    try:
        return env.grade(request.episode_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Episode '{request.episode_id}' not found.",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# /baseline — run the built-in baseline agent
# ---------------------------------------------------------------------------

class BaselineRequest(BaseModel):
    model: str = "gpt-4o-mini"
    ticket_index: Optional[int] = 0


@app.post("/baseline", response_model=BaselineResult, tags=["openenv"])
def run_baseline(request: BaselineRequest) -> BaselineResult:
    """
    Run the heuristic baseline agent against all three tasks.

    The built-in baseline does NOT require an OpenAI key — it uses the
    deterministic heuristic baseline from `baseline.py`.
    If you want to run the LLM baseline, call `baseline.py` directly.
    """
    try:
        from baseline import run_heuristic_baseline
        scores = run_heuristic_baseline(
            ticket_index=request.ticket_index or 0
        )
        avg = round(sum(s["score"] for s in scores) / len(scores), 4)
        return BaselineResult(
            model="heuristic-baseline",
            scores=[
                {"task_id": s["task_id"], "score": s["score"], "details": s}
                for s in scores
            ],
            average_score=avg,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))