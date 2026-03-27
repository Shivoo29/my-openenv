"""
Pydantic models for the SupportEnv OpenEnv environment.

Domain: Customer Support Ticket Resolution (SaaS company)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Core domain model
# ---------------------------------------------------------------------------

class TicketInfo(BaseModel):
    """A single customer-support ticket."""
    ticket_id: str
    subject: str
    body: str
    customer_tier: str = Field(
        description="Subscription tier: free | pro | enterprise"
    )
    account_age_days: int
    previous_tickets: int
    attachments: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """Everything the agent sees at each step."""
    task_id: str = Field(description="Task identifier: task1 | task2 | task3")
    task_description: str = Field(description="Human-readable task description")
    episode_id: str = Field(description="Unique episode UUID")
    ticket: TicketInfo
    thread_history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Ordered list of {'role': 'agent'|'system', 'content': str}"
    )
    available_actions: List[str] = Field(
        description="List of valid action_type values for this step"
    )
    step_number: int
    max_steps: int
    hint: Optional[str] = Field(
        default=None,
        description="Optional hint shown to the agent (may be None)"
    )


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """
    The action an agent submits via POST /step.

    Only the fields relevant to the chosen action_type need to be populated.
    """
    action_type: str = Field(
        description=(
            "One of: classify | extract | respond | escalate | resolve | submit"
        )
    )

    # ---- Task 1: classify -------------------------------------------------
    category: Optional[str] = Field(
        default=None,
        description=(
            "Ticket category: billing | technical | account | "
            "feature_request | complaint | general"
        )
    )
    priority: Optional[str] = Field(
        default=None,
        description="Priority level: low | medium | high | critical"
    )

    # ---- Task 2: extract --------------------------------------------------
    extracted_entities: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Key-value pairs extracted from the ticket. Expected keys vary by "
            "ticket; see task description."
        )
    )
    required_actions: Optional[List[str]] = Field(
        default=None,
        description="List of actions that must be taken to resolve the ticket"
    )

    # ---- Task 3: respond / resolve ----------------------------------------
    response_text: Optional[str] = Field(
        default=None,
        description="Full text of the agent's response to the customer"
    )
    resolution_steps: Optional[List[str]] = Field(
        default=None,
        description="Ordered list of steps to resolve the ticket"
    )

    # ---- escalate ---------------------------------------------------------
    escalation_team: Optional[str] = Field(
        default=None,
        description=(
            "Team to escalate to: billing_team | engineering | "
            "account_management | legal"
        )
    )
    escalation_reason: Optional[str] = Field(
        default=None,
        description="Brief reason for escalation"
    )

    # ---- submit -----------------------------------------------------------
    # No extra fields — signals the agent is done with the episode.


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    """Per-step reward signal."""
    step_reward: float = Field(
        description="Reward earned this step (can be negative)"
    )
    total_reward: float = Field(
        description="Cumulative reward for the episode so far"
    )
    explanation: str = Field(
        description="Human-readable explanation of what drove the reward"
    )


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# State (returned by GET /state)
# ---------------------------------------------------------------------------

class State(BaseModel):
    task_id: str
    episode_id: str
    step_number: int
    max_steps: int
    done: bool
    total_reward: float
    history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Ordered list of (action, reward) pairs for the episode"
    )
    final_score: Optional[float] = Field(
        default=None,
        description="Grader score 0.0–1.0, set when episode is done"
    )


# ---------------------------------------------------------------------------
# Task metadata (returned by GET /tasks)
# ---------------------------------------------------------------------------

class TaskInfo(BaseModel):
    task_id: str
    name: str
    description: str
    difficulty: str = Field(description="easy | medium | hard")
    max_steps: int
    action_schema: Dict[str, Any] = Field(
        description="JSON Schema fragment describing required Action fields"
    )


# ---------------------------------------------------------------------------
# Baseline / grader response shapes
# ---------------------------------------------------------------------------

class GraderResponse(BaseModel):
    episode_id: str
    task_id: str
    score: float = Field(description="Final grader score 0.0–1.0")
    breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-criterion partial scores"
    )
    feedback: str


class BaselineScore(BaseModel):
    task_id: str
    score: float
    details: Dict[str, Any] = Field(default_factory=dict)


class BaselineResult(BaseModel):
    model: str
    scores: List[BaselineScore]
    average_score: float