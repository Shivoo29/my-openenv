"""
Pydantic models for SupportEnv — Customer Support Ticket Triage.

Domain: SaaS customer support automation
Tasks: classification, information extraction, resolution generation
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Ticket Info (what the agent sees)
# ---------------------------------------------------------------------------

class TicketInfo(BaseModel):
    """A customer support ticket presented to the agent."""
    ticket_id: str
    subject: str
    body: str
    customer_tier: str = Field(description="free | pro | enterprise")
    account_age_days: int
    previous_tickets: int
    attachments: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """Everything the agent sees at each step."""
    task_id: str = Field(description="task1 | task2 | task3")
    task_description: str
    episode_id: str
    ticket: TicketInfo
    thread_history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Ordered list of {'role': 'agent'|'system', 'content': str}",
    )
    available_actions: List[str]
    step_number: int
    max_steps: int
    hint: Optional[str] = None


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """Agent action for support ticket processing."""
    action_type: str = Field(
        description="classify | extract | respond | resolve | escalate | submit"
    )
    # Task 1: Classification
    category: Optional[str] = Field(
        default=None,
        description="billing | technical | account | feature_request | complaint | general",
    )
    priority: Optional[str] = Field(
        default=None,
        description="low | medium | high | critical",
    )
    # Task 2: Extraction
    extracted_entities: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Key-value pairs extracted from the ticket",
    )
    required_actions: Optional[List[str]] = Field(
        default=None,
        description="List of actions needed to resolve the ticket",
    )
    # Task 3: Resolution
    response_text: Optional[str] = Field(
        default=None,
        description="Customer-facing response text",
    )
    resolution_steps: Optional[List[str]] = Field(
        default=None,
        description="Ordered list of internal resolution steps",
    )
    # Escalation
    escalation_team: Optional[str] = Field(default=None)
    escalation_reason: Optional[str] = Field(default=None)


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    """Per-step reward signal."""
    step_reward: float
    total_reward: float
    explanation: str


# ---------------------------------------------------------------------------
# Step Result
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class State(BaseModel):
    task_id: str
    episode_id: str
    step_number: int
    max_steps: int
    done: bool
    total_reward: float
    history: List[Dict[str, Any]] = Field(default_factory=list)
    final_score: Optional[float] = None


# ---------------------------------------------------------------------------
# Task Metadata
# ---------------------------------------------------------------------------

class TaskInfo(BaseModel):
    task_id: str
    name: str
    description: str
    difficulty: str
    max_steps: int


# ---------------------------------------------------------------------------
# Grader Response
# ---------------------------------------------------------------------------

class GraderResponse(BaseModel):
    episode_id: str
    task_id: str
    score: float = Field(description="Final grader score 0.0–1.0")
    breakdown: Dict[str, float] = Field(default_factory=dict)
    feedback: str


class BaselineResult(BaseModel):
    """Result of running the baseline agent."""
    task_id: str
    episode_id: str
    final_score: float
    step_count: int
    total_reward: float
    actions: List[Dict[str, Any]]
