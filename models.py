"""
Pydantic models for DevOpsEnv OpenEnv environment.

Domain: Linux DevOps & SRE Troubleshooting
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# System State Models
# ---------------------------------------------------------------------------

class SystemState(BaseModel):
    """Current state of the mock Linux server."""
    task_id: str
    available_commands: List[str]
    filesystem_snapshot: str
    running_processes: List[Dict[str, Any]]
    service_status: Dict[str, str]
    logs: str
    http_ports_open: List[int]
    docker_containers: List[Dict[str, str]]
    cpu_usage: float
    memory_usage_mb: int


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """Everything the agent sees at each step."""
    task_id: str = Field(description="task1 | task2 | task3")
    task_description: str = Field(description="Human-readable task description")
    episode_id: str = Field(description="Unique episode UUID")
    system_state: SystemState
    thread_history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Ordered list of {'role': 'agent'|'system', 'content': str}"
    )
    available_actions: List[str]
    step_number: int
    max_steps: int
    hint: Optional[str] = Field(default=None)


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """Agent action: run a bash command, edit a file, or submit."""
    action_type: str = Field(description="bash_cmd | file_edit | submit")
    command: Optional[str] = Field(default=None, description="Bash command to execute")
    file_path: Optional[str] = Field(default=None, description="Absolute path to file to edit")
    file_content: Optional[str] = Field(default=None, description="New full content for the file")
    summary: Optional[str] = Field(default=None, description="Final summary of actions taken")


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
    final_score: Optional[float] = Field(default=None)


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