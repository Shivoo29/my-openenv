"""
Core SupportEnv environment logic.

Manages episode lifecycle:
  reset() → Observation
  step(action) → StepResult
  state() → State
  grade() → GraderResponse

Sessions are stored in-memory (dict keyed by episode_id).
For multi-replica deployments replace with Redis or a database.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from data import TASK_META, get_tickets
from graders import grade_episode
from models import (
    Action,
    GraderResponse,
    Observation,
    Reward,
    State,
    StepResult,
    TicketInfo,
)

# In-memory store:  episode_id → EpisodeState dict
_EPISODES: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Step-level reward helpers
# ---------------------------------------------------------------------------

_STEP_COST = -0.02        # small cost per step to discourage wasted steps
_SUBMIT_REWARD = 0.05     # small bonus for explicitly submitting
_MAX_STEP_PENALTY = -0.10  # penalty for hitting max_steps without submitting


def _step_reward_task1(action: Action, done: bool) -> tuple[float, str]:
    """Immediate step reward for task1 episodes."""
    if action.action_type == "classify":
        if action.category and action.priority:
            return 0.05, "Classify action with both fields set."
        return 0.02, "Classify action (partial — missing category or priority)."
    if action.action_type == "submit":
        return _SUBMIT_REWARD, "Episode submitted."
    return _STEP_COST, f"Action '{action.action_type}' not primary for this task."


def _step_reward_task2(action: Action, done: bool) -> tuple[float, str]:
    """Immediate step reward for task2 episodes."""
    if action.action_type == "extract":
        n_entities = len(action.extracted_entities or {})
        n_actions = len(action.required_actions or [])
        reward = min(n_entities * 0.01 + n_actions * 0.02, 0.15)
        return reward, f"Extracted {n_entities} entities, {n_actions} actions."
    if action.action_type == "submit":
        return _SUBMIT_REWARD, "Episode submitted."
    return _STEP_COST, f"Action '{action.action_type}' not primary for this task."


def _step_reward_task3(action: Action, done: bool) -> tuple[float, str]:
    """Immediate step reward for task3 episodes."""
    if action.action_type in ("respond", "resolve"):
        text_len = len((action.response_text or "").split())
        steps_n = len(action.resolution_steps or [])
        reward = min(text_len * 0.001 + steps_n * 0.02, 0.20)
        return reward, f"Response: {text_len} words, {steps_n} steps."
    if action.action_type == "escalate":
        return 0.03, "Escalation action recorded."
    if action.action_type == "submit":
        return _SUBMIT_REWARD, "Episode submitted."
    return _STEP_COST, f"Action '{action.action_type}' not primary for this task."


_STEP_REWARD_FNS = {
    "task1": _step_reward_task1,
    "task2": _step_reward_task2,
    "task3": _step_reward_task3,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reset(task_id: str, ticket_index: Optional[int] = None) -> Observation:
    """
    Create a new episode for the given task.

    Args:
        task_id: "task1" | "task2" | "task3"
        ticket_index: which ticket to use (0-based); if None, defaults to 0.

    Returns the initial Observation.
    """
    if task_id not in TASK_META:
        raise ValueError(f"Unknown task_id {task_id!r}. Valid: {list(TASK_META)}")

    meta = TASK_META[task_id]
    tickets = get_tickets(task_id)
    idx = 0 if ticket_index is None else int(ticket_index) % len(tickets)

    raw_ticket = tickets[idx]
    ticket = TicketInfo(
        ticket_id=raw_ticket["ticket_id"],
        subject=raw_ticket["subject"],
        body=raw_ticket["body"],
        customer_tier=raw_ticket["customer_tier"],
        account_age_days=raw_ticket["account_age_days"],
        previous_tickets=raw_ticket["previous_tickets"],
        attachments=raw_ticket.get("attachments", []),
    )

    episode_id = str(uuid.uuid4())
    _EPISODES[episode_id] = {
        "task_id": task_id,
        "ticket_index": idx,
        "step_number": 0,
        "max_steps": meta["max_steps"],
        "done": False,
        "total_reward": 0.0,
        "action_history": [],
        "final_score": None,
    }

    return Observation(
        task_id=task_id,
        task_description=meta["description"],
        episode_id=episode_id,
        ticket=ticket,
        thread_history=[],
        available_actions=meta["available_actions"],
        step_number=0,
        max_steps=meta["max_steps"],
        hint=_get_hint(task_id, 0),
    )


def step(episode_id: str, action: Action) -> StepResult:
    """
    Advance the episode by one step.

    Raises:
        KeyError: episode_id not found
        ValueError: episode is already done
    """
    ep = _EPISODES[episode_id]  # raises KeyError if not found

    if ep["done"]:
        raise ValueError(f"Episode {episode_id} is already done.")

    task_id: str = ep["task_id"]
    meta = TASK_META[task_id]
    tickets = get_tickets(task_id)

    ep["step_number"] += 1
    ep["action_history"].append(action.model_dump())

    # Determine if done
    done = False
    if action.action_type == "submit":
        done = True
    elif ep["step_number"] >= ep["max_steps"]:
        done = True

    # Compute step reward
    reward_fn = _STEP_REWARD_FNS[task_id]
    step_r, explanation = reward_fn(action, done)

    if done and action.action_type != "submit":
        step_r += _MAX_STEP_PENALTY
        explanation += f" Max steps reached — {_MAX_STEP_PENALTY} penalty."

    # Apply final grader bonus when episode ends
    final_score = None
    if done:
        final_score, _, grader_feedback = grade_episode(
            task_id, ep["ticket_index"], ep["action_history"]
        )
        ep["final_score"] = final_score
        bonus = final_score * 1.0  # scale grader score to up to +1.0 reward
        step_r += bonus
        explanation += f" Grader bonus: +{bonus:.3f} (score={final_score:.3f})."

    ep["total_reward"] = round(ep["total_reward"] + step_r, 4)
    ep["done"] = done

    # Build next observation
    raw_ticket = tickets[ep["ticket_index"]]
    ticket = TicketInfo(
        ticket_id=raw_ticket["ticket_id"],
        subject=raw_ticket["subject"],
        body=raw_ticket["body"],
        customer_tier=raw_ticket["customer_tier"],
        account_age_days=raw_ticket["account_age_days"],
        previous_tickets=raw_ticket["previous_tickets"],
        attachments=raw_ticket.get("attachments", []),
    )

    thread_history = [
        {"role": "agent", "content": str(a)}
        for a in ep["action_history"]
    ]

    obs = Observation(
        task_id=task_id,
        task_description=meta["description"],
        episode_id=episode_id,
        ticket=ticket,
        thread_history=thread_history,
        available_actions=meta["available_actions"] if not done else [],
        step_number=ep["step_number"],
        max_steps=ep["max_steps"],
        hint=_get_hint(task_id, ep["step_number"]) if not done else None,
    )

    reward = Reward(
        step_reward=round(step_r, 4),
        total_reward=ep["total_reward"],
        explanation=explanation,
    )

    info: dict = {
        "step": ep["step_number"],
        "max_steps": ep["max_steps"],
    }
    if done:
        info["final_score"] = final_score

    return StepResult(observation=obs, reward=reward, done=done, info=info)


def state(episode_id: str) -> State:
    """Return the current state of an episode."""
    ep = _EPISODES[episode_id]
    return State(
        task_id=ep["task_id"],
        episode_id=episode_id,
        step_number=ep["step_number"],
        max_steps=ep["max_steps"],
        done=ep["done"],
        total_reward=ep["total_reward"],
        history=ep["action_history"],
        final_score=ep.get("final_score"),
    )


def grade(episode_id: str) -> GraderResponse:
    """
    Run (or re-run) the grader on a finished episode and return the result.

    Raises:
        ValueError: episode is not done yet.
    """
    ep = _EPISODES[episode_id]
    if not ep["done"]:
        raise ValueError(
            f"Episode {episode_id} is not done yet (step {ep['step_number']}/{ep['max_steps']})."
        )

    score, breakdown, feedback = grade_episode(
        ep["task_id"], ep["ticket_index"], ep["action_history"]
    )
    ep["final_score"] = score

    return GraderResponse(
        episode_id=episode_id,
        task_id=ep["task_id"],
        score=score,
        breakdown=breakdown,
        feedback=feedback,
    )


# ---------------------------------------------------------------------------
# Hints
# ---------------------------------------------------------------------------

_HINTS = {
    "task1": [
        "Read the ticket carefully. What is the customer's main problem?",
        "Review your classification — does the priority match the urgency in the message?",
        "Submit your final answer.",
    ],
    "task2": [
        "Scan the ticket for IDs, names, dates, and amounts.",
        "List every action that support staff must take to close this ticket.",
        "Verify you haven't missed any entity.",
        "Double-check required_actions against the ticket body.",
        "Submit your final answer.",
    ],
    "task3": [
        "Read the ticket carefully — what is the customer asking for?",
        "Draft a professional response that acknowledges the issue.",
        "Ensure your resolution_steps are actionable and ordered.",
        "Check that your response covers all required information.",
        "Add any missing keywords (e.g. apology, timeline).",
        "Review tone — is it empathetic and professional?",
        "Verify step completeness before submitting.",
        "Submit your final answer.",
    ],
}


def _get_hint(task_id: str, step_number: int) -> Optional[str]:
    hints = _HINTS.get(task_id, [])
    if step_number < len(hints):
        return hints[step_number]
    return None