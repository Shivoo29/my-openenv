"""
Core SupportEnv environment logic.

Simulates a customer support ticket triage workflow:
- Task 1 (easy):   Ticket Classification — assign category + priority
- Task 2 (medium): Information Extraction — pull entities + required actions
- Task 3 (hard):   Resolution Generation — write response + resolution steps

Manages episode lifecycle:
  reset(task_id, ticket_index) → Observation
  step(episode_id, action)     → StepResult
  get_state(episode_id)        → State
  grade(episode_id)            → (score, breakdown, feedback)
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, Optional, Tuple

from data import TASK_META, get_task_meta, get_tickets
from graders import grade_task
from models import (
    Action,
    Observation,
    Reward,
    State,
    StepResult,
    TicketInfo,
)

# In-memory store: episode_id → episode dict
_EPISODES: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Reward constants (match openenv.yaml)
# ---------------------------------------------------------------------------

STEP_COST = -0.02
SUBMIT_BONUS = 0.05
MAX_STEP_PENALTY = -0.10


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def reset(task_id: str, ticket_index: int = 0) -> Observation:
    """Create a new episode for the given task and ticket."""
    if task_id not in TASK_META:
        raise ValueError(f"Unknown task_id {task_id!r}. Valid: {list(TASK_META)}")

    meta = TASK_META[task_id]
    tickets = get_tickets(task_id)

    if ticket_index < 0 or ticket_index >= len(tickets):
        raise ValueError(
            f"ticket_index {ticket_index} out of range [0, {len(tickets) - 1}]"
        )

    ticket_data = tickets[ticket_index]
    safe_meta = get_task_meta(task_id)

    episode_id = str(uuid.uuid4())
    _EPISODES[episode_id] = {
        "task_id": task_id,
        "ticket_index": ticket_index,
        "ticket_data": ticket_data,
        "step_number": 0,
        "max_steps": meta["max_steps"],
        "done": False,
        "total_reward": 0.0,
        "action_history": [],
        "final_score": None,
    }

    ticket_info = TicketInfo(
        ticket_id=ticket_data["ticket_id"],
        subject=ticket_data["subject"],
        body=ticket_data["body"],
        customer_tier=ticket_data["customer_tier"],
        account_age_days=ticket_data["account_age_days"],
        previous_tickets=ticket_data["previous_tickets"],
        attachments=ticket_data.get("attachments", []),
    )

    return Observation(
        task_id=task_id,
        task_description=safe_meta["description"],
        episode_id=episode_id,
        ticket=ticket_info,
        thread_history=[],
        available_actions=safe_meta["available_actions"],
        step_number=0,
        max_steps=meta["max_steps"],
        hint=_get_hint(task_id, 0),
    )


def step(episode_id: str, action: Action) -> StepResult:
    """Advance the episode by one step."""
    ep = _EPISODES.get(episode_id)
    if ep is None:
        raise KeyError(f"Episode {episode_id} not found")
    if ep["done"]:
        raise ValueError(f"Episode {episode_id} is already done.")

    task_id = ep["task_id"]

    ep["step_number"] += 1
    ep["action_history"].append(action.model_dump())

    # Determine if done
    done = False
    if action.action_type == "submit":
        done = True
    elif ep["step_number"] >= ep["max_steps"]:
        done = True

    # Calculate step reward
    step_reward, explanation = _calculate_step_reward(task_id, action, ep, done)

    # Apply grader bonus on terminal step
    if done:
        final_score, _breakdown, _feedback = grade_task(task_id, ep)
        ep["final_score"] = final_score
        # Grader score is the terminal bonus (0–1)
        step_reward += final_score
        explanation += f" | Grader score: {final_score:.3f}"

        # Penalty for running out of steps without submitting
        if action.action_type != "submit" and ep["step_number"] >= ep["max_steps"]:
            step_reward += MAX_STEP_PENALTY
            explanation += f" | Max-step penalty: {MAX_STEP_PENALTY}"
    else:
        final_score = None

    ep["total_reward"] = round(ep["total_reward"] + step_reward, 4)
    ep["done"] = done

    # Build observation
    ticket_data = ep["ticket_data"]
    safe_meta = get_task_meta(task_id)

    ticket_info = TicketInfo(
        ticket_id=ticket_data["ticket_id"],
        subject=ticket_data["subject"],
        body=ticket_data["body"],
        customer_tier=ticket_data["customer_tier"],
        account_age_days=ticket_data["account_age_days"],
        previous_tickets=ticket_data["previous_tickets"],
        attachments=ticket_data.get("attachments", []),
    )

    thread_history = [
        {"role": "agent", "content": _summarize_action(a)}
        for a in ep["action_history"]
    ]

    obs = Observation(
        task_id=task_id,
        task_description=safe_meta["description"],
        episode_id=episode_id,
        ticket=ticket_info,
        thread_history=thread_history,
        available_actions=safe_meta["available_actions"] if not done else [],
        step_number=ep["step_number"],
        max_steps=ep["max_steps"],
        hint=None if done else _get_hint(task_id, ep["step_number"]),
    )

    reward = Reward(
        step_reward=round(min(max(step_reward, 0.01), 0.99), 4),
        total_reward=ep["total_reward"],
        explanation=explanation,
    )

    info: Dict[str, Any] = {"step": ep["step_number"]}
    if done:
        info["final_score"] = final_score

    return StepResult(observation=obs, reward=reward, done=done, info=info)


def get_state(episode_id: str) -> State:
    """Return the current state of an episode."""
    ep = _EPISODES.get(episode_id)
    if ep is None:
        raise KeyError(f"Episode {episode_id} not found")

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


def grade(episode_id: str) -> Tuple[float, Dict[str, float], str]:
    """Grade a finished episode."""
    ep = _EPISODES.get(episode_id)
    if ep is None:
        raise KeyError(f"Episode {episode_id} not found")
    if not ep.get("done"):
        raise ValueError(f"Episode {episode_id} is not done yet")

    task_id = ep["task_id"]
    score, breakdown, feedback = grade_task(task_id, ep)
    ep["final_score"] = score
    return score, breakdown, feedback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _calculate_step_reward(
    task_id: str, action: Action, ep: Dict[str, Any], done: bool
) -> Tuple[float, str]:
    """Dense per-step reward."""
    reward = STEP_COST  # small cost per step

    if action.action_type == "submit":
        reward += SUBMIT_BONUS
        return reward, "Submitted for grading"

    # Partial-progress signals based on task
    if task_id == "task1":
        if action.action_type == "classify":
            if action.category:
                reward += 0.02
            if action.priority:
                reward += 0.02
            return reward, f"Classified: category={action.category}, priority={action.priority}"

    elif task_id == "task2":
        if action.action_type == "extract":
            n_entities = len(action.extracted_entities) if action.extracted_entities else 0
            n_actions = len(action.required_actions) if action.required_actions else 0
            reward += min(n_entities * 0.005, 0.04)
            reward += min(n_actions * 0.005, 0.02)
            return reward, f"Extracted {n_entities} entities, {n_actions} actions"

    elif task_id == "task3":
        if action.action_type == "respond":
            text_len = len(action.response_text or "")
            n_steps = len(action.resolution_steps) if action.resolution_steps else 0
            if text_len > 0:
                reward += min(text_len * 0.0001, 0.03)
            if n_steps > 0:
                reward += min(n_steps * 0.005, 0.02)
            return reward, f"Response ({text_len} chars), {n_steps} resolution steps"

    return reward, "Step taken"


def _summarize_action(action_dict: Dict[str, Any]) -> str:
    """One-line summary of an action for thread_history."""
    atype = action_dict.get("action_type", "unknown")
    if atype == "classify":
        return f"classify(category={action_dict.get('category')}, priority={action_dict.get('priority')})"
    elif atype == "extract":
        ents = action_dict.get("extracted_entities") or {}
        acts = action_dict.get("required_actions") or []
        return f"extract(entities={list(ents.keys())}, actions={acts})"
    elif atype == "respond":
        text = (action_dict.get("response_text") or "")[:60]
        steps = action_dict.get("resolution_steps") or []
        return f"respond(text='{text}...', steps={len(steps)})"
    elif atype == "submit":
        return "submit()"
    return f"{atype}()"


def _get_hint(task_id: str, step: int) -> Optional[str]:
    """Contextual hints to guide the agent."""
    if step == 0:
        hints = {
            "task1": "Read the ticket carefully and classify by category and priority.",
            "task2": "Extract all entities (IDs, names, amounts) and identify required actions.",
            "task3": "Write a professional response and list resolution steps.",
        }
        return hints.get(task_id)
    return None
