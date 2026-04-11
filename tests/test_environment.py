"""Environment behavior: action validation and progressive observations."""

from __future__ import annotations

import pytest

from environment import reset, step
from models import Action


def test_rejects_cross_task_actions() -> None:
    obs = reset("task3", 0)
    with pytest.raises(ValueError, match="not valid for task3"):
        step(obs.episode_id, Action(action_type="classify"))


def test_thread_history_includes_system_revelations() -> None:
    obs = reset("task1", 0)
    r = step(
        obs.episode_id,
        Action(action_type="classify", category="billing", priority="high"),
    )
    roles = [m.get("role") for m in r.observation.thread_history]
    assert "system" in roles
    assert roles.count("agent") >= 1
