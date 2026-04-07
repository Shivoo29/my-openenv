"""
Comprehensive tests for SupportEnv.

Run with: pytest tests_new.py -v
"""
import pytest

import environment as env
from models import Action, Observation, StepResult, State
from data import TASK_META


class TestReset:
    """Test episode reset functionality."""

    def test_reset_task1(self):
        obs = env.reset("task1")
        assert isinstance(obs, Observation)
        assert obs.task_id == "task1"
        assert obs.episode_id is not None
        assert obs.ticket is not None
        assert obs.step_number == 0
        assert obs.max_steps == TASK_META["task1"]["max_steps"]

    def test_reset_task2(self):
        obs = env.reset("task2")
        assert obs.task_id == "task2"
        assert obs.ticket.ticket_id.startswith("T2-")

    def test_reset_task3(self):
        obs = env.reset("task3")
        assert obs.task_id == "task3"
        assert obs.ticket.ticket_id.startswith("T3-")

    def test_reset_with_ticket_index(self):
        obs = env.reset("task1", ticket_index=2)
        assert obs.ticket.ticket_id == "T1-003"

    def test_reset_invalid_task(self):
        with pytest.raises(ValueError):
            env.reset("task_unknown")

    def test_reset_invalid_ticket_index(self):
        with pytest.raises(ValueError):
            env.reset("task1", ticket_index=99)

    def test_reset_creates_episode(self):
        obs = env.reset("task1")
        assert obs.episode_id in env._EPISODES
        ep = env._EPISODES[obs.episode_id]
        assert ep["task_id"] == "task1"
        assert ep["step_number"] == 0
        assert ep["done"] is False

    def test_reset_hint_on_first_step(self):
        obs = env.reset("task1")
        assert obs.hint is not None


class TestStep:
    """Test step execution."""

    def test_step_classify(self):
        obs = env.reset("task1")
        action = Action(action_type="classify", category="billing", priority="high")
        result = env.step(obs.episode_id, action)
        assert isinstance(result, StepResult)
        assert result.observation.step_number == 1
        assert result.reward.step_reward is not None

    def test_step_extract(self):
        obs = env.reset("task2")
        action = Action(
            action_type="extract",
            extracted_entities={"customer_name": "Alice"},
            required_actions=["issue_refund"],
        )
        result = env.step(obs.episode_id, action)
        assert result.observation.step_number == 1

    def test_step_respond(self):
        obs = env.reset("task3")
        action = Action(
            action_type="respond",
            response_text="Thank you for reaching out. We sincerely apologize for the inconvenience and will resolve this immediately.",
            resolution_steps=["verify_account", "issue_refund"],
        )
        result = env.step(obs.episode_id, action)
        assert result.observation.step_number == 1

    def test_step_submit_marks_done(self):
        obs = env.reset("task1")
        result = env.step(obs.episode_id, Action(action_type="submit"))
        assert result.done is True
        assert result.observation.available_actions == []

    def test_step_invalid_episode(self):
        with pytest.raises(KeyError):
            env.step("nonexistent-id", Action(action_type="submit"))

    def test_step_after_done_raises(self):
        obs = env.reset("task1")
        env.step(obs.episode_id, Action(action_type="submit"))
        with pytest.raises(ValueError):
            env.step(obs.episode_id, Action(action_type="submit"))

    def test_step_max_steps_ends_episode(self):
        obs = env.reset("task1")
        max_steps = obs.max_steps
        for i in range(max_steps):
            action = Action(action_type="classify", category="general", priority="low")
            result = env.step(obs.episode_id, action)
        assert result.done is True

    def test_thread_history_grows(self):
        obs = env.reset("task1")
        env.step(obs.episode_id, Action(action_type="classify", category="billing", priority="high"))
        result = env.step(obs.episode_id, Action(action_type="submit"))
        assert len(result.observation.thread_history) == 2


class TestState:
    """Test state retrieval."""

    def test_get_state_initial(self):
        obs = env.reset("task1")
        state = env.get_state(obs.episode_id)
        assert isinstance(state, State)
        assert state.episode_id == obs.episode_id
        assert state.task_id == "task1"
        assert state.step_number == 0
        assert state.done is False

    def test_get_state_invalid(self):
        with pytest.raises(KeyError):
            env.get_state("bad-id")

    def test_state_history_after_step(self):
        obs = env.reset("task2")
        env.step(obs.episode_id, Action(action_type="extract", extracted_entities={}, required_actions=[]))
        state = env.get_state(obs.episode_id)
        assert len(state.history) == 1
        assert state.history[0]["action_type"] == "extract"


class TestGraders:
    """Test grading for each task."""

    def test_grade_task1_perfect(self):
        """Correct category + priority on ticket 0 (billing/high)."""
        obs = env.reset("task1", ticket_index=0)
        env.step(obs.episode_id, Action(action_type="classify", category="billing", priority="high"))
        env.step(obs.episode_id, Action(action_type="submit"))
        score, breakdown, feedback = env.grade(obs.episode_id)
        assert score >= 0.9
        assert breakdown["category_correct"] == 0.50
        assert breakdown["priority_correct"] == 0.40

    def test_grade_task1_wrong_category(self):
        obs = env.reset("task1", ticket_index=0)
        env.step(obs.episode_id, Action(action_type="classify", category="technical", priority="high"))
        env.step(obs.episode_id, Action(action_type="submit"))
        score, breakdown, _ = env.grade(obs.episode_id)
        assert breakdown["category_correct"] == 0.0
        assert breakdown["priority_correct"] == 0.40

    def test_grade_task1_no_classify_action(self):
        obs = env.reset("task1")
        env.step(obs.episode_id, Action(action_type="submit"))
        score, _, _ = env.grade(obs.episode_id)
        assert score == 0.0

    def test_grade_task2_entities(self):
        obs = env.reset("task2", ticket_index=0)
        env.step(obs.episode_id, Action(
            action_type="extract",
            extracted_entities={
                "customer_name": "Robert Chen",
                "account_id": "ACC-78234",
                "invoice_number": "INV-20240312",
                "incorrect_amount": "199.00",
                "correct_amount": "99.00",
                "refund_amount": "100.00",
            },
            required_actions=["issue_refund", "send_corrected_invoice"],
        ))
        env.step(obs.episode_id, Action(action_type="submit"))
        score, breakdown, _ = env.grade(obs.episode_id)
        assert breakdown["entity_coverage"] == pytest.approx(0.60, abs=0.01)
        assert breakdown["action_coverage"] == pytest.approx(0.30, abs=0.01)

    def test_grade_task3_keywords_and_steps(self):
        obs = env.reset("task3", ticket_index=0)
        env.step(obs.episode_id, Action(
            action_type="respond",
            response_text=(
                "We sincerely apologize for the password reset issue. "
                "We will send a new reset email and ask you to check your spam folder "
                "and whitelist our domain. We will have this resolved within the hour."
            ),
            resolution_steps=[
                "verify_email_delivery",
                "check_spam_filters",
                "manual_password_reset",
                "follow_up_confirmation",
            ],
        ))
        env.step(obs.episode_id, Action(action_type="submit"))
        score, breakdown, _ = env.grade(obs.episode_id)
        assert score >= 0.7
        assert breakdown["length_adequate"] == 0.10
        assert breakdown["no_empty_steps"] == 0.05

    def test_grade_not_done_raises(self):
        obs = env.reset("task1")
        with pytest.raises(ValueError):
            env.grade(obs.episode_id)

    def test_grade_invalid_episode_raises(self):
        with pytest.raises(KeyError):
            env.grade("bad-id")

    def test_score_in_range(self):
        for task_id in ["task1", "task2", "task3"]:
            obs = env.reset(task_id)
            env.step(obs.episode_id, Action(action_type="submit"))
            score, _, _ = env.grade(obs.episode_id)
            assert 0.0 <= score <= 1.0


class TestRewards:
    """Test reward signals."""

    def test_step_reward_is_float(self):
        obs = env.reset("task1")
        result = env.step(obs.episode_id, Action(action_type="classify", category="billing", priority="high"))
        assert isinstance(result.reward.step_reward, float)

    def test_total_reward_accumulates(self):
        obs = env.reset("task2")
        r1 = env.step(obs.episode_id, Action(action_type="extract", extracted_entities={}, required_actions=[]))
        r2 = env.step(obs.episode_id, Action(action_type="submit"))
        assert r2.reward.total_reward != r1.reward.total_reward

    def test_submit_bonus_applied(self):
        obs = env.reset("task1")
        result = env.step(obs.episode_id, Action(action_type="submit"))
        # submit_bonus=0.05 minus step_cost=0.02 = +0.03 base before grader
        assert result.reward.step_reward > 0.0


@pytest.fixture(autouse=True)
def cleanup():
    yield
    env._EPISODES.clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
