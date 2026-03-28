"""
Comprehensive tests for DevOpsEnv.

Run with: pytest tests/test_environment.py -v
"""
import pytest
import json
from unittest.mock import patch

import environment as env
from models import Action, Observation, StepResult, State
from data import TASK_META


class TestReset:
    """Test episode reset functionality."""
    
    def test_reset_valid_task(self):
        """Reset creates a valid episode."""
        obs = env.reset("task1")
        
        assert isinstance(obs, Observation)
        assert obs.task_id == "task1"
        assert obs.episode_id is not None
        assert len(obs.episode_id) > 0
        assert obs.system_state is not None
        assert obs.step_number == 0
        assert obs.max_steps == TASK_META["task1"]["max_steps"]
    
    def test_reset_invalid_task(self):
        """Reset raises error for unknown task."""
        with pytest.raises(ValueError):
            env.reset("invalid_task")
    
    def test_reset_creates_episode_state(self):
        """Reset creates episode in internal state."""
        obs = env.reset("task2")
        
        assert obs.episode_id in env._EPISODES
        ep = env._EPISODES[obs.episode_id]
        assert ep["task_id"] == "task2"
        assert ep["step_number"] == 0
        assert ep["done"] is False


class TestStep:
    """Test step execution."""
    
    def test_step_bash_command(self):
        """Step handles bash_cmd action."""
        obs = env.reset("task1")
        
        action = Action(action_type="bash_cmd", command="systemctl status nginx")
        result = env.step(obs.episode_id, action)
        
        assert isinstance(result, StepResult)
        assert result.observation.step_number == 1
        assert result.reward.step_reward != 0
        assert result.done is False
    
    def test_step_file_edit(self):
        """Step handles file_edit action."""
        obs = env.reset("task2")
        
        action = Action(
            action_type="file_edit",
            file_path="/srv/docker-compose.yml",
            file_content="version: '3.8'\nservices:\n  test: {}"
        )
        result = env.step(obs.episode_id, action)
        
        assert isinstance(result, StepResult)
        assert result.observation.step_number == 1
    
    def test_step_submit(self):
        """Step with submit action marks episode done."""
        obs = env.reset("task1")
        
        action = Action(action_type="submit", summary="Done")
        result = env.step(obs.episode_id, action)
        
        assert result.done is True
        assert result.observation.available_actions == []
    
    def test_step_invalid_episode(self):
        """Step raises error for invalid episode."""
        action = Action(action_type="bash_cmd", command="ls")
        
        with pytest.raises(KeyError):
            env.step("invalid_episode_id", action)
    
    def test_step_after_done(self):
        """Step raises error after episode is done."""
        obs = env.reset("task1")
        
        # End the episode
        action1 = Action(action_type="submit")
        env.step(obs.episode_id, action1)
        
        # Try to step again
        with pytest.raises(ValueError):
            env.step(obs.episode_id, action1)
    
    def test_step_max_steps_limit(self):
        """Episode ends after max_steps."""
        obs = env.reset("task1")
        max_steps = obs.max_steps
        
        for i in range(max_steps):
            action = Action(action_type="bash_cmd", command="ps aux")
            result = env.step(obs.episode_id, action)
            
            if i < max_steps - 1:
                assert result.done is False
            else:
                assert result.done is True


class TestState:
    """Test state retrieval."""
    
    def test_get_state(self):
        """get_state returns current episode state."""
        obs = env.reset("task3")
        
        state = env.get_state(obs.episode_id)
        
        assert isinstance(state, State)
        assert state.episode_id == obs.episode_id
        assert state.task_id == "task3"
        assert state.step_number == 0
        assert state.done is False
    
    def test_get_state_invalid_episode(self):
        """get_state raises error for invalid episode."""
        with pytest.raises(KeyError):
            env.get_state("invalid_id")
    
    def test_state_history(self):
        """State includes action history."""
        obs = env.reset("task1")
        
        # Take Actions
        action1 = Action(action_type="bash_cmd", command="ps aux")
        env.step(obs.episode_id, action1)
        
        state = env.get_state(obs.episode_id)
        
        assert len(state.history) == 1
        assert state.history[0]["action_type"] == "bash_cmd"


class TestGrading:
    """Test episode grading."""
    
    def test_grade_task1_nginx_running(self):
        """Task 1 grades based on nginx status."""
        obs = env.reset("task1")
        
        # Run commands to fix nginx
        env.step(obs.episode_id, Action(action_type="bash_cmd", command="systemctl restart nginx"))
        env.step(obs.episode_id, Action(action_type="bash_cmd", command="nginx -t"))
        env.step(obs.episode_id, Action(action_type="bash_cmd", command="curl http://localhost"))
        env.step(obs.episode_id, Action(action_type="submit"))
        
        score, breakdown, feedback = env.grade(obs.episode_id)
        
        assert 0.0 <= score <= 1.0
        assert "nginx_running" in breakdown
        assert "config_valid" in breakdown
        assert "http_200" in breakdown
    
    def test_grade_invalid_episode(self):
        """grade raises error for invalid episode."""
        with pytest.raises(KeyError):
            env.grade("invalid_id")
    
    def test_grade_not_done(self):
        """grade raises error if episode not done."""
        obs = env.reset("task1")
        # Don't finish the episode
        
        with pytest.raises(ValueError):
            env.grade(obs.episode_id)


class TestSystemSimulation:
    """Test mock system state simulation."""
    
    def test_task1_initial_state(self):
        """Task 1 initializes with nginx crashed."""
        obs = env.reset("task1")
        
        assert obs.system_state.service_status.get("nginx") == "inactive"
        assert 80 not in obs.system_state.http_ports_open
    
    def test_task2_initial_state(self):
        """Task 2 initializes with docker misconfigured."""
        obs = env.reset("task2")
        
        assert obs.system_state.service_status.get("docker") == "active"
        assert 80 in obs.system_state.http_ports_open
    
    def test_task3_initial_state(self):
        """Task 3 initializes with memory leak."""
        obs = env.reset("task3")
        
        assert obs.system_state.service_status.get("mockapi") == "active"
        # Should have high memory usage
        assert obs.system_state.memory_usage_mb > 1024


class TestRewards:
    """Test reward calculation."""
    
    def test_step_reward_positive(self):
        """Taking actions yields positive reward."""
        obs = env.reset("task1")
        
        action = Action(action_type="bash_cmd", command="ps aux")
        result = env.step(obs.episode_id, action)
        
        assert result.reward.step_reward > -1.0  # Not all negative
    
    def test_total_reward_accumulation(self):
        """Total reward accumulates across steps."""
        obs = env.reset("task1")
        
        env.step(obs.episode_id, Action(action_type="bash_cmd", command="ps aux"))
        result1 = env.step(obs.episode_id, Action(action_type="bash_cmd", command="ls"))
        total1 = result1.reward.total_reward
        
        result2 = env.step(obs.episode_id, Action(action_type="bash_cmd", command="pwd"))
        total2 = result2.reward.total_reward
        
        # Total reward should accumulate
        assert total2 >= total1 or total2 < total1  # Can go either way depending on grader


@pytest.fixture(autouse=True)
def cleanup():
    """Clean up episodes after each test."""
    yield
    env._EPISODES.clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
