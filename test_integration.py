"""
Quick integration test for DevOpsEnv.

This runs a full episode for each task to verify everything works.
"""
import environment as env
from models import Action


def test_task(task_id: str, max_test_steps: int = 5):
    """Test a single task."""
    print(f"\n{'='*60}")
    print(f"Testing {task_id}")
    print(f"{'='*60}")
    
    # Reset
    print("1. Calling reset()...")
    obs = env.reset(task_id)
    episode_id = obs.episode_id
    print(f"[OK] Episode created: {episode_id}")
    print(f"  Task: {obs.task_description[:80]}...")
    print(f"  Max steps: {obs.max_steps}")
    print(f"  System state: cpu={obs.system_state.cpu_usage:.1f}%, mem={obs.system_state.memory_usage_mb}MB")
    
    # Take steps
    print(f"\n2. Taking {max_test_steps} steps...")
    for i in range(max_test_steps):
        if task_id == "task1":
            if i == 0:
                action = Action(action_type="bash_cmd", command="systemctl status nginx")
            elif i == 1:
                action = Action(action_type="bash_cmd", command="systemctl try-restart nginx")
            elif i == 2:
                action = Action(action_type="bash_cmd", command="nginx -t")
            else:
                action = Action(action_type="bash_cmd", command="curl http://localhost")
        elif task_id == "task2":
            if i == 0:
                action = Action(action_type="bash_cmd", command="cat /srv/docker-compose.yml")
            elif i == 1:
                action = Action(
                    action_type="file_edit",
                    file_path="/srv/docker-compose.yml",
                    file_content="version: '3.8'\nservices:\n  mockapi:\n    image: mockapi:latest\n    ports:\n      - \"3000:3000\""
                )
            elif i == 2:
                action = Action(action_type="bash_cmd", command="docker-compose up -d")
            else:
                action = Action(action_type="bash_cmd", command="docker ps")
        else:  # task3
            if i == 0:
                action = Action(action_type="bash_cmd", command="ps aux | grep python")
            elif i == 1:
                action = Action(action_type="bash_cmd", command="kill 300")
            elif i == 2:
                action = Action(
                    action_type="file_edit",
                    file_path="/opt/mockapi/app.py",
                    file_content="import json\nfrom flask import Flask\n\napp = Flask(__name__)\n\n@app.route('/api/data')\ndef get_data():\n    return json.dumps({'status': 'ok'})\n\nif __name__ == '__main__':\n    app.run()\n"
                )
            else:
                action = Action(action_type="bash_cmd", command="python3 /opt/mockapi/app.py &")
        
        try:
            result = env.step(episode_id, action)
            print(f"  Step {i+1}: {action.action_type} - Reward: {result.reward.step_reward:+.3f}")
            if result.done:
                print(f"  -> Episode completed early")
                break
        except Exception as e:
            print(f"  Step {i+1} ERROR: {e}")
            break
    
    # Check state
    print(f"\n3. Calling get_state()...")
    state = env.get_state(episode_id)
    print(f"[OK] State: step_number={state.step_number}, total_reward={state.total_reward:.3f}, done={state.done}")
    
    # Finish episode if not already done
    if not state.done:
        print(f"\n4. Calling submit()...")
        result = env.step(episode_id, Action(action_type="submit"))
        print(f"[OK] Episode submitted, done={result.done}")
    
    # Grade
    print(f"\n5. Calling grade()...")
    try:
        score, breakdown, feedback = env.grade(episode_id)
        print(f"[OK] Score: {score:.3f}/1.0")
        print(f"  Breakdown: {breakdown}")
        print(f"  Feedback: {feedback}")
    except Exception as e:
        print(f"[ERROR] Grading error: {e}")


def main():
    """Run all tests."""
    print("DevOpsEnv Integration Test")
    print("="*60)
    
    try:
        test_task("task1", max_test_steps=5)
        test_task("task2", max_test_steps=5)
        test_task("task3", max_test_steps=5)
        
        print(f"\n{'='*60}")
        print("[OK] All tests completed successfully!")
        print(f"{'='*60}")
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
