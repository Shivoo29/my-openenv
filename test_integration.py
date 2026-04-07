"""
Integration test for SupportEnv.

Runs a full episode for each task and prints results.
Usage: python test_integration.py
"""
import environment as env
from models import Action


def test_task(task_id: str) -> bool:
    """Run a full episode for a task. Returns True if passed."""
    print(f"\n{'='*60}")
    print(f"Testing {task_id}")
    print(f"{'='*60}")

    # Reset
    print("1. reset()...")
    obs = env.reset(task_id, ticket_index=0)
    episode_id = obs.episode_id
    print(f"   [OK] episode_id={episode_id[:8]}...")
    print(f"   ticket_id={obs.ticket.ticket_id}  subject={obs.ticket.subject[:50]}")
    print(f"   max_steps={obs.max_steps}  hint={obs.hint}")

    # Take a relevant action
    print("2. step() with task action...")
    if task_id == "task1":
        action = Action(action_type="classify", category="billing", priority="high")
    elif task_id == "task2":
        action = Action(
            action_type="extract",
            extracted_entities={"customer_name": "Robert Chen", "account_id": "ACC-78234"},
            required_actions=["issue_refund"],
        )
    else:  # task3
        action = Action(
            action_type="respond",
            response_text=(
                "We sincerely apologize for the inconvenience with your password reset. "
                "We will manually reset your password and send a new email immediately. "
                "Please check your spam folder and whitelist our domain. "
                "We will resolve this within the next 30 minutes."
            ),
            resolution_steps=[
                "verify_email_delivery",
                "check_spam_filters",
                "manual_password_reset",
                "follow_up_confirmation",
            ],
        )

    result = env.step(episode_id, action)
    print(f"   [OK] step_reward={result.reward.step_reward:+.4f}  done={result.done}")

    # Submit
    print("3. step() submit...")
    if not result.done:
        result = env.step(episode_id, Action(action_type="submit"))
    print(f"   [OK] done={result.done}  total_reward={result.reward.total_reward:.4f}")

    # State
    print("4. get_state()...")
    state = env.get_state(episode_id)
    print(f"   [OK] steps={state.step_number}  history_len={len(state.history)}")

    # Grade
    print("5. grade()...")
    score, breakdown, feedback = env.grade(episode_id)
    print(f"   [OK] score={score:.4f}/1.0")
    print(f"   breakdown: {', '.join(f'{k}={v:.2f}' for k, v in breakdown.items())}")
    print(f"   feedback: {feedback}")

    passed = score >= 0.0  # just verify pipeline works
    return passed


def main():
    print("SupportEnv Integration Test")
    print("=" * 60)

    results = []
    for task_id in ["task1", "task2", "task3"]:
        try:
            ok = test_task(task_id)
            results.append((task_id, ok, None))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            results.append((task_id, False, str(exc)))
        finally:
            env._EPISODES.clear()

    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    all_ok = True
    for task_id, ok, err in results:
        status = "[PASS]" if ok else "[FAIL]"
        print(f"  {status} {task_id}" + (f" — {err}" if err else ""))
        if not ok:
            all_ok = False

    if all_ok:
        print("\n[OK] All integration tests passed!")
    else:
        print("\n[FAIL] Some tests failed.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
