"""
Baseline LLM inference agent for DevOpsEnv.

This script reads an OpenEnv environment's state() and uses an LLM to generate
actions that solve the DevOps tasks.

Usage:
  python inference.py --task task1 --model gpt-4 --hf-token <token>
"""
import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Optional

import requests
from google import genai
from openai import OpenAI

# Load .env values from current folder (if present) before reading config.
def _load_dotenv_from_workspace() -> None:
    """Load KEY=VALUE pairs from .env into os.environ without overriding existing vars."""
    dotenv_path = Path(__file__).resolve().parent / ".env"
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        # Remove surrounding quotes if present.
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]

        os.environ.setdefault(key, value)


_load_dotenv_from_workspace()

# Read config from environment/.env
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:7860")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")
GEMINI_DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")


def _get_openai_client() -> OpenAI:
    """Create an OpenAI-compatible client for OpenAI-style chat completions."""
    api_key = os.environ.get("OPENAI_API_KEY", "sk-test")
    client_kwargs = {"api_key": api_key}
    if OPENAI_BASE_URL:
        client_kwargs["base_url"] = OPENAI_BASE_URL
    return OpenAI(**client_kwargs)


def _get_gemini_client() -> genai.Client:
    """Create a Gemini client using the official google-genai SDK."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required for Gemini models")
    return genai.Client(api_key=api_key)


def _is_gemini_model(model: str) -> bool:
    """Detect whether the requested model should use the Gemini SDK path."""
    m = (model or "").lower()
    return "gemini" in m


def _resolve_gemini_model(model: str) -> str:
    """Map shorthand Gemini model names to concrete model IDs."""
    m = (model or "").strip()
    if not m or m.lower() == "gemini":
        return GEMINI_DEFAULT_MODEL
    return m


def _generate_action_text(
    model: str,
    system_prompt: str,
    user_prompt: str,
    openai_client: Optional[OpenAI],
    gemini_client: Optional[genai.Client],
) -> str:
    """Generate model output text using Gemini SDK or OpenAI-compatible chat."""
    if _is_gemini_model(model):
        if gemini_client is None:
            raise ValueError("Gemini client was not initialized")
        gemini_model = _resolve_gemini_model(model)
        combined_prompt = (
            f"System instructions:\n{system_prompt}\n\n"
            f"User request:\n{user_prompt}"
        )
        response = gemini_client.models.generate_content(
            model=gemini_model,
            contents=combined_prompt,
        )
        return response.text or ""

    if openai_client is None:
        raise ValueError("OpenAI client was not initialized")

    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=1000,
    )
    return response.choices[0].message.content or ""


def send_request(method: str, endpoint: str, **kwargs):
    """Send HTTP request to the environment server."""
    url = f"{API_BASE_URL}{endpoint}"
    response = requests.request(method, url, timeout=10, **kwargs)
    response.raise_for_status()
    return response.json()


def run_agent(task_id: str, max_steps: int = 20, model: Optional[str] = None) -> dict:
    """Run the agent on a specific task."""
    model = model or MODEL_NAME
    if _is_gemini_model(model):
        model = _resolve_gemini_model(model)
    openai_client: Optional[OpenAI] = None
    gemini_client: Optional[genai.Client] = None
    if _is_gemini_model(model):
        gemini_client = _get_gemini_client()
    else:
        openai_client = _get_openai_client()
    
    # Initialize episode
    print(f"\n{'='*60}")
    print(f"Starting task: {task_id}")
    print(f"Model: {model}")
    print(f"{'='*60}\n")
    
    obs = send_request("POST", "/reset", json={"task_id": task_id})
    episode_id = obs["episode_id"]
    max_steps = obs["max_steps"]
    
    print(f"Episode ID: {episode_id}")
    print(f"Max Steps: {max_steps}")
    print(f"\nTask: {obs['task_description']}\n")
    
    step_count = 0
    total_reward = 0.0
    actions_taken = []
    
    while step_count < max_steps:
        step_count += 1
        
        # Get current state
        state = send_request("GET", f"/state?episode_id={episode_id}")
        
        # Prepare prompt for LLM
        system_prompt = """You are an expert Linux DevOps engineer/SRE. 
Your job is to diagnose and fix broken systems using bash commands and file edits.
You are interacting with a simulated Linux environment.

Available actions:
1. bash_cmd: Execute a bash command
2. file_edit: Edit a file
3. submit: Submit when the task is complete

Respond in JSON format with this structure:
{
  "action_type": "bash_cmd" | "file_edit" | "submit",
  "command": "command to execute" (if bash_cmd),
  "file_path": "/path/to/file" (if file_edit),
  "file_content": "new file content" (if file_edit),
  "summary": "why you're taking this action"
}

Be strategic:
- Start by diagnosing the system
- Use ps, systemctl, curl, etc. to understand issues
- Fix the root cause
- Submit when done
"""
        
        user_prompt = f"""
Current system state:
- Task: {obs['task_description']}
- Step: {state['step_number']}/{state['max_steps']}
- Reward so far: {state['total_reward']:.3f}

System status:
{json.dumps(obs['system_state'], indent=2)}

Previous actions: {len(state['history'])} taken so far

History of commands:
{json.dumps(state['history'][-3:], indent=2) if state['history'] else 'None yet'}

What should I do next? Think step-by-step about what the issue is and how to fix it.
"""
        
        try:
            # Call LLM (Gemini SDK or OpenAI-compatible chat)
            response_text = _generate_action_text(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                openai_client=openai_client,
                gemini_client=gemini_client,
            )
            try:
                # Try to extract JSON from response
                if "```json" in response_text:
                    json_str = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    json_str = response_text.split("```")[1].split("```")[0]
                else:
                    json_str = response_text
                
                action_data = json.loads(json_str)
            except (json.JSONDecodeError, IndexError):
                print(f"Failed to parse LLM response: {response_text[:100]}")
                # Fallback to simple diagnosis
                action_data = {"action_type": "bash_cmd", "command": "ps aux"}
        
        except Exception as e:
            print(f"LLM error: {e}. Falling back to heuristic...")
            # Fallback heuristic actions
            if step_count == 1:
                action_data = {"action_type": "bash_cmd", "command": "systemctl status nginx"}
            else:
                action_data = {"action_type": "submit", "summary": "Diagnostics complete"}
        
        # Step in environment
        try:
            result = send_request("POST", "/step", json={
                "episode_id": episode_id,
                "action": action_data
            })
            
            obs = result["observation"]
            reward = result["reward"]
            done = result["done"]
            
            step_count = obs["step_number"]
            total_reward = reward["total_reward"]
            
            actions_taken.append(action_data)
            
            print(f"\nStep {step_count}/{max_steps}")
            print(f"Action: {action_data['action_type']}")
            if action_data.get("command"):
                print(f"Command: {action_data['command']}")
            elif action_data.get("file_path"):
                print(f"File: {action_data['file_path']}")
            
            print(f"Reward: {reward['step_reward']:+.3f} (total: {total_reward:.3f})")
            print(f"Info: {reward['explanation'][:100]}")
            
            if done:
                print(f"\n{'='*60}")
                print("EPISODE COMPLETE!")
                print(f"Final Reward: {total_reward:.3f}")
                print(f"Steps taken: {step_count}")
                print(f"{'='*60}\n")
                break
        
        except Exception as e:
            print(f"Step error: {e}")
            break
        
        # Small delay to avoid rate limiting
        time.sleep(0.5)
    
    # Grade the episode
    try:
        grade_result = send_request("POST", "/grader", json={"episode_id": episode_id})
        print(f"\nGrader Results:")
        print(f"Score: {grade_result['score']:.3f}/1.0")
        print(f"Breakdown: {json.dumps(grade_result['breakdown'], indent=2)}")
        print(f"Feedback: {grade_result['feedback']}")
    except Exception as e:
        print(f"Grading error: {e}")
    
    return {
        "task_id": task_id,
        "episode_id": episode_id,
        "final_reward": total_reward,
        "step_count": step_count,
        "actions": actions_taken,
    }


def main():
    parser = argparse.ArgumentParser(description="Run DevOpsEnv baseline agent")
    parser.add_argument("--task", default="task1", help="Task ID (task1, task2, or task3)")
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model name (default: env var MODEL_NAME). "
            "For Gemini, pass a real model ID like gemini-3-flash-preview "
            "or use --model gemini to auto-resolve to GEMINI_MODEL."
        ),
    )
    parser.add_argument("--api-url", default=None, help="API URL (default: env var API_BASE_URL)")
    parser.add_argument("--hf-token", default=None, help="HF token (default: env var HF_TOKEN)")
    parser.add_argument(
        "--openai-base-url",
        default=None,
        help="OpenAI-compatible base URL for non-OpenAI providers (for example Gemini OpenAI API)",
    )
    
    args = parser.parse_args()
    
    # Override env variables if provided
    global API_BASE_URL, MODEL_NAME, HF_TOKEN, OPENAI_BASE_URL
    if args.api_url:
        API_BASE_URL = args.api_url
    if args.model:
        MODEL_NAME = args.model
    if args.hf_token:
        HF_TOKEN = args.hf_token
    if args.openai_base_url:
        OPENAI_BASE_URL = args.openai_base_url
    
    try:
        result = run_agent(args.task, model=MODEL_NAME)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
