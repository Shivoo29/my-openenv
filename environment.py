"""
Core DevOpsEnv environment logic.

Simulates a broken Linux server with:
- Task 1: Crashed Nginx service needing restart
- Task 2: Misconfigured Docker container
- Task 3: Memory leak in Python mock API

Manages episode lifecycle:
  reset() → Observation
  step(action) → StepResult
  get_state() → State
  grade() → (score, breakdown, feedback)
"""
from __future__ import annotations

import uuid
import json
import re
from typing import Any, Dict, Optional, Tuple, List

from data import TASK_META
from graders import grade_task
from models import (
    Action,
    Observation,
    Reward,
    State,
    StepResult,
    SystemState,
)

# In-memory store: episode_id → EpisodeState dict
_EPISODES: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Mock filesystem and system state
# ---------------------------------------------------------------------------

def _create_initial_state_task1() -> Dict[str, Any]:
    """Task 1: Nginx is crashed."""
    return {
        "running_processes": [
            {"pid": 100, "name": "systemd"},
            {"pid": 105, "name": "sshd"},
            # nginx NOT running
        ],
        "service_status": {
            "nginx": "inactive",
            "docker": "active",
            "mockapi": "active",
        },
        "http_ports_open": [8080],  # 80 is down
        "docker_containers": [],
        "logs": "2026-03-29 01:30:00 nginx crashed\nCore dump detected.\n",
        "files": {
            NGINX_CONFIG_PATH: """
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    sendfile on;
    keepalive_timeout 65;

    server {
        listen 80 default_server;
        server_name _;
        location / {
            return 200 "OK\\n";
        }
    }
}""",
            "/etc/systemd/system/nginx.service": """
[Unit]
Description=The NGINX HTTP and reverse proxy server
After=network.target

[Service]
Type=forking
PIDFile=/var/run/nginx.pid
ExecStartPre=/usr/sbin/nginx -t
ExecStart=/usr/sbin/nginx
ExecReload=/bin/kill -s HUP $MAINPID
ExecStop=/bin/kill -s QUIT $MAINPID
PrivateTmp=true

[Install]
WantedBy=multi-user.target""",
        },
        "cpu_usage": 45.2,
        "memory_usage_mb": 256,
    }


def _create_initial_state_task2() -> Dict[str, Any]:
    """Task 2: Docker misconfigured."""
    return {
        "running_processes": [
            {"pid": 100, "name": "systemd"},
            {"pid": 105, "name": "sshd"},
            {"pid": 200, "name": "dockerd"},
        ],
        "service_status": {
            "nginx": "active",
            "docker": "active",
            "mockapi": "inactive",
        },
        "http_ports_open": [80],
        "docker_containers": [
            {"id": "abc123", "name": "mockapi-svc", "status": "running", "ports": "8000->3000/tcp"}
        ],
        "logs": "docker: port 3000 already in use\n",
        "files": {
            "/srv/docker-compose.yml": """
version: '3.8'
services:
  mockapi:
    image: mockapi:latest
    ports:
      - "3000:3000"
    environment:
      - PORT=3000
    volumes:
      - ./app.py:/app/app.py""",
        },
        "cpu_usage": 62.0,
        "memory_usage_mb": 1024,
    }


def _create_initial_state_task3() -> Dict[str, Any]:
    """Task 3: Memory leak in mock API."""
    return {
        "running_processes": [
            {"pid": 100, "name": "systemd"},
            {"pid": 105, "name": "sshd"},
            {"pid": 300, "name": "python3", "rss_mb": 2048, "user": "appuser"},  # MEMORY LEAK
        ],
        "service_status": {
            "nginx": "active",
            "docker": "active",
            "mockapi": "active",
        },
        "http_ports_open": [80, 5000],
        "docker_containers": [],
        "logs": (
            "2026-03-29 01:45:00 mockapi started\n"
            "2026-03-29 01:46:00 memory usage: 512 MB\n"
            "2026-03-29 01:47:00 memory usage: 1024 MB\n"
            "2026-03-29 01:48:00 memory usage: 1536 MB (WARNING: HIGH)\n"
            "2026-03-29 01:49:00 memory usage: 2048 MB (CRITICAL)\n"
        ),
        "files": {
            "/opt/mockapi/app.py": """
import json
from flask import Flask

app = Flask(__name__)

# BUG: This list grows unbounded
request_cache = []

@app.route('/api/data', methods=['GET'])
def get_data():
    data = {"timestamp": 123456, "value": 42}
    request_cache.append(data)  # MEMORY LEAK!
    return json.dumps(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
""",
        },
        "cpu_usage": 85.5,
        "memory_usage_mb": 2048,
    }


NGINX_CONFIG_PATH = "/etc/nginx/nginx.conf"
DOCKER_COMPOSE_PATH = "/srv/docker-compose.yml"
MOCK_API_PATH = "/opt/mockapi/app.py"


def _build_system_state(task_id: str, ep_state: Dict[str, Any]) -> SystemState:
    """Build a SystemState object from episode state."""
    state_dict = ep_state["system_state"]
    return SystemState(
        task_id=task_id,
        available_commands=["systemctl", "nginx", "docker", "curl", "ps", "cat", "vim"],
        filesystem_snapshot=json.dumps({
            k: v for k, v in state_dict.get("files", {}).items()
        }),
        running_processes=state_dict.get("running_processes", []),
        service_status=state_dict.get("service_status", {}),
        logs=state_dict.get("logs", ""),
        http_ports_open=state_dict.get("http_ports_open", []),
        docker_containers=state_dict.get("docker_containers", []),
        cpu_usage=state_dict.get("cpu_usage", 0.0),
        memory_usage_mb=state_dict.get("memory_usage_mb", 0),
    )


# ---------------------------------------------------------------------------
# Dynamic execution simulation
# ---------------------------------------------------------------------------

def _simulate_bash_cmd(cmd: str, task_id: str, ep_state: Dict[str, Any]) -> str:
    """Simulate bash command execution."""
    state_dict = ep_state["system_state"]
    lower_cmd = cmd.lower()

    # Task 1: Nginx commands
    if task_id == "task1":
        if "systemctl restart nginx" in lower_cmd or "systemctl start nginx" in lower_cmd:
            state_dict["service_status"]["nginx"] = "active"
            state_dict["running_processes"].append({"pid": 999, "name": "nginx"})
            state_dict["http_ports_open"] = [80]
            return "Job for nginx.service started successfully."
        elif "systemctl status nginx" in lower_cmd:
            if state_dict["service_status"]["nginx"] == "active":
                return "● nginx.service - NGINX HTTP Server\n   Loaded: loaded (/etc/systemd/system/nginx.service)\n   Active: active (running)"
            return "● nginx.service - NGINX HTTP Server\n   Active: inactive (dead)"
        elif "nginx -t" in lower_cmd:
            return "nginx: the configuration file /etc/nginx/nginx.conf syntax is ok\nnginx: configuration file /etc/nginx/nginx.conf test is successful"
        elif "curl http://localhost:80" in lower_cmd or "curl http://localhost" in lower_cmd:
            if 80 in state_dict["http_ports_open"]:
                return "OK"
            return "curl: (7) Failed to connect to localhost port 80: Connection refused"

    # Task 2: Docker commands
    elif task_id == "task2":
        if "docker-compose up -d" in lower_cmd:
            if DOCKER_COMPOSE_PATH in state_dict["files"]:
                compose_content = state_dict["files"][DOCKER_COMPOSE_PATH]
                # Check if port is now correct
                if "3000:3000" in compose_content:
                    state_dict["docker_containers"] = [
                        {"id": "xyz789", "name": "mockapi-svc", "status": "running", "ports": "3000:3000/tcp"}
                    ]
                    state_dict["service_status"]["mockapi"] = "active"
                    return "Creating mockapi ... done"
            return "ERROR: docker-compose.yml not found or invalid"
        elif "docker ps" in lower_cmd:
            if state_dict["docker_containers"]:
                return "\n".join([f"{c['id']} {c['name']} {c['status']}" for c in state_dict["docker_containers"]])
            return "No containers running"

    # Task 3: Process/memory commands
    elif task_id == "task3":
        if "ps aux" in lower_cmd or "ps aux grep python" in lower_cmd:
            output = ""
            for proc in state_dict["running_processes"]:
                if proc.get("name") == "python3":
                    output += f"appuser {proc['pid']} 85.5 {proc.get('rss_mb', 512)} python3 /opt/mockapi/app.py\n"
            return output if output else "No python processes found"
        elif "kill" in lower_cmd:
            if "300" in lower_cmd or "python" in lower_cmd:
                state_dict["running_processes"] = [p for p in state_dict["running_processes"] if p.get("name") != "python3"]
                state_dict["service_status"]["mockapi"] = "inactive"
                return "Process killed"
            return "Process not found"
        elif "python3 /opt/mockapi/app.py &" in lower_cmd or "python3 /opt/mockapi/app.py" in lower_cmd:
            state_dict["running_processes"].append({"pid": 301, "name": "python3", "rss_mb": 128, "user": "appuser"})
            state_dict["service_status"]["mockapi"] = "active"
            state_dict["http_ports_open"] = [80, 5000]
            return "Application started"

    return f"Command '{cmd}' executed (simulated)"


def _simulate_file_edit(file_path: str, new_content: str, ep_state: Dict[str, Any]) -> str:
    """Simulate file editing."""
    state_dict = ep_state["system_state"]
    
    if file_path not in state_dict.get("files", {}):
        return f"ERROR: File {file_path} not found"

    # Detect task 2: Check docker-compose.yml fix
    if file_path == DOCKER_COMPOSE_PATH and "3000:3000" in new_content:
        state_dict["files"][file_path] = new_content
        return f"File {file_path} updated successfully"

    # Detect task 3: Check mock API fix
    elif file_path == MOCK_API_PATH and "request_cache = []" not in new_content:
        # Verify fix removes the memory leak
        state_dict["files"][file_path] = new_content
        return f"File {file_path} patched successfully"

    state_dict["files"][file_path] = new_content
    return f"File {file_path} edited"


# ---------------------------------------------------------------------------
# Reward calculation
# ---------------------------------------------------------------------------

def _calculate_step_reward(task_id: str, action: Action, ep_state: Dict[str, Any]) -> Tuple[float, str]:
    """Calculate reward based on action and task."""
    base_step_cost = -0.01
    reward = base_step_cost

    if action.action_type == "bash_cmd":
        cmd = action.command or ""
        reward += 0.05
        explanation = f"Executed: {cmd[:50]}"
        return reward, explanation

    elif action.action_type == "file_edit":
        reward += 0.03
        explanation = f"Edited: {action.file_path}"
        return reward, explanation

    elif action.action_type == "submit":
        reward += 0.1
        explanation = "Episode submitted for grading"
        return reward, explanation

    return reward, "Step taken"


# ---------------------------------------------------------------------------
# Core API functions
# ---------------------------------------------------------------------------

def reset(task_id: str) -> Observation:
    """Create a new episode for the given task."""
    if task_id not in TASK_META:
        raise ValueError(f"Unknown task_id {task_id!r}. Valid: {list(TASK_META)}")

    meta = TASK_META[task_id]
    
    # Initialize system state based on task
    if task_id == "task1":
        initial_sys_state = _create_initial_state_task1()
    elif task_id == "task2":
        initial_sys_state = _create_initial_state_task2()
    elif task_id == "task3":
        initial_sys_state = _create_initial_state_task3()
    else:
        initial_sys_state = {}

    episode_id = str(uuid.uuid4())
    _EPISODES[episode_id] = {
        "task_id": task_id,
        "step_number": 0,
        "max_steps": meta["max_steps"],
        "done": False,
        "total_reward": 0.0,
        "action_history": [],
        "final_score": None,
        "system_state": initial_sys_state,
    }

    system_state = _build_system_state(task_id, _EPISODES[episode_id])

    return Observation(
        task_id=task_id,
        task_description=meta["description"],
        episode_id=episode_id,
        system_state=system_state,
        thread_history=[],
        available_actions=meta["available_actions"],
        step_number=0,
        max_steps=meta["max_steps"],
        hint="Start by diagnosing the system state with basic commands.",
    )


def step(episode_id: str, action: Action) -> StepResult:
    """Advance the episode by one step."""
    ep = _EPISODES.get(episode_id)
    if ep is None:
        raise KeyError(f"Episode {episode_id} not found")

    if ep["done"]:
        raise ValueError(f"Episode {episode_id} is already done.")

    task_id = ep["task_id"]
    meta = TASK_META[task_id]

    ep["step_number"] += 1
    ep["action_history"].append(action.model_dump())

    # Execute action
    if action.action_type == "bash_cmd":
        cmd_output = _simulate_bash_cmd(action.command or "", task_id, ep)
        ep["action_history"][-1]["output"] = cmd_output
    elif action.action_type == "file_edit":
        edit_result = _simulate_file_edit(action.file_path or "", action.file_content or "", ep)
        ep["action_history"][-1]["result"] = edit_result

    # Determine if done
    done = False
    if action.action_type == "submit":
        done = True
    elif ep["step_number"] >= ep["max_steps"]:
        done = True

    # Calculate reward
    step_reward, explanation = _calculate_step_reward(task_id, action, ep)

    # Apply grader bonus when done
    if done:
        final_score, breakdown, grader_feedback = grade_task(task_id, ep)
        ep["final_score"] = final_score
        bonus = final_score * 0.5
        step_reward += bonus
        explanation += f" | Grader score: {final_score:.3f} (+{bonus:.3f} bonus)"
    else:
        final_score = None

    ep["total_reward"] = round(ep["total_reward"] + step_reward, 4)
    ep["done"] = done

    # Build observation
    system_state = _build_system_state(task_id, ep)
    thread_history = [
        {"role": "agent", "content": str(a)} for a in ep["action_history"]
    ]

    obs = Observation(
        task_id=task_id,
        task_description=meta["description"],
        episode_id=episode_id,
        system_state=system_state,
        thread_history=thread_history,
        available_actions=meta["available_actions"] if not done else [],
        step_number=ep["step_number"],
        max_steps=ep["max_steps"],
        hint=None if done else "Continue diagnosing and fixing the issue.",
    )

    reward = Reward(
        step_reward=round(step_reward, 4),
        total_reward=ep["total_reward"],
        explanation=explanation,
    )

    info = {"step": ep["step_number"]}
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

