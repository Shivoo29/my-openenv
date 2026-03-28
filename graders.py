"""
Graders for DevOpsEnv tasks.

Deterministic scoring based on system state changes and action validity.
"""
from typing import Any, Dict, Tuple


def grade_task(task_id: str, episode_state: Dict[str, Any]) -> Tuple[float, Dict[str, float], str]:
    """
    Grade a completed task episode.
    
    Returns (score, breakdown, feedback)
    """
    if task_id == "task1":
        return grade_task1(episode_state)
    elif task_id == "task2":
        return grade_task2(episode_state)
    elif task_id == "task3":
        return grade_task3(episode_state)
    else:
        return 0.0, {}, "Unknown task"


def grade_task1(episode_state: Dict[str, Any]) -> Tuple[float, Dict[str, float], str]:
    """
    Grade Task 1: Restart Nginx.
    
    Success criteria:
    - nginx service is running (30%)
    - nginx config is valid (30%)
    - HTTP 200 response on port 80 (40%)
    """
    state_dict = episode_state.get("system_state", {})
    action_history = episode_state.get("action_history", [])
    
    breakdown = {
        "nginx_running": 0.0,
        "config_valid": 0.0,
        "http_200": 0.0,
    }
    
    # Check if nginx is running
    service_status = state_dict.get("service_status", {})
    if service_status.get("nginx") == "active":
        breakdown["nginx_running"] = 0.3
    
    # Check if config validation was attempted and passed
    config_valid = False
    for action in action_history:
        output = action.get("output", "")
        if output and ("syntax is ok" in str(output).lower() or "test is successful" in str(output).lower()):
            config_valid = True
            breakdown["config_valid"] = 0.3
            break
    
    # Check if HTTP 200 response was achieved
    http_ports = state_dict.get("http_ports_open", [])
    if 80 in http_ports:
        # Verify http 200 response was confirmed
        for action in action_history:
            output = action.get("output", "")
            cmd = action.get("command", "")
            if output and cmd and "OK" in str(output) and "curl" in str(cmd).lower():
                breakdown["http_200"] = 0.4
                break
    
    score = sum(breakdown.values())
    feedback = f"Task 1 Grading: nginx_running={breakdown['nginx_running']:.1f}, config_valid={breakdown['config_valid']:.1f}, http_200={breakdown['http_200']:.1f}"
    
    return min(score, 1.0), breakdown, feedback


def grade_task2(episode_state: Dict[str, Any]) -> Tuple[float, Dict[str, float], str]:
    """
    Grade Task 2: Fix Docker configuration.
    
    Success criteria:
    - docker-compose.yml was edited (25%)
    - docker-compose up -d was successful (25%)
    - Container is running (25%)
    - Service accessible on correct port (25%)
    """
    state_dict = episode_state.get("system_state", {})
    action_history = episode_state.get("action_history", [])
    files = state_dict.get("files", {})
    
    breakdown = {
        "file_edited": 0.0,
        "compose_ran": 0.0,
        "container_running": 0.0,
        "port_accessible": 0.0,
    }
    
    # Check if docker-compose.yml was edited correctly
    compose_file = "/srv/docker-compose.yml"
    if compose_file in files:
        content = files[compose_file]
        if content and "3000:3000" in str(content):
            breakdown["file_edited"] = 0.25
    
    # Check if docker-compose up -d was run
    for action in action_history:
        cmd = action.get("command")
        if cmd and "docker-compose up -d" in str(cmd):
            output = action.get("output", "")
            if output and ("done" in str(output).lower() or "created" in str(output).lower()):
                breakdown["compose_ran"] = 0.25
            break
    
    # Check if container is running
    containers = state_dict.get("docker_containers", [])
    if containers:
        for container in containers:
            if container.get("status") == "running" and "mockapi" in str(container.get("name", "")):
                breakdown["container_running"] = 0.25
                break
    
    # Check if port is correctly mapped
    if containers:
        for container in containers:
            if "3000:3000" in str(container.get("ports", "")):
                breakdown["port_accessible"] = 0.25
                break
    
    score = sum(breakdown.values())
    feedback = f"Task 2 Grading: file_edited={breakdown['file_edited']:.2f}, compose_ran={breakdown['compose_ran']:.2f}, container_running={breakdown['container_running']:.2f}, port_accessible={breakdown['port_accessible']:.2f}"
    
    return min(score, 1.0), breakdown, feedback


def grade_task3(episode_state: Dict[str, Any]) -> Tuple[float, Dict[str, float], str]:
    """
    Grade Task 3: Fix memory leak.
    
    Success criteria:
    - Process was killed (25%)
    - Code was fixed (removing the leak) (25%)
    - Service was restarted (25%)
    - Memory usage decreased (25%)
    """
    state_dict = episode_state.get("system_state", {})
    action_history = episode_state.get("action_history", [])
    files = state_dict.get("files", {})
    processes = state_dict.get("running_processes", [])
    
    breakdown = {
        "process_killed": 0.0,
        "code_fixed": 0.0,
        "service_restarted": 0.0,
        "memory_reduced": 0.0,
    }
    
    # Check if python process was killed
    has_python_leak = False
    if processes:
        has_python_leak = any(p.get("name") == "python3" and p.get("rss_mb", 512) > 1024 for p in processes)
    if not has_python_leak:
        # Process was killed
        breakdown["process_killed"] = 0.25
    
    # Check if code was fixed (removed the memory leak)
    app_file = "/opt/mockapi/app.py"
    if app_file in files:
        content = files[app_file]
        # Memory leak is the unbounded list append - check if it is fixed
        if content and ("request_cache.append" not in str(content) or "request_cache = []" not in str(content)):
            # If it has been removed or replaced with something better
            if "request_cache" not in str(content) or "# " in str(content):
                breakdown["code_fixed"] = 0.25
    
    # Check if service was restarted
    service_status = state_dict.get("service_status", {})
    if service_status.get("mockapi") == "active":
        # And there is a newer process
        for action in action_history:
            cmd = action.get("command", "")
            if cmd and "python3" in str(cmd) and ("start" in str(cmd) or "&" in str(cmd)):
                breakdown["service_restarted"] = 0.25
                break
    
    # Check if memory usage decreased
    initial_memory = 2048
    current_memory = state_dict.get("memory_usage_mb", 2048)
    if current_memory < initial_memory * 0.75:  # At least 25% improvement
        breakdown["memory_reduced"] = 0.25
    
    score = sum(breakdown.values())
    feedback = f"Task 3 Grading: process_killed={breakdown['process_killed']:.2f}, code_fixed={breakdown['code_fixed']:.2f}, service_restarted={breakdown['service_restarted']:.2f}, memory_reduced={breakdown['memory_reduced']:.2f}"
    
    return min(score, 1.0), breakdown, feedback
