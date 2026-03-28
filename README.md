---
title: DevOpsEnv
emoji: 🛠️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
  - devops
  - sre
  - troubleshooting
  - agent-evaluation
pinned: false
---

# DevOpsEnv

DevOpsEnv is a practice environment where an agent acts like a junior SRE.

In each episode, the agent gets a broken Linux-like system and must fix it by:
- Running shell commands
- Editing files
- Submitting when the fix is done

The server gives rewards during the episode and a final score at the end.

## What It Simulates (Simple)

There are 3 tasks:
- Task 1: Nginx is down. Bring service back and verify HTTP is OK.
- Task 2: Docker compose port mapping is wrong. Fix and redeploy.
- Task 3: Python API has memory leak behavior. Diagnose and reduce memory usage.

## How It Works

Step by step:
1. Call POST /reset with task_id.
2. You get episode_id plus current system_state.
3. Call POST /step with an action.
4. Repeat steps until done, or send action_type submit.
5. Call POST /grader to get final score and breakdown.

Main endpoints:
- GET /health
- GET /tasks
- POST /reset
- POST /step
- GET /state
- POST /grader

## Action Types

- bash_cmd: Run a command like systemctl status nginx
- file_edit: Replace content of a file path
- submit: End episode and grade

## Quick Start (Normal)

### 1) Install

Windows PowerShell:

python -m pip install -r requirements.txt

### 2) Start server

python -m uvicorn app:app --host 0.0.0.0 --port 7860

### 3) Check health

In another terminal:

Invoke-WebRequest -Uri "http://127.0.0.1:7860/health" -UseBasicParsing

If working, response includes status: healthy.

### 4) Run built-in integration test

python test_integration.py

If working, you should see all 3 tasks run and a final success message.

## Minimal API Example (Normal)

PowerShell example:

$reset = Invoke-WebRequest -Uri "http://127.0.0.1:7860/reset" -Method POST -ContentType "application/json" -Body '{"task_id":"task1"}' | Select-Object -ExpandProperty Content | ConvertFrom-Json
$episodeId = $reset.episode_id

$step = @{
  episode_id = $episodeId
  action = @{
    action_type = "bash_cmd"
    command = "systemctl restart nginx"
  }
} | ConvertTo-Json -Depth 5

Invoke-WebRequest -Uri "http://127.0.0.1:7860/step" -Method POST -ContentType "application/json" -Body $step

## Test With LLM (OpenAI Key)

1) Keep API server running.
2) Set key and run inference:

PowerShell:

$env:OPENAI_API_KEY = "your-openai-key"
python inference.py --task task1 --model gpt-4o-mini

You should see step logs, rewards, and a grader score.

## Test With Gemini API Key

inference.py now supports OpenAI-compatible base URLs.

Use Gemini via OpenAI-compatible endpoint:

PowerShell:

$env:GEMINI_API_KEY = "your-gemini-key"
$env:OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
python inference.py --task task1 --model gemini-2.5-flash

Notes:
- You can also use OPENAI_API_KEY instead of GEMINI_API_KEY.
- If your model name is unavailable, switch to a Gemini model enabled on your key.
- Keep the environment server running at http://127.0.0.1:7860 (or pass --api-url).

## Docker

Build:

docker build -t devopsenv .

Run:

docker run -p 7860:7860 devopsenv

Then open:
- http://127.0.0.1:7860/health
- http://127.0.0.1:7860/docs

## Project Files

- app.py: FastAPI API
- environment.py: episode logic and simulator
- graders.py: deterministic scoring
- data.py: task metadata
- models.py: Pydantic schemas
- inference.py: LLM baseline runner
- test_integration.py: local end-to-end check

## Troubleshooting

- Port already in use:
  - change server port or stop old process.
- 400/404 from API:
  - check episode_id and task_id values.
- LLM errors:
  - verify API key, model name, and OPENAI_BASE_URL for Gemini.
