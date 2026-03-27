# SETUP.md — Local Development Guide

## Prerequisites

- Python 3.10+ ([download](https://www.python.org/downloads/))
- Git
- Docker (optional, for containerised run)
- An OpenAI API key (optional, only for the LLM baseline)

---

## 1. Clone the repository

```bash
git clone https://github.com/Shivoo29/dummy_1.git
cd dummy_1
git checkout claude/openenv-ai-agent-environment-qJ9pB
```

---

## 2. Create a virtual environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Run the server

```bash
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

- API: http://localhost:7860
- Interactive docs (Swagger UI): http://localhost:7860/docs
- ReDoc: http://localhost:7860/redoc

---

## 5. Quick smoke test

```bash
# Health check
curl http://localhost:7860/health

# List tasks
curl http://localhost:7860/tasks

# Start a task1 episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1", "ticket_index": 0}'

# The response contains an episode_id — use it below
EPISODE_ID="<paste episode_id here>"

# Submit a classification action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d "{\"episode_id\": \"$EPISODE_ID\", \"action\": {\"action_type\": \"classify\", \"category\": \"billing\", \"priority\": \"high\"}}"

# Submit to close the episode
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d "{\"episode_id\": \"$EPISODE_ID\", \"action\": {\"action_type\": \"submit\"}}"

# Grade the episode
curl -X POST http://localhost:7860/grader \
  -H "Content-Type: application/json" \
  -d "{\"episode_id\": \"$EPISODE_ID\"}"
```

---

## 6. Run the baseline

### Heuristic baseline (no API key required)

```bash
# Single ticket (ticket_index 0)
python baseline.py --mode heuristic

# All 5 tickets per task, averaged
python baseline.py --mode heuristic --all-tickets
```

Expected output:
```
task1: 0.8600  (scores: [1.0, 1.0, 1.0, 1.0, 0.3])
task2: 0.5614  (scores: [0.8, 0.386, 0.45, 0.7, 0.471])
task3: 0.9895  (scores: [1.0, 0.992, 0.961, 0.994, 1.0])
OVERALL AVERAGE: 0.8036
```

### LLM baseline (requires OpenAI API key)

```bash
export OPENAI_API_KEY="sk-..."          # macOS/Linux
# $env:OPENAI_API_KEY="sk-..."          # Windows PowerShell

python baseline.py --mode llm --model gpt-4o-mini
python baseline.py --mode llm --model gpt-4o-mini --all-tickets
```

---

## 7. Run with Docker

```bash
# Build
docker build -t supportenv .

# Run (no API key needed for heuristic mode)
docker run -p 7860:7860 supportenv

# Run with OpenAI key for LLM baseline
docker run -p 7860:7860 -e OPENAI_API_KEY="sk-..." supportenv
```

---

## 8. Project layout

```
dummy_1/
├── app.py            FastAPI server — all HTTP endpoints
├── environment.py    Episode lifecycle: reset / step / state / grade
├── graders.py        Deterministic graders for all 3 tasks
├── data.py           15 pre-defined tickets + ground truth answers
├── models.py         Pydantic typed models (Observation, Action, Reward…)
├── baseline.py       Heuristic + LLM baseline inference scripts
├── openenv.yaml      OpenEnv spec metadata
├── Dockerfile        HF Spaces-compatible container (port 7860)
├── requirements.txt  Python dependencies
├── README.md         Full environment documentation
└── SETUP.md          This file
```

---

## 9. Key files to edit when extending

| What you want to change | File to edit |
|------------------------|-------------|
| Add / modify tickets | `data.py` — `TASK1/2/3_TICKETS` lists |
| Change grader weights | `graders.py` — `grade_task1/2/3()` |
| Add a new task | `data.py` (add task meta) + `graders.py` + `app.py` (`_ACTION_SCHEMAS`) |
| Change reward shaping | `environment.py` — `_step_reward_task*` functions and constants |
| Add an endpoint | `app.py` |
| Change typed models | `models.py` |

---

## 10. Deploy to Hugging Face Spaces

1. Create a new Space at https://huggingface.co/new-space
   - SDK: **Docker**
   - Visibility: Public
2. Add the HF Space as a remote:
   ```bash
   git remote add hf https://huggingface.co/spaces/<your-username>/<space-name>
   ```
3. Push:
   ```bash
   git push hf claude/openenv-ai-agent-environment-qJ9pB:main
   ```
4. The Space auto-builds from the `Dockerfile` and exposes port 7860.

---

## 11. Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Only for LLM baseline | Your OpenAI API key |
| `PORT` | No (default 7860) | Override server port |

---

## 12. Running tests

```bash
python -c "
import environment as env
from models import Action

# Verify all 3 tasks reset and grade correctly
for task_id in ['task1', 'task2', 'task3']:
    for i in range(5):
        obs = env.reset(task_id, i)
        env.step(obs.episode_id, Action(action_type='submit'))
        gr = env.grade(obs.episode_id)
        assert 0.0 <= gr.score <= 1.0, f'Score out of range: {gr.score}'
        print(f'{task_id} ticket[{i}]: score={gr.score:.4f} OK')

print('All tests passed.')
"
```