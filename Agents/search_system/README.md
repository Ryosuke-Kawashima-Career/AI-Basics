# Search AI Agent System

## How to run

```bash
uv run python -m src.main
```

A2A servers:

```bash
uv run python -m agents.websearcher.server
uv run python -m agents.planner.server
uv run python -m agents.critics.server
```

or

```bash
uv run uvicorn agents.websearcher.server:a2a_app --host localhost --port 8101
```

## Run the whole system

```bash
uv run python -m src.run_all "Explain RAG vs fine-tuning"
```
