# CLAUDE.md

## TOP RULE — never run LLM calls or experiments without asking first

- **Do not run anything that calls the LLM** (`dq-agent run`, any script that invokes `get_llm()`,
  ad-hoc API probes, "quick tests", reruns to check reproducibility) **unless the user has
  explicitly approved that specific run in the current conversation.** It burns tokens and time.
- This holds even if the run seems like an obvious next step, and even right after a fix.
  Approval for one run does not carry over to the next one.
- When asked to **diagnose/identify an issue, only diagnose.** Use existing artifacts (traces in
  `runs/*/trace.jsonl`, `report.json`, logs, code reading) and the free offline tests. Report the
  cause, propose the fix and the verification run, then **wait for a yes**.
- Do not run experiments to "figure out" a problem before the evidence already on disk has been
  exhausted and the user has agreed to the experiment.
- Offline checks are fine without asking: `uv run pytest` (uses `FakeLLM`, no API calls),
  `uv run ruff check .`, `uv run ruff format .`, reading files/traces.
- If unsure whether something calls the LLM, assume it does and ask.

## Project

Agentic data quality MVP, see `DATA_QUALITY_AGENT_MVP.md`. Python 3.12, uv, LangGraph, DuckDB,
OpenRouter free model (pinned in `config.py`, not `.env`; never the `openrouter/free` router).

- Setup/tests: `uv sync`, `uv run pytest`, `uv run ruff check .`
- LLM responses are cached in `runs/llm_cache`; any change to prompts, model, seed, temperature,
  `max_tokens` or reasoning effort changes the cache key and causes real (billed) calls.
- Run long jobs in the background with live progress, never as a silent foreground command.
