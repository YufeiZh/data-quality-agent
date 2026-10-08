from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="DQ_", extra="ignore")

    # LLM via OpenRouter (OpenAI-compatible API)
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "nvidia/nemotron-3-super-120b-a12b:free"  # pinned; never use the router
    llm_seed: int = 42
    llm_provider: str = ""  # optional: pin an OpenRouter upstream provider (no fallbacks)
    llm_temperature: float = 0.0
    llm_max_tokens: int = 6000  # caps hidden reasoning + answer; bounds per-call latency
    # Sent to OpenRouter as reasoning.effort for every model; "" disables sending it.
    llm_reasoning_effort: str = "minimal"
    llm_cache: bool = True  # record/replay LLM responses for exact reruns
    llm_timeout_s: float = 60.0  # per request; free models can stall
    llm_max_retries: int = 1

    # Budgets
    max_tool_calls: int = 20
    max_tokens_per_run: int = 100_000
    min_queries: int = 4  # earliest the agent may conclude
    max_pushbacks: int = 2
    max_wall_s: float = 300.0
    # DEV/REPRO ONLY. Forces the SQL sandbox to one thread so unordered/tied query results come
    # back in a stable order => identical prompts => LLM cache replays. Slower; OFF in real use.
    dev_single_thread_sandbox: bool = False
    query_timeout_s: float = 10.0
    query_max_rows: int = 200

    # Storage
    runs_dir: str = "runs"


def get_settings() -> Settings:
    return Settings()
