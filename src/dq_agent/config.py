from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="DQ_", extra="ignore")

    # LLM via OpenRouter (OpenAI-compatible API)
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "openrouter/free"
    llm_temperature: float = 0.0

    # Budgets
    max_tool_calls: int = 20
    max_tokens_per_run: int = 100_000
    query_timeout_s: float = 10.0
    query_max_rows: int = 200

    # Storage
    runs_dir: str = "runs"


def get_settings() -> Settings:
    return Settings()
