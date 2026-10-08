from langchain_openai import ChatOpenAI

from .config import Settings, get_settings


def get_llm(settings: Settings | None = None) -> ChatOpenAI:
    s = settings or get_settings()
    return ChatOpenAI(
        model=s.llm_model,
        api_key=s.openrouter_api_key or "missing",
        base_url=s.openrouter_base_url,
        temperature=s.llm_temperature,
    )
