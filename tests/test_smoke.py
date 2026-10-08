import duckdb
import langgraph.graph  # noqa: F401

from dq_agent.config import get_settings
from dq_agent.llm import get_llm


def test_settings_defaults():
    assert get_settings().openrouter_base_url.startswith("https://openrouter.ai")


def test_llm_constructs():
    assert get_llm() is not None


def test_duckdb_works():
    assert duckdb.sql("select 1").fetchone() == (1,)
