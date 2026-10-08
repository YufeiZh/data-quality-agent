import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from langchain_openai import ChatOpenAI

from .config import Settings, get_settings


def get_llm(settings: Settings | None = None):
    s = settings or get_settings()
    extra: dict = {}
    if s.llm_provider:
        extra["provider"] = {"order": [s.llm_provider], "allow_fallbacks": False}
    if s.llm_reasoning_effort:
        extra["reasoning"] = {"effort": s.llm_reasoning_effort}
    llm = ChatOpenAI(
        model=s.llm_model,
        api_key=s.openrouter_api_key or "missing",
        base_url=s.openrouter_base_url,
        temperature=s.llm_temperature,
        timeout=s.llm_timeout_s,
        max_retries=s.llm_max_retries,
        seed=s.llm_seed,
        max_tokens=s.llm_max_tokens,
        extra_body=extra or None,
    )
    if not s.llm_cache:
        return llm
    return CachedLLM(
        llm,
        Path(s.runs_dir) / "llm_cache",
        f"{s.llm_model}|{s.llm_seed}|{s.llm_temperature}|{s.llm_max_tokens}|{s.llm_reasoning_effort}",
    )


class CachedLLM:
    """Record/replay wrapper: identical (model, seed, temperature, messages) -> identical reply.

    Provider-side seeding is best-effort, so this is what makes reruns and evaluations exact.
    A cache hit costs no tokens against the API (recorded usage is replayed for accounting).
    """

    def __init__(self, inner, cache_dir: Path, salt: str):
        self.inner, self.dir, self.salt = inner, Path(cache_dir), salt
        self.hits = self.misses = 0

    def _key(self, messages) -> str:
        blob = json.dumps([self.salt, messages], sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def invoke(self, messages):
        path = self.dir / f"{self._key(messages)}.json"
        if path.exists():
            self.hits += 1
            d = json.loads(path.read_text())
            return SimpleNamespace(
                content=d["content"],
                usage_metadata=d["usage_metadata"],
                response_metadata={**d["response_metadata"], "cache": "hit"},
            )
        resp = self.inner.invoke(messages)
        self.misses += 1
        d = {
            "content": resp.content,
            "usage_metadata": {
                "total_tokens": (getattr(resp, "usage_metadata", None) or {}).get("total_tokens", 0)
            },
            "response_metadata": {
                "model_name": (getattr(resp, "response_metadata", None) or {}).get("model_name")
            },
        }
        self.dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(d))
        return SimpleNamespace(
            **{**d, "response_metadata": {**d["response_metadata"], "cache": "miss"}}
        )
