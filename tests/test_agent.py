import json
from types import SimpleNamespace

from dq_agent.agent import extract_json, run
from dq_agent.config import Settings
from dq_agent.synth.orders import generate


class FakeLLM:
    """Scripted replies; records what it was shown."""

    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def invoke(self, messages):
        self.seen.append(list(messages))
        return SimpleNamespace(content=self.replies.pop(0), usage_metadata={"total_tokens": 10})


def q(sql, h="h"):
    return json.dumps({"action": "query", "hypothesis": h, "sql": sql})


def test_extract_json_handles_fences_and_prose():
    assert extract_json('Sure!\n```json\n{"a": {"b": 1}}\n```') == {"a": {"b": 1}}
    assert extract_json("no json") is None


def test_loop_end_to_end(tmp_path):
    csv = generate(tmp_path)
    sql = "SELECT * FROM t WHERE ship_date < order_date"
    llm = FakeLLM(
        [
            "garbage",  # invalid reply -> recovered
            q("DROP TABLE t"),  # rejected -> error fed back
            q(f"SELECT count(*) FROM ({sql})"),
            json.dumps({"action": "conclude", "findings": [
                {"hypothesis": "ship_date >= order_date", "verdict": "supported",
                 "evidence_sql": sql, "violations": 30}]}),
        ]
    )  # fmt: skip
    rep = run(
        csv,
        llm,
        Settings(runs_dir=str(tmp_path / "runs"), min_queries=0, max_pushbacks=0, llm_cache=False),
    )
    assert rep["findings"][0]["verified_count"] == 30
    assert rep["tool_calls"] == 3 and rep["tokens"] == 40
    assert "ERROR" in llm.seen[2][-1]["content"]
    # LLM was only shown aggregates, never the raw table
    assert "O000001" not in json.dumps(llm.seen[0])
    trace = (tmp_path / "runs" / rep["run_id"] / "trace.jsonl").read_text()
    assert '"event": "query"' in trace


def test_budget_forces_conclusion(tmp_path):
    csv = generate(tmp_path)
    llm = FakeLLM([q("SELECT 1")] * 2 + ['{"action":"conclude","findings":[]}'])
    rep = run(
        csv,
        llm,
        Settings(runs_dir=str(tmp_path / "runs"), max_tool_calls=2, min_queries=0, max_pushbacks=0),
    )
    assert rep["tool_calls"] == 2 and rep["findings"] == []
    assert "Budget exhausted" in llm.seen[-1][-1]["content"]


def test_llm_failure_ends_cleanly(tmp_path):
    class Boom:
        def invoke(self, messages):
            raise TimeoutError("stalled")

    rep = run(
        generate(tmp_path),
        Boom(),
        Settings(runs_dir=str(tmp_path / "runs"), min_queries=0, max_pushbacks=0, llm_cache=False),
    )
    assert rep["findings"] == [] and "stalled" in rep["error"]


def test_recount_handles_aggregate_and_rowset(tmp_path):
    from dq_agent.agent import _recount
    from dq_agent.sql_tool import ReadOnlyQuery

    ro = ReadOnlyQuery(generate(tmp_path), max_rows=20)
    bad = "FROM t WHERE ship_date < order_date"
    assert _recount(ro, f"SELECT count(*) AS n {bad}") == 30  # aggregate evidence
    assert _recount(ro, f"SELECT * {bad};") == 30  # row set larger than the 20-row cap
    assert _recount(ro, f"SELECT * {bad} LIMIT 5") == 30  # sampling LIMIT ignored
    assert _recount(ro, "SELECT nonsense FROM t") is None
    assert _recount(ro, None) is None


def test_llm_is_pinned_and_seeded():
    from dq_agent.llm import get_llm

    llm = get_llm(Settings(_env_file=None, llm_provider="Nvidia", llm_cache=False))
    assert llm.seed == 42 and llm.temperature == 0
    assert llm.extra_body["provider"]["allow_fallbacks"] is False
    assert llm.extra_body["reasoning"] == {"effort": "minimal"}
    assert ":free" in llm.model_name and llm.model_name != "openrouter/free"


def test_conclude_gate_pushes_back_then_allows(tmp_path):
    csv = generate(tmp_path)
    done = '{"action":"conclude","findings":[]}'
    llm = FakeLLM([done, q("SELECT count(*) FROM t WHERE customer_id IS NULL"), done])
    cfg = Settings(runs_dir=str(tmp_path / "r"), min_queries=1, llm_cache=False, max_pushbacks=1)
    run(csv, llm, cfg)
    assert "Not done yet" in llm.seen[1][-1]["content"]
    assert "customer_id" in llm.seen[1][-1]["content"]


def test_unjustified_refutation_is_downgraded(tmp_path):
    csv = generate(tmp_path)
    sql = "SELECT * FROM t WHERE ship_date < order_date"
    concl = {"action": "conclude", "findings": [
        {"hypothesis": "h", "verdict": "refuted", "evidence_sql": sql},
        {"hypothesis": "h2", "verdict": "refuted", "evidence_sql": sql,
         "legitimate_explanation": "backdated by policy"}]}  # fmt: skip
    cfg = Settings(runs_dir=str(tmp_path / "r"), min_queries=0, max_pushbacks=0, llm_cache=False)
    f = run(csv, FakeLLM([json.dumps(concl)]), cfg)["findings"]
    assert f[0]["verdict"] == "insufficient" and f[0]["llm_verdict"] == "refuted"
    assert f[1]["verdict"] == "refuted"


def test_cached_llm_replays(tmp_path):
    from dq_agent.llm import CachedLLM

    inner = FakeLLM(["first", "second"])
    c = CachedLLM(inner, tmp_path / "c", "salt")
    msgs = [{"role": "user", "content": "x"}]
    assert c.invoke(msgs).content == "first"
    again = c.invoke(msgs)
    assert again.content == "first" and again.response_metadata["cache"] == "hit"
    assert c.invoke([{"role": "user", "content": "y"}]).content == "second"
    assert (c.hits, c.misses) == (1, 2)


def test_query_results_are_deterministic(tmp_path):
    from dq_agent.sql_tool import ReadOnlyQuery

    csv = generate(tmp_path)
    sql = "SELECT status, count(*) AS n FROM t GROUP BY status, quantity HAVING n > 0 LIMIT 15"
    outs = {json.dumps(ReadOnlyQuery(csv, dev_single_thread=True).run(sql).rows) for _ in range(5)}
    assert len(outs) == 1
