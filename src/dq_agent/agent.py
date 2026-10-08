"""Single-agent investigation loop (LangGraph).

profile -> plan -> [query -> plan]* -> report

The LLM replies in plain JSON (no native tool-calling needed, which free models handle poorly):
  {"action": "query", "hypothesis": "...", "sql": "SELECT ..."}
  {"action": "conclude", "findings": [{hypothesis, verdict, evidence_sql, summary, ...}]}
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Protocol, TypedDict

import sqlglot
from langgraph.graph import END, START, StateGraph
from sqlglot import exp

from .config import Settings, get_settings
from .profiler import Profiler
from .sql_tool import ReadOnlyQuery, SQLRejected
from .trace import Trace

SYSTEM = """You are a data quality investigator. You never see the full table; you see a profile \
summary and can run read-only DuckDB SQL against one table named `t`.

Goal: beyond the profiler's signals, hypothesise IMPLICIT constraints (cross-column logic, \
date ordering, arithmetic relations, unit consistency) that the data should satisfy, then test \
each with SQL on the full table. A hypothesis is:
- "supported": violations exist and are not explained by legitimate business behaviour
- "refuted": few/no violations, or violations are plainly legitimate (say why)
- "insufficient": cannot tell
Also explain the likely root cause of the profiler signals you consider real. Do not label \
legitimate rare events (e.g. large but consistent bulk orders) as defects.

Reply with ONE JSON object and nothing else. Either
{"action":"query","hypothesis":"<what you are testing>","sql":"<one SELECT on t>"}
or, when done (or told the budget is exhausted),
{"action":"conclude","findings":[{"hypothesis":"...","verdict":"supported|refuted|insufficient",\
"columns":["..."],"legitimate_explanation":"<only if refuted with violations>",\
"violations":<int or null>,"evidence_sql":"<SELECT returning offending rows>",\
"root_cause":"...","suggested_fix":"..."}]}

Rules:
- Queries return at most 20 rows; prefer aggregates (count(*)) and small samples.
- Investigate EVERY profiler signal (decide if it is a real defect or legitimate) and ALSO test at \
least three cross-column hypotheses of your own (e.g. date ordering, arithmetic relations).
- Give exactly one finding per hypothesis/signal you examined. "violations" must be the exact \
count of rows in evidence_sql, which must be a plain SELECT of the offending rows (include the \
table's identifier column; no LIMIT, no aggregates).
- A "refuted" finding that still has violations > 0 MUST include "legitimate_explanation"."""


class LLM(Protocol):
    def invoke(self, messages: list[Any]) -> Any: ...


class State(TypedDict, total=False):
    messages: list[dict]
    profile: dict
    tool_calls: int
    tokens: int
    last: dict | None
    findings: list[dict]
    done: bool
    pushbacks: int
    queried: list[str]
    error: str | None


def extract_json(text: str) -> dict | None:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            depth += (text[i] == "{") - (text[i] == "}")
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    break
        start = text.find("{", start + 1)
    return None


def _served(resp) -> str | None:
    return (getattr(resp, "response_metadata", None) or {}).get("model_name")


def build_graph(
    path: str | Path,
    llm: LLM,
    settings: Settings | None = None,
    run_id: str | None = None,
    verbose: bool = False,
):
    s = settings or get_settings()
    run_id = run_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    trace = Trace(s.runs_dir, run_id, verbose=verbose)
    t0 = time.monotonic()
    profiler = Profiler(Path(path).stem, path)
    ro = ReadOnlyQuery(
        path,
        max_rows=20,
        timeout_s=s.query_timeout_s,
        dev_single_thread=s.dev_single_thread_sandbox,
    )

    def profile(state: State) -> State:
        prof = profiler.run().summary()
        trace.log(
            "profile", n_signals=len(prof["signals"]), model=s.llm_model, seed=s.llm_seed,
            dev_single_thread_sandbox=s.dev_single_thread_sandbox,
            provider=s.llm_provider or None, temperature=s.llm_temperature,
        )  # fmt: skip
        brief = json.dumps(
            {
                "n_rows": prof["n_rows"],
                "columns": [
                    {k: v for k, v in c.items() if v not in (None, {})} for c in prof["columns"]
                ],
                "signals": [
                    {k: sg[k] for k in ("evidence_id", "kind", "columns", "count", "description")}
                    for sg in prof["signals"]
                ],
            }
        )
        return {
            "profile": prof,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Profile summary:\n{brief}\n\nBegin."},
            ],
            "tool_calls": 0,
            "tokens": 0,
            "findings": [],
            "pushbacks": 0,
            "queried": [],
            "done": False,
        }

    def plan(state: State) -> State:
        msgs = list(state["messages"])
        over = (
            state["tool_calls"] >= s.max_tool_calls
            or state["tokens"] >= s.max_tokens_per_run
            or time.monotonic() - t0 > s.max_wall_s
        )
        if over:
            msgs.append({"role": "user", "content": "Budget exhausted. Conclude now."})
        try:
            resp = llm.invoke(msgs)
        except (
            Exception
        ) as e:  # timeout, rate limit, provider error: stop cleanly with what we have
            trace.log("llm_error", error=f"{type(e).__name__}: {str(e)[:300]}")
            return {"messages": msgs, "done": True, "last": None, "error": str(e)[:300]}
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        used = (getattr(resp, "usage_metadata", None) or {}).get("total_tokens", 0)
        tokens = state["tokens"] + used
        meta = getattr(resp, "response_metadata", None) or {}
        trace.log("llm", content=text, tokens=used, served_model=_served(resp),
                  finish_reason=meta.get("finish_reason"), cache=meta.get("cache"))  # fmt: skip
        msgs.append({"role": "assistant", "content": text})
        action = extract_json(text)
        if action and action.get("action") == "conclude":
            gap = _coverage_gap(state, profile_signals(state), s)
            if gap and not over and state["pushbacks"] < s.max_pushbacks:
                trace.log("pushback", reason=gap)
                msgs.append(
                    {"role": "user", "content": f"Not done yet: {gap} Continue investigating."}
                )
                return {"messages": msgs, "tokens": tokens, "pushbacks": state["pushbacks"] + 1,
                        "last": None}  # fmt: skip
            return {"messages": msgs, "tokens": tokens, "findings": action.get("findings", []),
                    "done": True, "last": None}  # fmt: skip
        if over:  # model ignored the instruction to stop
            return {"messages": msgs, "tokens": tokens, "done": True, "last": None}
        if not action or action.get("action") != "query" or not action.get("sql"):
            trace.log("invalid_reply")
            msgs.append(
                {
                    "role": "user",
                    "content": "Invalid reply. Output exactly one plain JSON object as specified "
                    "(no tool-call syntax, no prose).",
                }
            )
            return {"messages": msgs, "tokens": tokens, "tool_calls": state["tool_calls"] + 1,
                    "last": None}  # fmt: skip
        return {"messages": msgs, "tokens": tokens, "last": action}

    def query(state: State) -> State:
        a = state["last"]
        try:
            out = json.dumps(ro.run(a["sql"]).to_dict(), default=str)
            trace.log("query", hypothesis=a.get("hypothesis"), sql=a["sql"], ok=True)
        except SQLRejected as e:
            out = f"ERROR: {e}"
            trace.log("query", hypothesis=a.get("hypothesis"), sql=a["sql"], ok=False, error=str(e))
        msgs = state["messages"] + [{"role": "user", "content": f"Result:\n{out[:6000]}"}]
        return {
            "messages": msgs,
            "tool_calls": state["tool_calls"] + 1,
            "queried": state["queried"] + [a["sql"].lower()],
        }

    def route(state: State) -> str:
        if state["done"]:
            return "report"
        return "query" if state.get("last") else "plan"

    def report(state: State) -> State:
        findings = state["findings"]
        for f in findings:  # independent re-run so claimed counts are checked, not trusted
            f["verified_count"] = _recount(ro, f.get("evidence_sql"))
            if (
                f.get("verdict") == "refuted"
                and (f["verified_count"] or 0) > 0
                and not f.get("legitimate_explanation")
            ):  # a refutation with live violations and no justification is not trustworthy
                f["llm_verdict"], f["verdict"] = "refuted", "insufficient"
            claimed = f.get("violations")
            f["count_matches_claim"] = (
                None if claimed is None or f["verified_count"] is None
                else claimed == f["verified_count"]
            )  # fmt: skip
        result = {
            "run_id": run_id,
            "tool_calls": state["tool_calls"],
            "tokens": state["tokens"],
            "profile_signals": state["profile"]["signals"],
            "error": state.get("error"),
            "elapsed_s": round(time.monotonic() - t0, 1),
            "findings": findings,
        }
        (trace.dir / "report.json").write_text(json.dumps(result, indent=2, default=str))
        trace.log("report", n_findings=len(findings))
        return {"findings": findings}

    g = StateGraph(State)
    g.add_node("profile", profile)
    g.add_node("plan", plan)
    g.add_node("query", query)
    g.add_node("report", report)
    g.add_edge(START, "profile")
    g.add_edge("profile", "plan")
    g.add_conditional_edges("plan", route, {"query": "query", "plan": "plan", "report": "report"})
    g.add_edge("query", "plan")
    g.add_edge("report", END)
    return g.compile(), run_id, trace.dir


def profile_signals(state: State) -> list[dict]:
    return state["profile"]["signals"]


def _coverage_gap(state: State, signals: list[dict], s: Settings) -> str | None:
    """Deterministic gate: why the agent may not conclude yet (None = free to conclude)."""
    sqls = " ".join(state["queried"])
    missing = sorted({c for sg in signals for c in sg["columns"] if c.lower() not in sqls})
    parts = []
    if len(state["queried"]) < s.min_queries:
        parts.append(f"run at least {s.min_queries} queries (so far {len(state['queried'])}).")
    if missing:
        parts.append(f"no query yet touches profiler-flagged columns: {', '.join(missing)}.")
    return " ".join(parts) or None


def _recount(ro: ReadOnlyQuery, sql: str | None) -> int | None:
    """Independently re-run the model's evidence query and derive the offending-row count.

    A single-cell numeric result from a count(...) query is the count itself; anything else is
    treated as a row set and counted.
    """
    if not sql:
        return None
    try:
        sql = _drop_outer_limit(sql)
        r = ro.run(sql)
        if (
            len(r.rows) == 1
            and len(r.columns) == 1
            and isinstance(r.rows[0][0], int | float)
            and "count(" in sql.lower().replace(" ", "")
        ):
            return int(r.rows[0][0])
        return ro.run(f"SELECT count(*) FROM ({sql}) AS _e").rows[0][0]
    except SQLRejected:
        return None


def _drop_outer_limit(sql: str) -> str:
    """Evidence queries are often sampled with LIMIT; the count must cover all offending rows."""
    try:
        tree = sqlglot.parse_one(sql.strip().rstrip(";"), read="duckdb")
    except sqlglot.errors.ParseError as e:
        raise SQLRejected(f"SQL parse error: {str(e)[:200]}") from e
    if isinstance(tree, exp.Select | exp.Union):
        tree.set("limit", None)
    return tree.sql(dialect="duckdb")


def run(
    path: str | Path, llm: LLM, settings: Settings | None = None, verbose: bool = False
) -> dict:
    graph, run_id, out_dir = build_graph(path, llm, settings, verbose=verbose)
    graph.invoke({}, {"recursion_limit": 200})
    return json.loads((out_dir / "report.json").read_text())
