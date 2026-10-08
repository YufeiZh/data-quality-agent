"""Score a run against the injected-anomaly ground truth.

A finding/signal "detects" an injected kind when the set of ids its evidence SQL returns covers
>= 80% of the kind's ids (recall) and >= 50% of the returned ids belong to the kind (precision).
Both the deterministic profiler (baseline A) and the agent's supported findings are scored.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..agent import _drop_outer_limit
from ..sql_tool import ReadOnlyQuery, SQLRejected

MIN_RECALL, MIN_PRECISION = 0.8, 0.5


def ids_for(ro: ReadOnlyQuery, sql: str | None, id_col: str) -> set[str] | None:
    """Ids returned by an evidence query; None if it cannot be scored (error / no id column)."""
    if not sql:
        return None
    try:
        r = ro.run(_drop_outer_limit(sql))
    except SQLRejected:
        return None
    if id_col not in r.columns:
        return None
    i = r.columns.index(id_col)
    return {str(row[i]) for row in r.rows if row[i] is not None}


def _match(ids: set[str], truth: dict) -> list[str]:
    hits = []
    for kind, t in truth.items():
        tids = set(t["order_ids"])
        inter = len(ids & tids)
        if inter / len(tids) >= MIN_RECALL and inter / max(len(ids), 1) >= MIN_PRECISION:
            hits.append(kind)
    return hits


def score_items(items: list[tuple[str, str | None]], ro, truth, id_col="order_id") -> dict:
    """items: (label, evidence_sql). Returns per-kind detection and false-positive lists."""
    detected: dict[str, str] = {}
    false_pos, legit_flagged, unscorable = [], [], []
    for label, sql in items:
        ids = ids_for(ro, sql, id_col)
        if ids is None:
            unscorable.append(label)
            continue
        hits = _match(ids, truth)
        if not hits:
            # Non-empty evidence overlapping no defect = noise; overlapping only legit = wrong flag.
            legit = [k for k, t in truth.items() if not t["defect"] and ids & set(t["order_ids"])]
            (legit_flagged if legit and not (ids & _defect_ids(truth)) else false_pos).append(label)
        for k in hits:
            if truth[k]["defect"]:
                detected.setdefault(k, label)
            else:
                legit_flagged.append(label)
    defects = [k for k, t in truth.items() if t["defect"]]
    found = [k for k in defects if k in detected]
    return {
        "recall": f"{len(found)}/{len(defects)}",
        "detected": {k: detected[k] for k in found},
        "missed": [k for k in defects if k not in detected],
        "false_positives": false_pos,
        "legit_wrongly_flagged": legit_flagged,
        "unscorable": unscorable,
    }


def _defect_ids(truth) -> set[str]:
    return {i for t in truth.values() if t["defect"] for i in t["order_ids"]}


def score_report(report_path: str | Path, csv_path: str | Path, id_col: str = "order_id") -> dict:
    rep = json.loads(Path(report_path).read_text())
    truth = json.loads(Path(csv_path).with_suffix(".truth.json").read_text())
    ro = ReadOnlyQuery(csv_path, max_rows=100_000)
    base = [(s["evidence_id"], s["sql"]) for s in rep["profile_signals"]]
    agent = [
        (f["hypothesis"][:80], f.get("evidence_sql"))
        for f in rep["findings"]
        if f.get("verdict") == "supported"
    ]
    return {
        "run_id": rep["run_id"],
        "cost": {k: rep.get(k) for k in ("tool_calls", "tokens", "elapsed_s")},
        "A_profiler_only": score_items(base, ro, truth, id_col),
        "C_agent": score_items(agent, ro, truth, id_col),
    }
