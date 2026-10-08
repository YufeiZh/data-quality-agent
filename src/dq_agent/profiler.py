"""Deterministic DuckDB profiler: full-table scan -> schema, stats, candidate anomaly signals.

The LLM never sees raw data; it receives `Profile.summary()` and fetches bounded samples
through `evidence()` using the `evidence_id` attached to every signal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import duckdb

NUMERIC_TYPES = (
    "TINYINT",
    "SMALLINT",
    "INTEGER",
    "BIGINT",
    "HUGEINT",
    "FLOAT",
    "DOUBLE",
    "DECIMAL",
)


def q(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    n_null: int
    n_distinct: int
    min: str | None = None
    max: str | None = None
    stats: dict[str, float] = field(default_factory=dict)  # numeric only


@dataclass
class Signal:
    evidence_id: str
    kind: str  # high_null | duplicate_rows | duplicate_key | mixed_type | numeric_outlier
    columns: list[str]
    count: int
    severity: str  # low | medium | high
    description: str
    sql: str  # replayable query returning the affected rows


@dataclass
class Profile:
    dataset_id: str
    n_rows: int
    columns: list[ColumnProfile]
    signals: list[Signal]

    def summary(self) -> dict:
        """Compact, LLM-safe view (no row data)."""
        return asdict(self)


def _severity(frac: float) -> str:
    return "high" if frac >= 0.05 else "medium" if frac >= 0.005 else "low"


def load_view(con: duckdb.DuckDBPyConnection, path: str | Path) -> None:
    p = str(path).replace("'", "''")
    reader = "read_parquet" if str(path).endswith(".parquet") else "read_csv_auto"
    extra = ", sample_size=-1" if reader == "read_csv_auto" else ""
    con.execute(f"CREATE OR REPLACE VIEW t AS SELECT * FROM {reader}('{p}'{extra})")


class Profiler:
    def __init__(self, dataset_id: str, path: str | Path):
        self.dataset_id = dataset_id
        self.con = duckdb.connect()
        load_view(self.con, path)
        self._signals: dict[str, Signal] = {}

    def run(self) -> Profile:
        n = self.con.execute("SELECT count(*) FROM t").fetchone()[0]
        cols = self._columns()
        signals: list[Signal] = []
        signals += self._null_signals(cols, n)
        signals += self._duplicate_signals(cols, n)
        signals += self._mixed_type_signals(cols)
        signals += self._outlier_signals(cols, n)
        self._signals = {s.evidence_id: s for s in signals}
        return Profile(self.dataset_id, n, cols, signals)

    def evidence(self, evidence_id: str, limit: int = 10) -> list[dict]:
        sig = self._signals[evidence_id]
        df = self.con.execute(f"SELECT * FROM ({sig.sql}) LIMIT {int(limit)}").df()
        return df.astype(object).where(df.notna(), None).to_dict("records")

    # ---- internals -------------------------------------------------------------------------
    def _columns(self) -> list[ColumnProfile]:
        out = []
        for name, dtype, *_ in self.con.execute("DESCRIBE t").fetchall():
            c = q(name)
            n_null, n_distinct, mn, mx = self.con.execute(
                f"SELECT count(*) FILTER (WHERE {c} IS NULL), count(DISTINCT {c}), "
                f"min({c})::VARCHAR, max({c})::VARCHAR FROM t"
            ).fetchone()
            cp = ColumnProfile(name, dtype, n_null, n_distinct, mn, mx)
            if dtype.startswith(NUMERIC_TYPES):
                row = self.con.execute(
                    f"SELECT avg({c}), stddev_samp({c}), quantile_cont({c}, 0.25), "
                    f"quantile_cont({c}, 0.5), quantile_cont({c}, 0.75) FROM t"
                ).fetchone()
                keys = ["mean", "std", "q1", "median", "q3"]
                cp.stats = {k: float(v) for k, v in zip(keys, row, strict=True) if v is not None}
            out.append(cp)
        return out

    def _sig(self, kind: str, cols: list[str], count: int, frac: float, desc: str, sql: str):
        eid = f"{kind}:{'+'.join(cols) or 'row'}"
        return Signal(eid, kind, cols, count, _severity(frac), desc, sql)

    def _null_signals(self, cols, n) -> list[Signal]:
        return [
            self._sig(
                "high_null",
                [c.name],
                c.n_null,
                c.n_null / n,
                f"{c.n_null} NULLs ({c.n_null / n:.2%}) in {c.name}",
                f"SELECT * FROM t WHERE {q(c.name)} IS NULL",
            )
            for c in cols
            if c.n_null > 0 and n > 0
        ]

    def _duplicate_signals(self, cols, n) -> list[Signal]:
        out = []
        n_dist = self.con.execute("SELECT count(*) FROM (SELECT DISTINCT * FROM t)").fetchone()[0]
        if n_dist < n:
            out.append(
                self._sig(
                    "duplicate_rows",
                    [],
                    n - n_dist,
                    (n - n_dist) / n,
                    f"{n - n_dist} fully duplicated rows",
                    "SELECT * EXCLUDE (_n) FROM (SELECT *, count(*) OVER (PARTITION BY "
                    + ", ".join(q(c.name) for c in cols)
                    + ") AS _n FROM t) WHERE _n > 1",
                )
            )
        for c in cols:  # near-unique column (identifier) that is not quite unique
            nn = n - c.n_null
            if nn and c.n_distinct / nn >= 0.95 and c.n_distinct < nn:
                k = q(c.name)
                out.append(
                    self._sig(
                        "duplicate_key",
                        [c.name],
                        nn - c.n_distinct,
                        (nn - c.n_distinct) / nn,
                        f"{c.name} is near-unique but has {nn - c.n_distinct} repeated values",
                        f"SELECT * FROM t QUALIFY count(*) OVER (PARTITION BY {k}) > 1 "
                        f"AND {k} IS NOT NULL",
                    )
                )
        return out

    def _mixed_type_signals(self, cols) -> list[Signal]:
        out = []
        for c in cols:
            if c.dtype != "VARCHAR":
                continue
            k = q(c.name)
            ok, tot = self.con.execute(
                f"SELECT count(TRY_CAST({k} AS DOUBLE)), count({k}) FROM t"
            ).fetchone()
            if tot and 0.9 <= ok / tot < 1:
                out.append(
                    self._sig(
                        "mixed_type",
                        [c.name],
                        tot - ok,
                        (tot - ok) / tot,
                        f"{c.name} is mostly numeric but {tot - ok} values are non-numeric",
                        f"SELECT * FROM t WHERE {k} IS NOT NULL "
                        f"AND TRY_CAST({k} AS DOUBLE) IS NULL",
                    )
                )
        return out

    def _outlier_signals(self, cols, n) -> list[Signal]:
        out = []
        for c in cols:
            s = c.stats
            if not s or s.get("q3") is None:
                continue
            iqr = s["q3"] - s["q1"]
            if iqr <= 0:
                continue
            lo, hi = s["q1"] - 3 * iqr, s["q3"] + 3 * iqr
            k = q(c.name)
            cnt = self.con.execute(
                f"SELECT count(*) FROM t WHERE {k} < {lo} OR {k} > {hi}"
            ).fetchone()[0]
            if cnt:
                out.append(
                    self._sig(
                        "numeric_outlier",
                        [c.name],
                        cnt,
                        cnt / n,
                        f"{cnt} values of {c.name} outside [{lo:.3g}, {hi:.3g}] (3*IQR fences)",
                        f"SELECT * FROM t WHERE {k} < {lo} OR {k} > {hi}",
                    )
                )
        return out
