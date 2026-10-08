"""Sandboxed read-only SQL execution against a single dataset (exposed to SQL as table `t`)."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import duckdb
import sqlglot
from sqlglot import exp

from .profiler import load_view

# Table functions / expressions that could touch the filesystem or network.
_BLOCKED_FUNCS = {
    "READ_CSV", "READ_CSV_AUTO", "READ_PARQUET", "READ_JSON", "READ_JSON_AUTO", "READ_TEXT",
    "READ_BLOB", "GLOB", "PARQUET_SCAN", "CSV_SCAN", "SNIFF_CSV", "QUERY", "QUERY_TABLE",
}  # fmt: skip


_FORBIDDEN = (
    exp.Command, exp.Create, exp.Insert, exp.Update, exp.Delete,
    exp.Drop, exp.Alter, exp.Copy, exp.Pragma, exp.Set, exp.Attach,
)  # fmt: skip


class SQLRejected(ValueError):
    pass


def validate_sql(sql: str) -> str:
    """Return a normalised single SELECT statement or raise SQLRejected."""
    try:
        stmts = [s for s in sqlglot.parse(sql, read="duckdb") if s is not None]
    except sqlglot.errors.ParseError as e:
        raise SQLRejected(f"SQL parse error: {str(e)[:200]}") from e
    if len(stmts) != 1:
        raise SQLRejected("exactly one statement is allowed")
    stmt = stmts[0]
    if not isinstance(stmt, exp.Select | exp.Union | exp.Subquery | exp.With):
        raise SQLRejected("only SELECT queries are allowed")
    for node in stmt.walk():
        if isinstance(node, _FORBIDDEN):
            raise SQLRejected(f"{type(node).__name__} not allowed")
        if isinstance(node, exp.Func):
            name = (node.sql_name() or "").upper()
            if name in _BLOCKED_FUNCS or name.startswith(("READ_", "PRAGMA_")):
                raise SQLRejected(f"function {name} not allowed")
        if isinstance(node, exp.Table) and isinstance(node.this, exp.Func):
            raise SQLRejected("table functions are not allowed")
        if isinstance(node, exp.Table) and node.name and node.name.lower() not in _cte_names(stmt):
            if node.name.lower() != "t":
                raise SQLRejected(f"unknown table {node.name!r}; the only table is `t`")
    return stmt.sql(dialect="duckdb")


def _cte_names(stmt: exp.Expression) -> set[str]:
    return {c.alias.lower() for c in stmt.find_all(exp.CTE)}


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[list]
    truncated: bool

    def to_dict(self) -> dict:
        return {"columns": self.columns, "rows": self.rows, "truncated": self.truncated}


class ReadOnlyQuery:
    def __init__(
        self,
        path: str | Path,
        max_rows: int = 200,
        timeout_s: float = 10.0,
        dev_single_thread: bool = False,
    ):
        self.max_rows, self.timeout_s = max_rows, timeout_s
        self.con = duckdb.connect()
        load_view(self.con, path)
        # Materialise, then cut off all file/network access for this connection.
        self.con.execute("CREATE TABLE t_data AS SELECT * FROM t")
        self.con.execute("DROP VIEW t")
        self.con.execute("ALTER TABLE t_data RENAME TO t")
        if dev_single_thread:  # DEV/REPRO ONLY (see Settings.dev_single_thread_sandbox)
            self.con.execute("SET threads=1")
        self.con.execute("SET enable_external_access=false")
        self.con.execute("SET lock_configuration=true")

    def run(self, sql: str) -> QueryResult:
        safe = validate_sql(sql)
        wrapped = f"SELECT * FROM ({safe}) AS _q LIMIT {self.max_rows + 1}"
        timer = threading.Timer(self.timeout_s, self.con.interrupt)
        timer.start()
        try:
            cur = self.con.execute(wrapped)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
        except duckdb.InterruptException as e:
            raise SQLRejected(f"query exceeded {self.timeout_s}s timeout") from e
        except duckdb.Error as e:
            raise SQLRejected(f"query error: {str(e)[:300]}") from e
        finally:
            timer.cancel()
        truncated = len(rows) > self.max_rows
        rows = [[_jsonable(v) for v in r] for r in rows[: self.max_rows]]
        return QueryResult(cols, rows, truncated)


def _jsonable(v):
    return v if v is None or isinstance(v, bool | int | float | str) else str(v)
