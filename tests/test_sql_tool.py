import pytest

from dq_agent.sql_tool import ReadOnlyQuery, SQLRejected
from dq_agent.synth.orders import generate


@pytest.fixture(scope="module")
def ro(tmp_path_factory):
    return ReadOnlyQuery(generate(tmp_path_factory.mktemp("o")), max_rows=5, timeout_s=2)


def test_select_ok(ro):
    r = ro.run("SELECT count(*) AS n FROM t WHERE ship_date < order_date")
    assert r.rows == [[30]]


def test_cte_ok(ro):
    assert ro.run("WITH x AS (SELECT * FROM t) SELECT count(*) FROM x").rows[0][0] == 5025


def test_row_limit(ro):
    r = ro.run("SELECT * FROM t")
    assert len(r.rows) == 5 and r.truncated


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE t",
        "DELETE FROM t",
        "UPDATE t SET quantity = 1",
        "SELECT 1; SELECT 2",
        "COPY t TO '/tmp/leak.csv'",
        "SELECT * FROM read_csv('/etc/passwd')",
        "SELECT * FROM '/etc/passwd'",
        "SELECT * FROM other_table",
        "ATTACH 'x.db'",
        "SET enable_external_access=true",
        "PRAGMA database_list",
    ],
)
def test_rejected(ro, sql):
    with pytest.raises(SQLRejected):
        ro.run(sql)


def test_timeout(ro):
    with pytest.raises(SQLRejected, match="timeout"):
        ro.run(
            "SELECT count(*) FROM t a, t b, t c, t d "
            "WHERE a.quantity || b.status || c.status = d.status"
        )
