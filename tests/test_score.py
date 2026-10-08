import json

from dq_agent.eval.score import score_items
from dq_agent.sql_tool import ReadOnlyQuery
from dq_agent.synth.orders import generate


def test_scoring_matches_truth(tmp_path):
    csv = generate(tmp_path)
    truth = json.loads((tmp_path / "orders.truth.json").read_text())
    ro = ReadOnlyQuery(csv, max_rows=100_000)
    r = score_items(
        [
            ("ship", "SELECT * FROM t WHERE ship_date < order_date"),
            ("sampled", "SELECT * FROM t WHERE ship_date < order_date LIMIT 3"),  # limit ignored
            ("noise", "SELECT * FROM t WHERE status = 'returned'"),
            ("bulk", "SELECT * FROM t WHERE TRY_CAST(quantity AS INT) > 400"),
            ("agg", "SELECT count(*) FROM t"),
        ],
        ro,
        truth,
    )
    assert "ship_before_order" in r["detected"]
    assert r["false_positives"] == ["noise"]
    assert r["legit_wrongly_flagged"] == ["bulk"]
    assert r["unscorable"] == ["agg"]
    assert "total_mismatch" in r["missed"]
