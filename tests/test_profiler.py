import json

import pytest

from dq_agent.profiler import Profiler
from dq_agent.synth.orders import generate


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("orders")
    csv = generate(d)
    return csv, json.loads((d / "orders.truth.json").read_text())


@pytest.fixture(scope="module")
def prof(data):
    return Profiler("orders", data[0])


@pytest.fixture(scope="module")
def profile(prof):
    return prof.run()


def sig(profile, eid):
    return next((s for s in profile.signals if s.evidence_id == eid), None)


def test_row_count(profile):
    assert profile.n_rows == 5025


def test_explicit_defects_found(profile):
    assert sig(profile, "high_null:customer_id").count == 250
    assert sig(profile, "duplicate_rows:row").count == 25
    assert sig(profile, "duplicate_key:order_id").count == 25
    assert sig(profile, "mixed_type:quantity").count == 12


def test_statistical_outliers_found(profile):
    assert sig(profile, "numeric_outlier:weight_kg").count >= 40


def test_clean_columns_not_flagged(profile):
    assert sig(profile, "high_null:order_date") is None


def test_evidence_is_bounded_and_matches_truth(prof, profile, data):
    rows = prof.evidence("mixed_type:quantity", limit=5)
    assert len(rows) == 5
    truth = set(data[1]["mixed_type_quantity"]["order_ids"])
    assert {r["order_id"] for r in prof.evidence("mixed_type:quantity", 100)} == truth


def test_summary_has_no_row_data(profile):
    s = profile.summary()
    assert "rows" not in s and s["n_rows"] == 5025


def test_duplicate_rows_evidence_is_runnable_in_sandbox(prof, profile, data):
    from dq_agent.eval.score import ids_for
    from dq_agent.sql_tool import ReadOnlyQuery

    sg = sig(profile, "duplicate_rows:row")
    ids = ids_for(ReadOnlyQuery(data[0], max_rows=100_000), sg.sql, "order_id")
    assert ids == set(data[1]["duplicate_rows"]["order_ids"])
