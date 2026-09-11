from unittest.mock import MagicMock

import pandas as pd

import cleaning


def test_load_and_clean_builds_expected_query_and_drops_duplicates(monkeypatch):
    raw = pd.DataFrame(
        {
            "PULocationID": [1, 1],
            "DOLocationID": [2, 2],
            "fare_amount": [15.0, 15.0],
        }
    )

    relation = MagicMock()
    relation.df.return_value = raw
    sql_mock = MagicMock(return_value=relation)
    monkeypatch.setattr(cleaning.duckdb, "sql", sql_mock)

    result = cleaning.load_and_clean("trips.parquet", "zones.csv")

    assert len(result) == 1
    pd.testing.assert_frame_equal(result.reset_index(drop=True), raw.iloc[[0]].reset_index(drop=True))

    query = sql_mock.call_args.args[0]
    assert "FROM 'trips.parquet' AS trips" in query
    assert "LEFT JOIN 'zones.csv' AS pu_zones" in query
    assert "LEFT JOIN 'zones.csv' AS do_zones" in query
    assert "trips.passenger_count > 0" in query
    assert "trips.trip_distance > 0" in query
    assert "trips.trip_distance < 50" in query
    assert "trips.fare_amount > 0" in query
    assert "trips.fare_amount < 120" in query
    assert "BETWEEN 0 AND 120" in query
    assert "EXCLUDE (store_and_fwd_flag)" in query


def test_load_and_clean_uses_default_paths(monkeypatch):
    relation = MagicMock()
    relation.df.return_value = pd.DataFrame({"x": [1]})
    sql_mock = MagicMock(return_value=relation)
    monkeypatch.setattr(cleaning.duckdb, "sql", sql_mock)

    cleaning.load_and_clean()

    query = sql_mock.call_args.args[0]
    assert cleaning.FARE_PATH in query
    assert cleaning.ZONE_PATH in query
