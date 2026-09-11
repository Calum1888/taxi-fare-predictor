from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import features


def test_date_time_features_adds_duration_hour_and_day_without_mutating_input():
    original = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(["2026-01-05 08:15:00", "2026-01-06 23:50:00"]),
            "tpep_dropoff_datetime": pd.to_datetime(["2026-01-05 08:45:00", "2026-01-07 00:10:00"]),
        }
    )

    result = features.date_time_features(original)

    assert "trip_duration" not in original.columns
    assert result["trip_duration"].tolist() == [30.0, 20.0]
    assert result["pickup_hour"].tolist() == [8, 23]
    assert result["pickup_dayofweek"].tolist() == [0, 1]


def test_add_rate_category_maps_known_unknown_and_missing_codes():
    df = pd.DataFrame({"RatecodeID": [1, 2, 3, 4, 5, 6, 99, 123, np.nan]})

    result = features.add_rate_category(df)

    assert result["rate_category"].tolist() == [
        "standard",
        "jfk",
        "newark",
        "nassau_westchester",
        "negotiated",
        "other",
        "unknown",
        "unknown",
        "unknown",
    ]
    assert "rate_category" not in df.columns


def test_add_airport_flag_marks_only_jfk_and_newark():
    df = pd.DataFrame({"rate_category": ["standard", "jfk", "newark", "unknown"]})

    result = features.add_airport_flag(df)

    assert result["is_airport_trip"].tolist() == [False, True, True, False]


def test_engineer_features_runs_all_feature_steps():
    df = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(["2026-01-05 08:00:00"]),
            "tpep_dropoff_datetime": pd.to_datetime(["2026-01-05 08:30:00"]),
            "RatecodeID": [2],
        }
    )

    result = features.engineer_features(df)

    assert result.loc[0, "trip_duration"] == 30.0
    assert result.loc[0, "pickup_hour"] == 8
    assert result.loc[0, "pickup_dayofweek"] == 0
    assert result.loc[0, "rate_category"] == "jfk"
    assert bool(result.loc[0, "is_airport_trip"]) is True


def test_load_zone_lookup_reads_expected_relative_data_path(monkeypatch):
    expected = pd.DataFrame({"LocationID": [1], "Borough": ["Manhattan"]})
    read_csv = MagicMock(return_value=expected)
    monkeypatch.setattr(features.pd, "read_csv", read_csv)

    result = features.load_zone_lookup()

    pd.testing.assert_frame_equal(result, expected)
    path = Path(read_csv.call_args.args[0])
    assert path.name == "taxi_zone_lookup.csv"
    assert path.parent.name == "data"


def test_create_zone_matrix_uses_mean_for_duplicate_zone_pairs():
    df = pd.DataFrame(
        {
            "PULocationID": [1, 1, 1, 2],
            "DOLocationID": [2, 2, 3, 1],
            "trip_distance": [4.0, 6.0, 10.0, 8.0],
        }
    )

    result = features.create_zone_matrix(df, "trip_distance")

    assert result.loc[1, 2] == 5.0
    assert result.loc[1, 3] == 10.0
    assert result.loc[2, 1] == 8.0


def test_transpose_fillna_uses_reverse_direction_only_for_missing_values():
    matrix = pd.DataFrame(
        [[1.0, np.nan], [7.0, 2.0]],
        index=[1, 2],
        columns=[1, 2],
    )

    result = features.transpose_fillna(matrix)

    assert result.loc[1, 2] == 7.0
    assert result.loc[2, 1] == 7.0
    assert result.loc[1, 1] == 1.0


@pytest.mark.parametrize(
    ("value_col", "expected"),
    [("trip_distance", 0.8), ("trip_duration", 8.0)],
)
def test_intra_zone_fillna_uses_correct_default_and_preserves_existing_values(value_col, expected):
    matrix = pd.DataFrame(
        [[np.nan, 3.0], [4.0, 9.0]],
        index=[1, 2],
        columns=[1, 2],
    )

    result = features.intra_zone_fillna(matrix.copy(), value_col)

    assert result.loc[1, 1] == expected
    assert result.loc[2, 2] == 9.0


def test_intra_zone_fillna_rejects_unsupported_value_column():
    matrix = pd.DataFrame([[np.nan]], index=[1], columns=[1])

    with pytest.raises(KeyError):
        features.intra_zone_fillna(matrix, "unsupported")


def test_borough_mean_fillna_uses_borough_pair_mean_for_missing_zone_pair():
    df = pd.DataFrame(
        {
            "pickup_borough": ["Manhattan", "Manhattan"],
            "dropoff_borough": ["Queens", "Queens"],
            "trip_distance": [10.0, 14.0],
        }
    )
    matrix = pd.DataFrame([[np.nan]], index=[1], columns=[2])
    lookup_ref = pd.DataFrame(
        {
            "LocationID": [1, 2],
            "Borough": ["Manhattan", "Queens"],
        }
    )

    result = features.borough_mean_fillna(df, matrix, lookup_ref, "trip_distance")

    assert result.loc[1, 2] == 12.0


def test_build_zone_lookup_runs_fallback_chain_and_fills_final_nans(monkeypatch):
    raw_matrix = pd.DataFrame([[1.0, np.nan], [np.nan, 2.0]], index=[1, 2], columns=[1, 2])
    after_transpose = raw_matrix.copy()
    after_intra = raw_matrix.copy()
    after_borough = pd.DataFrame([[1.0, 3.0], [np.nan, 2.0]], index=[1, 2], columns=[1, 2])
    lookup_ref = pd.DataFrame({"LocationID": [1, 2], "Borough": ["A", "B"]})

    monkeypatch.setattr(features, "load_zone_lookup", MagicMock(return_value=lookup_ref))
    monkeypatch.setattr(features, "create_zone_matrix", MagicMock(return_value=raw_matrix))
    monkeypatch.setattr(features, "transpose_fillna", MagicMock(return_value=after_transpose))
    monkeypatch.setattr(features, "intra_zone_fillna", MagicMock(return_value=after_intra))
    monkeypatch.setattr(features, "borough_mean_fillna", MagicMock(return_value=after_borough))

    result = features.build_zone_lookup(pd.DataFrame({"x": [1]}), "trip_distance")

    # mean of existing stacked values: (1 + 3 + 2) / 3 = 2
    assert result.loc[2, 1] == 2.0
    assert result.isna().sum().sum() == 0
    features.create_zone_matrix.assert_called_once()
    features.transpose_fillna.assert_called_once_with(raw_matrix)
    features.intra_zone_fillna.assert_called_once_with(after_transpose, "trip_distance")
    args = features.borough_mean_fillna.call_args.args
    assert args[1] is after_intra
    assert args[2] is lookup_ref
    assert args[3] == "trip_distance"


def test_add_zone_features_merges_lookup_value_and_keeps_unmatched_rows():
    df = pd.DataFrame(
        {
            "PULocationID": [1, 9],
            "DOLocationID": [2, 9],
            "fare_amount": [10.0, 20.0],
        }
    )
    lookup = pd.DataFrame([[np.nan, 5.5], [4.5, np.nan]], index=[1, 2], columns=[1, 2])

    result = features.add_zone_features(df, lookup, "mean_trip_distance")

    assert result.loc[0, "mean_trip_distance"] == 5.5
    assert pd.isna(result.loc[1, "mean_trip_distance"])
    assert len(result) == len(df)


def test_time_sorted_split_df_sorts_before_chronological_split():
    df = pd.DataFrame(
        {
            "when": pd.to_datetime(["2026-01-03", "2026-01-01", "2026-01-04", "2026-01-02"]),
            "id": [3, 1, 4, 2],
        }
    )

    train, test = features.time_sorted_split_df(df, "when", 0.25)

    assert train["id"].tolist() == [1, 2, 3]
    assert test["id"].tolist() == [4]
    assert train.index.tolist() == [0, 1, 2]
    assert test.index.tolist() == [3]
