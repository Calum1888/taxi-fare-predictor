"""
Tests for the filtering boundaries used in src/cleaning.py.

Since load_and_clean() reads from disk via DuckDB, we don't unit-test it
directly. Instead we test the boundary CONSTANTS and boundary LOGIC in
isolation using small in-memory DataFrames, so tests stay fast and don't
depend on the actual data file being present.

Run with: pytest tests/test_cleaning.py -v
"""

import sys
import os
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from cleaning import PASSENGER_COUNT_MIN, TRIP_DISTANCE_MIN, TRIP_DISTANCE_MAX, FARE_AMOUNT_MIN, FARE_AMOUNT_MAX, TRIP_DURATION_MIN, TRIP_DURATION_MAX

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors the WHERE clause logic in load_and_clean(), for testing
    the boundary values without needing a real parquet file on disk."""
    mask = (
        (df["passenger_count"] > PASSENGER_COUNT_MIN)
        & (df["trip_distance"] > TRIP_DISTANCE_MIN)
        & (df["trip_distance"] < TRIP_DISTANCE_MAX)
        & (df["fare_amount"] > FARE_AMOUNT_MIN)
        & (df["fare_amount"] < FARE_AMOUNT_MAX)
        & (df["trip_duration_min"] >= TRIP_DURATION_MIN)
        & (df["trip_duration_min"] <= TRIP_DURATION_MAX)
    )
    return df[mask]


@pytest.fixture
def sample_trips():
    return pd.DataFrame({
        "passenger_count": [1, 0, 2, 1, 1],
        "trip_distance":   [3.5, 2.0, -1.0, 60.0, 10.0],
        "fare_amount":     [15.0, 10.0, 12.0, 25.0, 150.0],
        "trip_duration_min": [12.0, 8.0, 5.0, 30.0, 45.0],
    })


def test_filters_out_zero_passenger_count(sample_trips):
    result = apply_filters(sample_trips)
    assert (result["passenger_count"] > PASSENGER_COUNT_MIN).all()


def test_filters_out_negative_distance(sample_trips):
    result = apply_filters(sample_trips)
    assert (result["trip_distance"] > 0).all()


def test_filters_out_distance_over_max(sample_trips):
    result = apply_filters(sample_trips)
    assert (result["trip_distance"] < TRIP_DISTANCE_MAX).all()


def test_filters_out_fare_over_max(sample_trips):
    result = apply_filters(sample_trips)
    assert (result["fare_amount"] < FARE_AMOUNT_MAX).all()


def test_valid_row_survives_all_filters(sample_trips):
    result = apply_filters(sample_trips)
    # Row 0 (index 0) should be the only fully valid row in this fixture
    assert len(result) == 1
    assert result.index[0] == 0


def test_negative_duration_excluded():
    df = pd.DataFrame({
        "passenger_count": [1],
        "trip_distance": [3.0],
        "fare_amount": [15.0],
        "trip_duration_min": [-5.0],
    })
    result = apply_filters(df)
    assert len(result) == 0