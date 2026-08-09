"""
Tests for src/features.py

Run with: pytest tests/test_features.py -v
"""

import sys
import os
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from features import date_time_features, add_rate_category, add_airport_flag


@pytest.fixture
def sample_trips():
    """A small, hand-built DataFrame covering the cases we care about."""
    return pd.DataFrame({
        "tpep_pickup_datetime": pd.to_datetime([
            "2026-01-15 08:30:00",   # weekday morning
            "2026-01-17 23:15:00",   # Saturday night
        ]),
        "tpep_dropoff_datetime": pd.to_datetime([
            "2026-01-15 08:45:00",
            "2026-01-17 23:40:00",
        ]),
        "RatecodeID": [1, 2],  # standard, JFK
    })


def test_add_datetime_features_creates_expected_columns(sample_trips):
    result = add_datetime_features(sample_trips)
    for col in ["trip_duration_min", "pickup_hour", "pickup_dayofweek", "pickup_month", "is_weekend"]:
        assert col in result.columns


def test_trip_duration_calculated_correctly(sample_trips):
    result = add_datetime_features(sample_trips)
    # First trip: 08:30 -> 08:45 = 15 minutes
    assert result["trip_duration_min"].iloc[0] == pytest.approx(15.0)


def test_weekend_flag_correct(sample_trips):
    result = add_datetime_features(sample_trips)
    # 2026-01-15 is a Thursday (weekday), 2026-01-17 is a Saturday (weekend)
    assert result["is_weekend"].iloc[0] == False
    assert result["is_weekend"].iloc[1] == True


def test_rate_category_maps_known_codes(sample_trips):
    result = add_rate_category(sample_trips)
    assert result["rate_category"].iloc[0] == "standard"
    assert result["rate_category"].iloc[1] == "jfk"


def test_rate_category_handles_unknown_code():
    df = pd.DataFrame({"RatecodeID": [99, 7]})  # 99 = known unknown, 7 = not in map at all
    result = add_rate_category(df)
    assert result["rate_category"].iloc[0] == "unknown"
    assert result["rate_category"].iloc[1] == "unknown"  # unmapped codes fall back to unknown


def test_airport_flag_true_for_jfk_and_newark():
    df = pd.DataFrame({"rate_category": ["jfk", "newark", "standard", "negotiated"]})
    result = add_airport_flag(df)
    assert result["is_airport_trip"].tolist() == [True, True, False, False]