from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

import interface


def _valid_input(**overrides):
    values = {
        "passenger_count": 2,
        "pickup_zone": "Midtown Center",
        "dropoff_zone": "JFK Airport",
        "rate_category": "standard",
        "pickup_hour": 11,
        "pickup_dayofweek": 4,
        "tip_percentage": 15.0,
    }
    values.update(overrides)
    return interface.FarePredictionInput(**values)


def test_get_zones_returns_alphabetically_sorted_zone_names(monkeypatch):
    monkeypatch.setattr(
        interface,
        "get_zone_name_to_id",
        MagicMock(return_value={"Queens": 3, "Bronx": 1, "Manhattan": 2}),
    )

    assert interface.get_zones() == {"zones": ["Bronx", "Manhattan", "Queens"]}


def test_serve_ui_contains_main_form_and_api_calls():
    html = interface.serve_ui()

    assert "Taxi Fare Predictor" in html
    assert "Estimate Fare" in html
    assert "fetch('/zones')" in html
    assert "fetch('/predict'" in html
    assert "tip_percentage" in html


def test_predict_calls_prediction_and_tip_helpers_and_returns_breakdown(monkeypatch):
    prediction = {
        "base_fare": 20.0,
        "surcharges": {"mta_state_surcharge": 0.5},
        "fare_with_surcharges": 22.5,
    }
    predict_fare = MagicMock(return_value=prediction)
    optional_tip = MagicMock(return_value=3.375)
    monkeypatch.setattr(interface, "predict_fare", predict_fare)
    monkeypatch.setattr(interface, "optional_tip", optional_tip)

    result = interface.predict(_valid_input())

    predict_fare.assert_called_once_with(
        passenger_count=2,
        pickup_zone="Midtown Center",
        dropoff_zone="JFK Airport",
        rate_category="standard",
        pickup_hour=11,
        pickup_dayofweek=4,
    )
    optional_tip.assert_called_once_with(fare=22.5, tip_percentage=15.0)
    assert result == {
        "base_fare": 20.0,
        "surcharges": {"mta_state_surcharge": 0.5},
        "fare_with_surcharges": 22.5,
        "tip_amount": 3.38,
        "total_fare": 25.88,
    }


def test_predict_uses_zero_tip_by_default(monkeypatch):
    monkeypatch.setattr(
        interface,
        "predict_fare",
        MagicMock(return_value={"base_fare": 10.0, "surcharges": {}, "fare_with_surcharges": 10.0}),
    )
    tip = MagicMock(return_value=0.0)
    monkeypatch.setattr(interface, "optional_tip", tip)

    interface.predict(_valid_input(tip_percentage=0.0))

    tip.assert_called_once_with(fare=10.0, tip_percentage=0.0)


def test_predict_converts_value_error_to_http_400(monkeypatch):
    monkeypatch.setattr(interface, "predict_fare", MagicMock(side_effect=ValueError("Unknown pickup zone")))

    with pytest.raises(HTTPException) as exc_info:
        interface.predict(_valid_input())

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Unknown pickup zone"
