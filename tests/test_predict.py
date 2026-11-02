from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import predict


# ---------------------------------------------------------------------------
# calculate_surcharges
# ---------------------------------------------------------------------------

NON_CBD_ZONE = 10  # not in CBD_ZONE_IDS
CBD_ZONE = 161     # Midtown Center, in CBD_ZONE_IDS


def _surcharges(pickup_hour=11, pickup_dayofweek=2, pickup_borough="Queens",
                dropoff_borough="Queens", pickup_zone_id=NON_CBD_ZONE,
                dropoff_zone_id=NON_CBD_ZONE):
    return predict.calculate_surcharges(
        pickup_zone_id=pickup_zone_id,
        dropoff_zone_id=dropoff_zone_id,
        pickup_borough=pickup_borough,
        dropoff_borough=dropoff_borough,
        pickup_hour=pickup_hour,
        pickup_dayofweek=pickup_dayofweek,
    )


def test_flat_surcharges_always_applied_and_nothing_else_off_peak_outside_manhattan():
    result = _surcharges()

    assert result["mta_state_surcharge"] == predict.MTA_STATE_SURCHARGE
    assert result["improvement_surcharge"] == predict.IMPROVEMENT_SURCHARGE
    assert result["rush_hour_surcharge"] == 0.0
    assert result["overnight_surcharge"] == 0.0
    assert result["nys_congestion_surcharge"] == 0.0
    assert result["cbd_congestion_fee"] == 0.0
    assert result["total_surcharges"] == pytest.approx(
        predict.MTA_STATE_SURCHARGE + predict.IMPROVEMENT_SURCHARGE
    )


@pytest.mark.parametrize("hour", [16, 17, 19])
def test_rush_hour_surcharge_applies_on_weekday_afternoons(hour):
    result = _surcharges(pickup_hour=hour, pickup_dayofweek=0)

    assert result["rush_hour_surcharge"] == predict.RUSH_HOUR_SURCHARGE


@pytest.mark.parametrize("dayofweek", [5, 6])
def test_rush_hour_surcharge_not_applied_at_weekends(dayofweek):
    result = _surcharges(pickup_hour=17, pickup_dayofweek=dayofweek)

    assert result["rush_hour_surcharge"] == 0.0


@pytest.mark.parametrize("hour", [15, 20])
def test_rush_hour_window_boundaries(hour):
    # 16:00 is inclusive, 20:00 is exclusive
    result = _surcharges(pickup_hour=hour, pickup_dayofweek=2)

    assert result["rush_hour_surcharge"] == 0.0


@pytest.mark.parametrize("hour,expected", [
    (19, 0.0), (20, predict.OVERNIGHT_SURCHARGE), (23, predict.OVERNIGHT_SURCHARGE),
    (0, predict.OVERNIGHT_SURCHARGE), (5, predict.OVERNIGHT_SURCHARGE), (6, 0.0),
])
def test_overnight_surcharge_window_wraps_midnight(hour, expected):
    result = _surcharges(pickup_hour=hour)

    assert result["overnight_surcharge"] == expected


def test_overnight_applies_at_weekends_too():
    result = _surcharges(pickup_hour=22, pickup_dayofweek=6)

    assert result["overnight_surcharge"] == predict.OVERNIGHT_SURCHARGE


@pytest.mark.parametrize("pu,do", [("Manhattan", "Queens"), ("Brooklyn", "Manhattan"),
                                   ("Manhattan", "Manhattan")])
def test_nys_congestion_surcharge_when_either_end_is_manhattan(pu, do):
    result = _surcharges(pickup_borough=pu, dropoff_borough=do)

    assert result["nys_congestion_surcharge"] == predict.NYS_CONGESTION_SURCHARGE


@pytest.mark.parametrize("pu_id,do_id", [(CBD_ZONE, NON_CBD_ZONE), (NON_CBD_ZONE, CBD_ZONE)])
def test_cbd_fee_when_either_end_is_a_cbd_zone(pu_id, do_id):
    result = _surcharges(pickup_zone_id=pu_id, dropoff_zone_id=do_id)

    assert result["cbd_congestion_fee"] == predict.CBD_CONGESTION_FEE


def test_newark_zone_is_excluded_from_cbd_zones():
    assert 1 not in predict.CBD_ZONE_IDS
    result = _surcharges(pickup_zone_id=1)

    assert result["cbd_congestion_fee"] == 0.0


def test_total_is_sum_of_all_items_in_worst_case_trip():
    result = _surcharges(pickup_hour=19, pickup_dayofweek=1, pickup_borough="Manhattan",
                         pickup_zone_id=CBD_ZONE)

    items = {k: v for k, v in result.items() if k != "total_surcharges"}
    assert result["total_surcharges"] == pytest.approx(sum(items.values()))
    assert result["total_surcharges"] == pytest.approx(0.5 + 1.0 + 2.5 + 2.5 + 0.75)


# ---------------------------------------------------------------------------
# optional_tip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fare,pct,expected", [(20.0, 15, 3.0), (20.0, 0, 0.0), (33.33, 100, 33.33)])
def test_optional_tip_is_percentage_of_fare(fare, pct, expected):
    assert predict.optional_tip(fare, pct) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# predict_fare (all file/MLflow dependencies are stubbed)
# ---------------------------------------------------------------------------

TRAINING_COLUMNS = [
    "trip_duration", "passenger_count", "trip_distance",
    "pickup_borough_Manhattan", "pickup_borough_Queens",
    "dropoff_borough_Manhattan", "dropoff_borough_Queens",
    "rate_category_standard", "rate_category_negotiated",
    "pickup_hour_14", "pickup_dayofweek_2",
]


@pytest.fixture
def stubbed_predict(monkeypatch):
    """Stub every loader predict_fare relies on, returning the fake model."""
    zones = {"Midtown Center": 161, "JFK Airport": 132, "Astoria": 7}
    boroughs = {"Midtown Center": "Manhattan", "JFK Airport": "Queens", "Astoria": "Queens"}

    ids = sorted(zones.values())
    distance = pd.DataFrame(np.arange(len(ids) ** 2, dtype=float).reshape(len(ids), -1),
                            index=ids, columns=ids)
    duration = distance * 10

    model = MagicMock()
    model.predict.return_value = np.array([42.123])

    monkeypatch.setattr(predict, "get_zone_name_to_id", lambda: zones)
    monkeypatch.setattr(predict, "get_zone_name_to_borough", lambda: boroughs)
    monkeypatch.setattr(predict, "get_distance_lookup", lambda: distance)
    monkeypatch.setattr(predict, "get_duration_lookup", lambda: duration)
    monkeypatch.setattr(predict, "get_training_columns", lambda: TRAINING_COLUMNS)
    monkeypatch.setattr(predict, "get_model", lambda: model)

    return model, distance, duration


def _call(**overrides):
    kwargs = dict(passenger_count=2, pickup_zone="JFK Airport", dropoff_zone="Midtown Center",
                  rate_category="standard", pickup_hour=14, pickup_dayofweek=2)
    kwargs.update(overrides)
    return predict.predict_fare(**kwargs)


def test_predict_fare_builds_model_input_aligned_to_training_columns(stubbed_predict):
    model, distance, duration = stubbed_predict

    _call()

    X = model.predict.call_args.args[0]
    assert list(X.columns) == TRAINING_COLUMNS
    assert len(X) == 1
    row = X.iloc[0]
    assert row["trip_distance"] == distance.loc[132, 161]
    assert row["trip_duration"] == duration.loc[132, 161]
    assert row["passenger_count"] == 2
    assert row["pickup_borough_Queens"] == 1
    assert row["pickup_borough_Manhattan"] == 0
    assert row["dropoff_borough_Manhattan"] == 1
    assert row["rate_category_standard"] == 1
    assert row["rate_category_negotiated"] == 0
    assert row["pickup_hour_14"] == 1
    assert row["pickup_dayofweek_2"] == 1


def test_predict_fare_uses_directional_lookup(stubbed_predict):
    model, distance, _ = stubbed_predict

    _call(pickup_zone="Midtown Center", dropoff_zone="JFK Airport")

    X = model.predict.call_args.args[0]
    assert X.iloc[0]["trip_distance"] == distance.loc[161, 132]
    assert distance.loc[161, 132] != distance.loc[132, 161]


def test_predict_fare_unseen_category_falls_back_to_all_zero_dummies(stubbed_predict):
    model, _, _ = stubbed_predict

    _call(pickup_hour=3, pickup_dayofweek=6, rate_category="other")

    X = model.predict.call_args.args[0]
    assert X.iloc[0]["pickup_hour_14"] == 0
    assert X.iloc[0]["pickup_dayofweek_2"] == 0
    assert X.iloc[0]["rate_category_standard"] == 0
    assert X.iloc[0]["rate_category_negotiated"] == 0


def test_predict_fare_returns_base_fare_surcharges_and_total(stubbed_predict):
    result = _call()

    expected_surcharges = predict.calculate_surcharges(132, 161, "Queens", "Manhattan", 14, 2)
    assert result["base_fare"] == 42.12
    assert result["surcharges"] == expected_surcharges
    assert result["fare_with_surcharges"] == round(42.123 + expected_surcharges["total_surcharges"], 2)
    assert isinstance(result["base_fare"], float)


@pytest.mark.parametrize("field", ["pickup_zone", "dropoff_zone"])
def test_predict_fare_rejects_unknown_zone(stubbed_predict, field):
    model, _, _ = stubbed_predict

    with pytest.raises(ValueError, match="Unrecognized zone name"):
        _call(**{field: "Atlantis"})

    model.predict.assert_not_called()


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------

def test_get_model_loads_once_and_caches(monkeypatch):
    fake_model = object()
    loader = MagicMock(return_value=fake_model)
    monkeypatch.setattr(predict, "load_production_model", loader)
    monkeypatch.setattr(predict, "_model", None)

    assert predict.get_model() is fake_model
    assert predict.get_model() is fake_model
    loader.assert_called_once()


def test_get_training_columns_loads_once_and_caches(monkeypatch):
    loader = MagicMock(return_value=["a", "b"])
    monkeypatch.setattr(predict, "load_training_columns", loader)
    monkeypatch.setattr(predict, "_training_columns", None)

    assert predict.get_training_columns() == ["a", "b"]
    predict.get_training_columns()
    loader.assert_called_once()


def test_load_production_model_sets_tracking_uri_and_loads(monkeypatch):
    # mlflow lazy-loads its flavour submodules, so swap the whole module reference
    fake_mlflow = MagicMock()
    fake_mlflow.pyfunc.load_model.return_value = "model"
    monkeypatch.setattr(predict, "mlflow", fake_mlflow)
    set_uri = fake_mlflow.set_tracking_uri
    load = fake_mlflow.pyfunc.load_model

    assert predict.load_production_model("models:/m@prod", "sqlite:///x.db") == "model"
    set_uri.assert_called_once_with("sqlite:///x.db")
    load.assert_called_once_with("models:/m@prod")


def test_load_training_columns_reads_committed_json():
    columns = predict.load_training_columns()

    assert columns[:3] == ["trip_duration", "passenger_count", "trip_distance"]
    assert len(columns) == len(set(columns))