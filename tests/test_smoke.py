"""
End-to-end smoke tests.

Unlike the unit tests, nothing here is mocked: these load the real committed
model file and lookup tables and call the FastAPI app the same way a user
would. They catch the problems unit tests can't, such as a missing model file,
a data file that isn't committed, or a broken path.

If any required artefact is missing, the whole module is skipped with a
message saying which file to add.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    PROJECT_ROOT / "models" / "xgb_model.json",
    PROJECT_ROOT / "data" / "model_training_columns.json",
    PROJECT_ROOT / "data" / "taxi_zone_lookup.csv",
    PROJECT_ROOT / "data" / "distance_lookup.csv",
    PROJECT_ROOT / "data" / "duration_lookup.csv",
]

_missing = [str(p.relative_to(PROJECT_ROOT)) for p in REQUIRED_FILES if not p.exists()]

pytestmark = pytest.mark.skipif(
    bool(_missing), reason=f"smoke test needs committed artefacts, missing: {_missing}"
)


@pytest.fixture(scope="module")
def client():
    # Import inside the fixture so a missing artefact gives a skip, not an
    # import error. Uses the same import path as `uvicorn src.interface:app`.
    from src.interface import app

    return TestClient(app)


def _payload(**overrides):
    payload = {
        "passenger_count": 1,
        "pickup_zone": "JFK Airport",
        "dropoff_zone": "Midtown Center",
        "rate_category": "standard",
        "pickup_hour": 11,
        "pickup_dayofweek": 2,  # Wednesday, outside rush hour
        "tip_percentage": 0.0,
    }
    payload.update(overrides)
    return payload


def test_ui_page_loads(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "Taxi Fare Predictor" in response.text


def test_zones_endpoint_returns_real_zone_list(client):
    response = client.get("/zones")

    assert response.status_code == 200
    zones = response.json()["zones"]
    assert len(zones) > 250  # TLC has ~260 zones
    assert "JFK Airport" in zones
    assert "Midtown Center" in zones


def test_jfk_to_midtown_fare_is_realistic(client):
    response = client.post("/predict", json=_payload())

    assert response.status_code == 200
    data = response.json()

    # A ~17 mile metered trip; loose bounds catch a broken model or lookup
    # (e.g. negative fares, or features in the wrong order) without being flaky.
    assert 30 <= data["base_fare"] <= 150
    assert data["fare_with_surcharges"] > data["base_fare"]
    assert data["total_fare"] == pytest.approx(data["fare_with_surcharges"])


def test_short_trip_is_cheaper_than_airport_trip(client):
    airport = client.post("/predict", json=_payload()).json()
    short = client.post(
        "/predict",
        json=_payload(pickup_zone="Upper East Side South", dropoff_zone="Upper East Side North"),
    ).json()

    assert 0 < short["base_fare"] < airport["base_fare"]


def test_tip_is_added_to_total(client):
    data = client.post("/predict", json=_payload(tip_percentage=20.0)).json()

    assert data["tip_amount"] == pytest.approx(0.2 * data["fare_with_surcharges"], abs=0.01)
    assert data["total_fare"] == pytest.approx(
        data["fare_with_surcharges"] + data["tip_amount"], abs=0.01
    )


def test_unknown_zone_returns_400(client):
    response = client.post("/predict", json=_payload(pickup_zone="Atlantis"))

    assert response.status_code == 400