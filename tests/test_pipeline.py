from unittest.mock import MagicMock

import pandas as pd

import pipeline


def test_build_dataset_calls_clean_then_engineer_with_default_arguments(monkeypatch):
    cleaned = pd.DataFrame({"raw": [1]})
    engineered = pd.DataFrame({"feature": [2]})
    load = MagicMock(return_value=cleaned)
    engineer = MagicMock(return_value=engineered)
    monkeypatch.setattr(pipeline, "load_and_clean", load)
    monkeypatch.setattr(pipeline, "engineer_features", engineer)

    result = pipeline.build_dataset()

    load.assert_called_once_with()
    engineer.assert_called_once_with(cleaned)
    assert result is engineered


def test_build_dataset_forwards_both_custom_paths(monkeypatch):
    cleaned = pd.DataFrame({"raw": [1]})
    load = MagicMock(return_value=cleaned)
    engineer = MagicMock(return_value=cleaned)
    monkeypatch.setattr(pipeline, "load_and_clean", load)
    monkeypatch.setattr(pipeline, "engineer_features", engineer)

    pipeline.build_dataset("trips.parquet", "zones.csv")

    load.assert_called_once_with(trips_path="trips.parquet", taxi_zones_path="zones.csv")


def test_build_dataset_forwards_only_path_that_is_supplied(monkeypatch):
    cleaned = pd.DataFrame({"raw": [1]})
    load = MagicMock(return_value=cleaned)
    monkeypatch.setattr(pipeline, "load_and_clean", load)
    monkeypatch.setattr(pipeline, "engineer_features", MagicMock(return_value=cleaned))

    pipeline.build_dataset(taxi_zones_path="zones.csv")

    load.assert_called_once_with(taxi_zones_path="zones.csv")
