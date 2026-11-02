from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

import train_and_tune
from train_and_tune import (
    BaseTrainer,
    LinearRegressionTrainer,
    RandomForestTrainer,
    TunableTrainer,
    XGBTrainer,
    time_sorted_split,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def trips():
    """10 trips deliberately out of chronological order."""
    pickup = pd.date_range("2026-01-01", periods=10, freq="h")
    order = [3, 7, 0, 9, 1, 5, 8, 2, 6, 4]
    return pd.DataFrame({
        "tpep_pickup_datetime": pickup[order],
        "trip_distance": np.array(order, dtype=float) + 1,
        "rate_category": ["standard", "jfk", "standard", "negotiated", "jfk",
                          "standard", "standard", "jfk", "negotiated", "standard"],
        "fare_amount": np.array(order, dtype=float) * 3 + 5,
    })


@pytest.fixture
def linear_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=60), "b": rng.normal(size=60)})
    y = pd.Series(2 * X["a"] - 3 * X["b"] + 1)
    return X.iloc[:45], y.iloc[:45], X.iloc[45:], y.iloc[45:]


# ---------------------------------------------------------------------------
# time_sorted_split
# ---------------------------------------------------------------------------

def _split(df, test_proportion=0.2):
    return time_sorted_split(df, num_features=["trip_distance"], cat_features=["rate_category"],
                             target="fare_amount", sort_index="tpep_pickup_datetime",
                             test_proportion=test_proportion)


def test_split_sizes_follow_test_proportion(trips):
    X_train, y_train, X_test, y_test = _split(trips, 0.2)

    assert len(X_train) == len(y_train) == 8
    assert len(X_test) == len(y_test) == 2


def test_split_is_chronological_with_test_after_train(trips):
    X_train, y_train, X_test, y_test = _split(trips)

    # trip_distance encodes chronological position (+1), so it must be sorted
    assert X_train["trip_distance"].tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert X_test["trip_distance"].tolist() == [9, 10]
    assert X_train["trip_distance"].max() < X_test["trip_distance"].min()


def test_split_keeps_features_and_target_aligned(trips):
    X_train, y_train, X_test, y_test = _split(trips)

    for X, y in [(X_train, y_train), (X_test, y_test)]:
        expected = (X["trip_distance"] - 1) * 3 + 5
        assert y.tolist() == expected.tolist()


def test_split_one_hot_encodes_with_drop_first_and_excludes_target_and_sort_col(trips):
    X_train, _, X_test, _ = _split(trips)

    assert list(X_train.columns) == list(X_test.columns)
    assert set(X_train.columns) == {"trip_distance", "rate_category_negotiated",
                                    "rate_category_standard"}
    assert "rate_category_jfk" not in X_train.columns  # dropped baseline
    assert "fare_amount" not in X_train.columns
    assert "tpep_pickup_datetime" not in X_train.columns


def test_split_does_not_mutate_input(trips):
    before = trips.copy()

    _split(trips)

    pd.testing.assert_frame_equal(trips, before)


def test_split_matches_features_time_sorted_split_df(trips):
    """Both split helpers must select the same rows, or the zone lookups leak."""
    import features

    df_train, df_test = features.time_sorted_split_df(trips, "tpep_pickup_datetime", 0.2)
    X_train, _, X_test, _ = _split(trips, 0.2)

    assert df_train["trip_distance"].tolist() == X_train["trip_distance"].tolist()
    assert df_test["trip_distance"].tolist() == X_test["trip_distance"].tolist()


# ---------------------------------------------------------------------------
# BaseTrainer
# ---------------------------------------------------------------------------

def test_train_fits_model_and_sets_best_model(linear_data):
    X_train, y_train, _, _ = linear_data
    trainer = BaseTrainer(LinearRegression(), "LinearRegression")

    trainer.train(X_train, y_train)

    assert trainer.best_model is trainer.model
    np.testing.assert_allclose(trainer.best_model.coef_, [2, -3], atol=1e-8)


def test_evaluate_returns_all_metrics_and_stores_them(linear_data):
    trainer = BaseTrainer(LinearRegression(), "LinearRegression")
    trainer.train(*linear_data[:2])

    metrics = trainer.evaluate(*linear_data)

    assert set(metrics) == {"train_rmse", "test_rmse", "train_mae", "test_mae",
                            "train_r2", "test_r2", "rmse_gap"}
    assert trainer.metrics is metrics
    # noiseless linear data -> perfect fit
    assert metrics["train_rmse"] == pytest.approx(0, abs=1e-8)
    assert metrics["test_rmse"] == pytest.approx(0, abs=1e-8)
    assert metrics["test_r2"] == pytest.approx(1.0)


def test_evaluate_metric_values_against_known_predictions():
    model = MagicMock()
    model.predict.side_effect = [np.array([1.0, 2.0]), np.array([0.0, 0.0])]
    trainer = BaseTrainer(model, "fake")

    metrics = trainer.evaluate(None, np.array([1.0, 2.0]), None, np.array([3.0, 4.0]))

    assert metrics["train_rmse"] == pytest.approx(0.0)
    assert metrics["test_rmse"] == pytest.approx(np.sqrt((9 + 16) / 2))
    assert metrics["test_mae"] == pytest.approx(3.5)
    assert metrics["rmse_gap"] == pytest.approx(metrics["test_rmse"] - metrics["train_rmse"])


def test_log_to_mlflow_logs_type_params_metrics_and_model(monkeypatch):
    # mlflow lazy-loads its flavour submodules, so swap the whole module reference
    fake_mlflow = MagicMock()
    monkeypatch.setattr(train_and_tune, "mlflow", fake_mlflow)
    log_param = fake_mlflow.log_param
    log_metric = fake_mlflow.log_metric
    log_model = fake_mlflow.sklearn.log_model

    trainer = BaseTrainer(LinearRegression(), "LinearRegression")
    trainer.metrics = {"test_rmse": 1.5, "test_r2": 0.9}

    trainer.log_to_mlflow({"max_depth": 4, "eta": 0.1})

    assert log_param.call_args_list[0].args == ("model_type", "LinearRegression")
    logged_params = {c.args[0]: c.args[1] for c in log_param.call_args_list}
    assert logged_params == {"model_type": "LinearRegression", "max_depth": 4, "eta": 0.1}
    logged_metrics = {c.args[0]: c.args[1] for c in log_metric.call_args_list}
    assert logged_metrics == {"test_rmse": 1.5, "test_r2": 0.9}
    log_model.assert_called_once()
    assert log_model.call_args.args[0] is trainer.best_model
    assert log_model.call_args.kwargs["name"] == "model"


def test_print_results_formats_metrics(capsys):
    trainer = BaseTrainer(LinearRegression(), "LinearRegression")
    trainer.metrics = {"train_rmse": 1.0, "test_rmse": 1.25, "rmse_gap": 0.25,
                       "train_r2": 0.95, "test_r2": 0.9}

    trainer.print_results()

    out = capsys.readouterr().out
    assert "Train RMSE: 1.000 | Test RMSE: 1.250 | Gap: 0.250" in out
    assert "Train R2: 0.950 | Test R2: 0.900" in out


# ---------------------------------------------------------------------------
# TunableTrainer + subclasses
# ---------------------------------------------------------------------------

def test_tune_sets_best_model_params_and_cv_results(linear_data, monkeypatch):
    monkeypatch.setattr(train_and_tune, "N_JOBS", 1)
    X_train, y_train, X_test, y_test = linear_data
    trainer = XGBTrainer(n_estimators=20)
    grid = {"max_depth": [2, 3], "learning_rate": [0.1, 0.3]}

    best = trainer.tune(X_train, y_train, grid, n_iter=2, cv=3,
                        scoring="neg_root_mean_squared_error")

    assert best is trainer.best_params_
    assert set(best) == {"max_depth", "learning_rate"}
    assert best["max_depth"] in grid["max_depth"]
    assert best["learning_rate"] in grid["learning_rate"]
    assert len(trainer.cv_results_["params"]) == 2
    assert trainer.best_model is not trainer.model  # refitted clone
    assert trainer.best_model.get_params()["max_depth"] == best["max_depth"]
    # tuned model is usable by evaluate()
    assert "test_rmse" in trainer.evaluate(X_train, y_train, X_test, y_test)


def test_tune_is_reproducible(linear_data, monkeypatch):
    monkeypatch.setattr(train_and_tune, "N_JOBS", 1)
    X_train, y_train, _, _ = linear_data
    grid = {"max_depth": [2, 3, 4, 5], "learning_rate": [0.05, 0.1, 0.3]}

    results = [
        XGBTrainer(n_estimators=10).tune(X_train, y_train, grid, n_iter=3, cv=2, scoring="r2")
        for _ in range(2)
    ]

    assert results[0] == results[1]


@pytest.mark.parametrize("trainer_cls,model_cls,name", [
    (LinearRegressionTrainer, LinearRegression, "LinearRegression"),
    (XGBTrainer, XGBRegressor, "XGBRegressor"),
    (RandomForestTrainer, RandomForestRegressor, "RandomForestRegressor"),
])
def test_subclasses_wrap_correct_model(trainer_cls, model_cls, name):
    trainer = trainer_cls()

    assert isinstance(trainer, TunableTrainer)
    assert isinstance(trainer.model, model_cls)
    assert trainer.model_name == name
    assert trainer.best_model is trainer.model
    assert trainer.best_params_ is None
    assert trainer.metrics == {}


@pytest.mark.parametrize("trainer_cls", [XGBTrainer, RandomForestTrainer])
def test_subclass_kwargs_are_passed_to_model(trainer_cls):
    trainer = trainer_cls(n_estimators=7, random_state=1)

    params = trainer.model.get_params()
    assert params["n_estimators"] == 7
    assert params["random_state"] == 1