from types import SimpleNamespace
from unittest.mock import MagicMock

import model_versioning


def test_promote_run_without_model_name_only_sets_production_tags(monkeypatch):
    client = MagicMock()
    monkeypatch.setattr(model_versioning, "MlflowClient", MagicMock(return_value=client))

    result = model_versioning.promote_run_to_production("run-123")

    assert result is True
    assert client.set_tag.call_args_list == [
        (("run-123", "mlflow.runName", "production"),),
        (("run-123", "stage", "production"),),
    ]
    client.create_registered_model.assert_not_called()
    client.create_model_version.assert_not_called()
    client.set_registered_model_alias.assert_not_called()


def test_promote_run_registers_version_and_sets_production_alias(monkeypatch):
    client = MagicMock()
    client.create_model_version.return_value = SimpleNamespace(version=7)
    monkeypatch.setattr(model_versioning, "MlflowClient", MagicMock(return_value=client))

    result = model_versioning.promote_run_to_production("run-123", "fare-model")

    assert result is True
    client.create_registered_model.assert_called_once_with("fare-model")
    client.create_model_version.assert_called_once_with(
        name="fare-model",
        source="runs:/run-123/model",
        run_id="run-123",
    )
    client.set_registered_model_alias.assert_called_once_with(
        name="fare-model",
        alias="production",
        version=7,
    )


def test_promote_run_continues_when_registered_model_already_exists(monkeypatch):
    client = MagicMock()
    client.create_registered_model.side_effect = Exception("already exists")
    client.create_model_version.return_value = SimpleNamespace(version=3)
    monkeypatch.setattr(model_versioning, "MlflowClient", MagicMock(return_value=client))

    result = model_versioning.promote_run_to_production("run-abc", "fare-model")

    assert result is True
    client.create_model_version.assert_called_once()
    client.set_registered_model_alias.assert_called_once_with(
        name="fare-model",
        alias="production",
        version=3,
    )
