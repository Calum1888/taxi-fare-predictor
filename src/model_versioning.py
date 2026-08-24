from mlflow.tracking import MlflowClient

def promote_run_to_production(run_id, model_name=None):
    client = MlflowClient()

    client.set_tag(run_id, "mlflow.runName", "production")
    client.set_tag(run_id, "stage", "production")

    if model_name:
        model_uri = f"runs:/{run_id}/model"

        try:
            client.create_registered_model(model_name)
        except Exception:
            pass

        mv = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id
        )

        client.set_registered_model_alias(
            name=model_name,
            alias="production",
            version=mv.version
        )

    return True