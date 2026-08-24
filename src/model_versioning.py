from mlflow.tracking import MlflowClient

def promote_run_to_production(run_id, model_name=None):
    client = MlflowClient()

    # Name the run
    client.set_tag(run_id, "mlflow.runName", "production")
    client.set_tag(run_id, "stage", "production")

    if model_name:
        model_uri = f"runs:/{run_id}/model"

        # Register model if not exists
        try:
            client.create_registered_model(model_name)
        except:
            pass  # already exists

        mv = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id
        )

        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage="Production"
        )

    return True
