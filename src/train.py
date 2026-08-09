import mlflow
mlflow.set_tracking_uri("sqlite:///C:/Dev/taxi-fare-predictor/mlflow.db")
mlflow.set_experiment("taxi-fare-prediction")