import pandas as pd
import json
import mlflow

FILL_VALUE = 0

COLUMN_NAME_PATH = '../data/model_training_columns.json'

MODEL_URI = 'models:/taxi-fare-xgboost@production'
TRACKING_URI = 'sqlite:///C:/Dev/taxi-fare-predictor/mlflow.db'

_model = None
_training_columns = None

def load_training_columns(path: str = COLUMN_NAME_PATH) -> list:
    with open(path, 'r') as f:
        return json.load(f)

def load_production_model(path: str = MODEL_URI, tracking_uri: str = TRACKING_URI):
    mlflow.set_tracking_uri(tracking_uri)
    model = mlflow.pyfunc.load_model(path)
    return model

def get_model():
    global _model
    if _model is None:
        _model = load_production_model()
    return _model

def get_training_columns():
    global _training_columns
    if _training_columns is None:
        _training_columns = load_training_columns()
    return _training_columns

def predict_fare(trip_duration: float,
                passenger_count: int,
                trip_distance: float,
                pickup_borough: str,
                dropoff_borough: str,
                rate_category: str,
                pickup_hour: int,
                pickup_dayofweek: int) -> float:

    df = pd.DataFrame({
    'trip_duration': [trip_duration],
    'passenger_count': [passenger_count],
    'trip_distance': [trip_distance],
    'pickup_borough': [pickup_borough],
    'dropoff_borough': [dropoff_borough],
    'rate_category': [rate_category],
    'pickup_hour': [pickup_hour],
    'pickup_dayofweek': [pickup_dayofweek]})

    column_names = get_training_columns()

    df_encoded = pd.get_dummies(df, columns=['pickup_borough', 'dropoff_borough', 'rate_category', 'pickup_hour', 'pickup_dayofweek'])
    df_aligned = df_encoded.reindex(columns=column_names, fill_value=FILL_VALUE)

    model = get_model()

    prediction = model.predict(df_aligned)

    return float(prediction[0])

