import pandas as pd
import json
import mlflow

from pathlib import Path

FILL_VALUE = 0

COLUMN_NAME_PATH = '../data/model_training_columns.json'
DISTANCE_LOOKUP_PATH = '../data/distance_lookup.csv'
DURATION_LOOKUP_PATH = '../data/duration_lookup.csv'

MODEL_URI = 'models:/taxi-fare-xgboost@production'
TRACKING_URI = 'sqlite:///C:/Dev/taxi-fare-predictor/mlflow.db'

_model = None
_training_columns = None
_distance_lookup = None
_duration_lookup = None
_zone_name_to_id = None
_zone_name_to_borough = None

def load_training_columns():
    base_dir = Path(__file__).resolve().parent
    path = base_dir / ".." / "data" / "model_training_columns.json"

    with open(path, 'r') as f:
        return json.load(f)

def load_zone_matrix(filename: str) -> pd.DataFrame:
    base_dir = Path(__file__).resolve().parent
    path = base_dir / ".." / "data" / filename
    matrix = pd.read_csv(path, index_col=0)
    matrix.columns = matrix.columns.astype(int)  # CSV round-trip turns column labels into strings
    return matrix

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

def get_distance_lookup():
    global _distance_lookup
    if _distance_lookup is None:
        _distance_lookup = load_zone_matrix('distance_lookup.csv')
    return _distance_lookup

def get_duration_lookup():
    global _duration_lookup
    if _duration_lookup is None:
        _duration_lookup = load_zone_matrix('duration_lookup.csv')
    return _duration_lookup

def load_zone_name_lookup() -> dict:
    base_dir = Path(__file__).resolve().parent
    path = base_dir / ".." / "data" / "taxi_zone_lookup.csv"
    lookup_df = pd.read_csv(path)

    return (
        lookup_df.set_index("Zone")["LocationID"]
        .to_dict()
    )

def get_zone_name_to_id():
    global _zone_name_to_id
    if _zone_name_to_id is None:
        _zone_name_to_id = load_zone_name_lookup()
    return _zone_name_to_id

def load_zone_name_lookup() -> dict:
    base_dir = Path(__file__).resolve().parent
    path = base_dir / ".." / "data" / "taxi_zone_lookup.csv"
    lookup_df = pd.read_csv(path)

    return (
        lookup_df.set_index("Zone")["LocationID"]
        .to_dict()
    )

def load_zone_borough_lookup() -> dict:
    base_dir = Path(__file__).resolve().parent
    path = base_dir / ".." / "data" / "taxi_zone_lookup.csv"
    lookup_df = pd.read_csv(path)

    return (
        lookup_df.set_index("Zone")["Borough"]
        .fillna("Unknown")
        .to_dict()
    )

def get_zone_name_to_id():
    global _zone_name_to_id
    if _zone_name_to_id is None:
        _zone_name_to_id = load_zone_name_lookup()
    return _zone_name_to_id

def get_zone_name_to_borough():
    global _zone_name_to_borough
    if _zone_name_to_borough is None:
        _zone_name_to_borough = load_zone_borough_lookup()
    return _zone_name_to_borough

def predict_fare(passenger_count: int,
                pickup_zone: str,
                dropoff_zone: str,
                rate_category: str,
                pickup_hour: int,
                pickup_dayofweek: int) -> float:

    zone_name_to_id = get_zone_name_to_id()
    zone_name_to_borough = get_zone_name_to_borough()

    try:
        pickup_zone_id = zone_name_to_id[pickup_zone]
        dropoff_zone_id = zone_name_to_id[dropoff_zone]
        pickup_borough = zone_name_to_borough[pickup_zone]
        dropoff_borough = zone_name_to_borough[dropoff_zone]
    except KeyError as e:
        raise ValueError(f"Unrecognized zone name: {e}")

    distance_lookup = get_distance_lookup()
    duration_lookup = get_duration_lookup()

    trip_distance = distance_lookup.loc[pickup_zone_id, dropoff_zone_id]
    trip_duration = duration_lookup.loc[pickup_zone_id, dropoff_zone_id]

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

def optional_tip(fare: float, tip_percentage: float) -> float:

    tip = fare * (tip_percentage/100)

    return tip