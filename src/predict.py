"""
predict.py

Inference module for the NYC Taxi Fare Predictor.

Loads the production XGBoost model (via MLflow) and the supporting lookup
tables (zone names, zone->borough, zone->zone mean distance/duration) needed
to turn a user-facing request — passenger count, pickup/dropoff zone names,
rate category, and pickup time — into a fare estimate.

Because the model was trained on real per-trip trip_distance/trip_duration
values but end users only provide pickup/dropoff zones (no actual trip has
happened yet), this module substitutes historical zone-pair averages
(distance_lookup.csv, duration_lookup.csv) as stand-in features at inference
time. See predict_fare() for details.

Also estimates NYC taxi surcharges (MTA State Surcharge, Improvement
Surcharge, rush hour/overnight surcharges, NYS Congestion Surcharge, and the
MTA Congestion/CBD fee) on top of the model's predicted base fare, per TLC's
published taxi fare rules.

Usage:
    from predict import predict_fare, optional_tip
    result = predict_fare(passenger_count=2, pickup_zone='JFK Airport',
                           dropoff_zone='Midtown Center', rate_category='standard',
                           pickup_hour=14, pickup_dayofweek=2)
"""

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

MTA_STATE_SURCHARGE = 0.50
IMPROVEMENT_SURCHARGE = 1.00
RUSH_HOUR_SURCHARGE = 2.50
OVERNIGHT_SURCHARGE = 1.00
NYS_CONGESTION_SURCHARGE = 2.50   # Manhattan south of 96th St
CBD_CONGESTION_FEE = 0.75         # Manhattan south of 60th St

RUSH_HOUR_START = 16
RUSH_HOUR_END = 20
OVERNIGHT_START = 20
OVERNIGHT_END = 6

# Empirically derived from cbd_congestion_fee in training data (trips where the
# fee was charged on nearly every trip involving this zone, regardless of the
# other endpoint) — see notebook analysis.
CBD_ZONE_IDS = {1, 4, 12, 13, 45, 48, 50, 68, 79, 84, 87, 88, 90, 99, 100, 105, 107,
                 113, 114, 125, 137, 144, 148, 158, 161, 162, 163, 164, 170, 186,
                 199, 209, 211, 224, 229, 230, 231, 232, 233, 234, 246, 249, 261}
# zone 1 (EWR / Newark Airport) excluded — likely a low-sample artifact
CBD_ZONE_IDS.discard(1)

_model = None
_training_columns = None
_distance_lookup = None
_duration_lookup = None
_zone_name_to_id = None
_zone_name_to_borough = None


def load_training_columns():
    """
    Loads the ordered list of feature column names the production model was
    trained on, from model_training_columns.json. Used to align a freshly
    encoded inference-time dataframe to the exact columns (and column order)
    the model expects, filling any missing dummy columns with FILL_VALUE.

    Returns:
        list: Column names in the order used during training.
    """
    base_dir = Path(__file__).resolve().parent
    path = base_dir / ".." / "data" / "model_training_columns.json"

    with open(path, 'r') as f:
        return json.load(f)

def load_zone_matrix(filename: str) -> pd.DataFrame:
    """
    Loads a saved PU/DO zone lookup matrix (distance or duration) from CSV.

    Re-casts column labels to int, since the CSV round-trip via to_csv/
    read_csv stores column headers as strings by default, which would
    otherwise break integer-based .loc lookups.

    Arguments:
        filename (str): CSV filename under the data/ directory,
            e.g. 'distance_lookup.csv'.

    Returns:
        pd.DataFrame: Zone x zone matrix with integer index and columns.
    """
    base_dir = Path(__file__).resolve().parent
    path = base_dir / ".." / "data" / filename
    matrix = pd.read_csv(path, index_col=0)
    matrix.columns = matrix.columns.astype(int)
    return matrix

def load_production_model(path: str = MODEL_URI, tracking_uri: str = TRACKING_URI):
    """
    Loads the current production model from the MLflow model registry.

    Arguments:
        path (str): MLflow model URI, e.g. 'models:/taxi-fare-xgboost@production'.
        tracking_uri (str): MLflow tracking store URI.

    Returns:
        mlflow.pyfunc.PyFuncModel: The loaded model, ready for .predict().
    """
    mlflow.set_tracking_uri(tracking_uri)
    model = mlflow.pyfunc.load_model(path)
    return model

def get_model():
    """
    Returns the cached production model, loading it on first call.

    Uses a module-level cache (_model) so the (relatively expensive) MLflow
    model load only happens once per process, rather than on every prediction.

    Returns:
        mlflow.pyfunc.PyFuncModel: The cached production model.
    """
    global _model
    if _model is None:
        _model = load_production_model()
    return _model

def get_training_columns():
    """
    Returns the cached training column list, loading it on first call.

    Returns:
        list: Column names in the order used during training.
    """
    global _training_columns
    if _training_columns is None:
        _training_columns = load_training_columns()
    return _training_columns

def get_distance_lookup():
    """
    Returns the cached PU/DO mean trip distance matrix, loading it on first call.

    Returns:
        pd.DataFrame: Zone x zone matrix of mean_distance_miles.
    """
    global _distance_lookup
    if _distance_lookup is None:
        _distance_lookup = load_zone_matrix('distance_lookup.csv')
    return _distance_lookup

def get_duration_lookup():
    """
    Returns the cached PU/DO mean trip duration matrix, loading it on first call.

    Returns:
        pd.DataFrame: Zone x zone matrix of mean_duration_minutes.
    """
    global _duration_lookup
    if _duration_lookup is None:
        _duration_lookup = load_zone_matrix('duration_lookup.csv')
    return _duration_lookup

def load_zone_name_lookup() -> dict:
    """
    Builds a mapping from taxi zone name to LocationID, from taxi_zone_lookup.csv.

    Rows with a missing Zone name are dropped, since a NaN zone name would
    otherwise break sorting/lookup by name (mixed str/float comparison).

    Returns:
        dict: {zone_name: location_id}.
    """
    base_dir = Path(__file__).resolve().parent
    path = base_dir / ".." / "data" / "taxi_zone_lookup.csv"
    lookup_df = pd.read_csv(path)
    lookup_df = lookup_df.dropna(subset=["Zone"])

    return (
        lookup_df.set_index("Zone")["LocationID"]
        .to_dict()
    )

def load_zone_borough_lookup() -> dict:
    """
    Builds a mapping from taxi zone name to borough, from taxi_zone_lookup.csv.

    Rows with a missing Zone name are dropped (see load_zone_name_lookup);
    missing Borough values are filled as 'Unknown' rather than dropped, since
    the zone itself is still usable even if its borough is unlabeled.

    Returns:
        dict: {zone_name: borough}.
    """
    base_dir = Path(__file__).resolve().parent
    path = base_dir / ".." / "data" / "taxi_zone_lookup.csv"
    lookup_df = pd.read_csv(path)
    lookup_df = lookup_df.dropna(subset=["Zone"])

    return (
        lookup_df.set_index("Zone")["Borough"]
        .fillna("Unknown")
        .to_dict()
    )

def get_zone_name_to_id():
    """
    Returns the cached zone name -> LocationID mapping, building it on first call.

    Returns:
        dict: {zone_name: location_id}.
    """
    global _zone_name_to_id
    if _zone_name_to_id is None:
        _zone_name_to_id = load_zone_name_lookup()
    return _zone_name_to_id

def get_zone_name_to_borough():
    """
    Returns the cached zone name -> borough mapping, building it on first call.

    Returns:
        dict: {zone_name: borough}.
    """
    global _zone_name_to_borough
    if _zone_name_to_borough is None:
        _zone_name_to_borough = load_zone_borough_lookup()
    return _zone_name_to_borough


def calculate_surcharges(pickup_zone_id: int,
                          dropoff_zone_id: int,
                          pickup_borough: str,
                          dropoff_borough: str,
                          pickup_hour: int,
                          pickup_dayofweek: int) -> dict:
    """
    Estimates NYC taxi surcharges on top of the metered/predicted base fare,
    per TLC's published taxi fare rules.

    Includes flat per-trip surcharges (MTA State Surcharge, Improvement
    Surcharge), time-based surcharges (rush hour, overnight), and Manhattan
    congestion-related surcharges (NYS Congestion Surcharge, MTA Congestion/
    CBD fee).

    Known approximation: the NYS Congestion Surcharge (Manhattan south of
    96th St) is applied whenever pickup or dropoff borough is Manhattan,
    rather than checking the exact 96th St boundary — this over-includes
    trips to/from upper Manhattan (e.g. Harlem, Washington Heights). The
    CBD/60th St fee uses an exact, data-derived zone list (CBD_ZONE_IDS)
    and does not have this limitation.

    Arguments:
        pickup_zone_id (int): Pickup taxi zone LocationID.
        dropoff_zone_id (int): Dropoff taxi zone LocationID.
        pickup_borough (str): Pickup borough name.
        dropoff_borough (str): Dropoff borough name.
        pickup_hour (int): Pickup hour, 0-23.
        pickup_dayofweek (int): Pickup day of week, 0=Monday ... 6=Sunday.

    Returns:
        dict: Itemized surcharge amounts, plus 'total_surcharges'.
    """
    surcharges = {}

    surcharges['mta_state_surcharge'] = MTA_STATE_SURCHARGE
    surcharges['improvement_surcharge'] = IMPROVEMENT_SURCHARGE

    is_weekday = pickup_dayofweek <= 4
    is_rush_hour = is_weekday and (RUSH_HOUR_START <= pickup_hour < RUSH_HOUR_END)
    is_overnight = (pickup_hour >= OVERNIGHT_START) or (pickup_hour < OVERNIGHT_END)

    surcharges['rush_hour_surcharge'] = RUSH_HOUR_SURCHARGE if is_rush_hour else 0.0
    surcharges['overnight_surcharge'] = OVERNIGHT_SURCHARGE if is_overnight else 0.0

    touches_manhattan = pickup_borough == 'Manhattan' or dropoff_borough == 'Manhattan'
    surcharges['nys_congestion_surcharge'] = NYS_CONGESTION_SURCHARGE if touches_manhattan else 0.0

    touches_cbd = pickup_zone_id in CBD_ZONE_IDS or dropoff_zone_id in CBD_ZONE_IDS
    surcharges['cbd_congestion_fee'] = CBD_CONGESTION_FEE if touches_cbd else 0.0

    surcharges['total_surcharges'] = round(sum(surcharges.values()), 2)

    return surcharges


def predict_fare(passenger_count: int,
                pickup_zone: str,
                dropoff_zone: str,
                rate_category: str,
                pickup_hour: int,
                pickup_dayofweek: int) -> dict:
    """
    Predicts a taxi fare for a requested trip, from user-facing inputs alone.

    The production model was trained on real per-trip trip_distance/
    trip_duration values, but at prediction time no real trip exists yet —
    the user only supplies pickup/dropoff zone names. To bridge this gap,
    trip_distance and trip_duration are substituted with historical zone-pair
    averages from the saved distance/duration lookup tables. This introduces
    a train/inference distribution mismatch (the model learns from real,
    noisy per-trip values but predicts from smoothed averages), so real-world
    accuracy is expected to be somewhat lower than training/test metrics
    computed on real trip data.

    Pickup/dropoff boroughs are derived internally from the zone name via
    the zone->borough lookup, rather than being taken as separate inputs, so
    the caller cannot supply a zone/borough pair that don't match.

    Arguments:
        passenger_count (int): Number of passengers.
        pickup_zone (str): Pickup taxi zone name (must match taxi_zone_lookup.csv).
        dropoff_zone (str): Dropoff taxi zone name (must match taxi_zone_lookup.csv).
        rate_category (str): Fare rate category, e.g. 'standard', 'JFK', 'negotiated'.
        pickup_hour (int): Pickup hour, 0-23.
        pickup_dayofweek (int): Pickup day of week, 0=Monday ... 6=Sunday.

    Raises:
        ValueError: If pickup_zone or dropoff_zone isn't a recognized zone name.

    Returns:
        dict: {
            'base_fare': model-predicted fare before surcharges,
            'surcharges': itemized surcharge breakdown (see calculate_surcharges),
            'fare_with_surcharges': base_fare + total surcharges
        }
    """
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

    base_fare = float(model.predict(df_aligned)[0])

    surcharges = calculate_surcharges(
        pickup_zone_id, dropoff_zone_id, pickup_borough, dropoff_borough,
        pickup_hour, pickup_dayofweek
    )

    fare_with_surcharges = round(base_fare + surcharges['total_surcharges'], 2)

    return {
        'base_fare': round(base_fare, 2),
        'surcharges': surcharges,
        'fare_with_surcharges': fare_with_surcharges
    }


def optional_tip(fare: float, tip_percentage: float) -> float:
    """
    Calculates a tip amount for a given fare and tip percentage.

    Arguments:
        fare (float): Fare amount to base the tip on (typically fare_with_surcharges).
        tip_percentage (float): Tip percentage, e.g. 15 for 15%.

    Returns:
        float: Tip amount in the same currency units as fare.
    """
    tip = fare * (tip_percentage / 100)
    return tip