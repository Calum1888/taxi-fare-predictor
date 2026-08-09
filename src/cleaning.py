import duckdb
import pandas as pd

FARE_PATH = '../data/yellow_tripdata_2026-01.parquet'
ZONE_PATH = '../data/taxi_zone_lookup.csv'

PASSENGER_COUNT_MIN = 0

# distance filters
TRIP_DISTANCE_MIN = 0
TRIP_DISTANCE_MAX = 50

# fare amount filters
FARE_AMOUNT_MIN = 0
FARE_AMOUNT_MAX = 120

# duration filters
TRIP_DURATION_MIN = 0
TRIP_DURATION_MAX = 120




def load_and_clean(trips_path: str, taxi_zones_path: str) -> pd.DataFrame:
    """
    The function takes a path to taxi trip data and returns a cleaned dataframe for feature engineering based on the criteria in the SQL query.

    Arguments:
        trips_path (str): A path to trip data.
        taxi_zones_path (str): A path to zone data.
    Returns:
        df (pd.DataFrame) A cleaned dataframe of trip information.
    """

    query = f"""
        SELECT
            trips.* EXCLUDE (store_and_fwd_flag),
            pu_zones.Borough AS pickup_borough,
            do_zones.Borough AS dropoff_borough
        FROM '{trips_path}' AS trips
        LEFT JOIN '{taxi_zones_path}' AS pu_zones
            ON trips.PULocationID = pu_zones.LocationID
        LEFT JOIN '{taxi_zones_path}' AS do_zones
            ON trips.DOLocationID = do_zones.LocationID
        WHERE
            trips.passenger_count > 0
            AND trips.trip_distance > {TRIP_DISTANCE_MIN}
            AND trips.trip_distance < {TRIP_DISTANCE_MAX}
            AND trips.fare_amount > {FARE_AMOUNT_MIN}
            AND trips.fare_amount < {FARE_AMOUNT_MAX}
            AND EXTRACT(EPOCH FROM (trips.tpep_dropoff_datetime - trips.tpep_pickup_datetime)) / 60
                BETWEEN {TRIP_DURATION_MIN} AND {TRIP_DURATION_MAX}
    """

    df = duckdb.sql(query).df()

    df = df.drop_duplicates()

    return df