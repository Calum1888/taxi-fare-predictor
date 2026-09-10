"""
The file contains functions that engineer features from the data. These include:
- calculating the duration of the trip, hour of pickup and day of pickup
- add an airport flag which have flat rates (JFK = $70, Newark = $**)
- categories the rates which are unknown/null
These decisions are based on the EDA conducted on the dataset (../notebooks/eda.ipynb).
"""

import pandas as pd

TIME_CONVERSION = 60 # dividing by 60 gives journey time in minutes
INTRA_ZONE_DEFAULT = 0.8

RATECODE_MAP = {
    1: "standard",
    2: "jfk",
    3: "newark",
    4: "nassau_westchester",
    5: "negotiated",
    6: "other",
    99: "unknown",
}

def date_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a dataframe and calculates the duration of a taxi trip, the hour of pickup/dropoff and the day of pickup/dropoff.

    Arguments:
        df (pd.DataFrame): A dataframe that contains columns with date information on taxi trips.
    Returns:
        df (pd.DataFrame): A datafram with new duration, hour and day columns.

    """
    df = df.copy()
    # duartion of trip column
    df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / TIME_CONVERSION
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['pickup_dayofweek'] = df['tpep_pickup_datetime'].dt.dayofweek  # 0 = Monday, ... , 6 = Sunday

    return df

def add_rate_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps the area rate codes to their numbers and adds unknown to ones wthout a code.
    Arugemnts:
        df (pd.DataFrame): A dataframe with RatecodeID columns.
    Returns:
        df (pd.DataFrame): Newly mapped area codes.
    """
    df = df.copy()
    df["rate_category"] = df["RatecodeID"].map(RATECODE_MAP).fillna("unknown")
    return df


def add_airport_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag trips associated with any flat-rate airport code (JFK/Newark).

    Arugemnts:
        df (pd.DataFrame): A dataframe.
    Returns:
        df (pd.Dataframe): A dataframe with flags for airport trips.
    """
    df = df.copy()
    df["is_airport_trip"] = df["rate_category"].isin(["jfk", "newark"])
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs all the engineering features.
    """
    df = date_time_features(df)
    df = add_rate_category(df)
    df = add_airport_flag(df)

    return df

def create_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:

    distance_lookup = (
    df.groupby(["PULocationID", "DOLocationID"])["trip_distance"]
    .mean()
    .reset_index()
    .rename(columns={"trip_distance": "mean_distance_miles"})
    )

    distance_matrix = distance_lookup.pivot(
        index="PULocationID", columns="DOLocationID", values="mean_distance_miles"
        )

    return distance_matrix

def transpose_fillna(distance_matrix: pd.DataFrame) -> pd.DataFrame:

    distance_matrix = distance_matrix.combine_first(distance_matrix.T)

    return distance_matrix

def intra_zone_fillna(distance_matrix: pd.DataFrame) -> pd.DataFrame:

    diag_indices = distance_matrix.index.intersection(distance_matrix.columns)
    for idx in diag_indices:
        if pd.isna(distance_matrix.loc[idx, idx]):
            distance_matrix.loc[idx, idx] = INTRA_ZONE_DEFAULT

    return distance_matrix

def borough_mean_fillna(df: pd.DataFrame, distance_matrix: pd.DataFrame, lookup_ref: pd.DataFrame) -> pd.DataFrame:

    lookup_df = pd.read_csv('../data/taxi_zone_lookup.csv')

# Create dictionary mapping LocationID -> Borough (filling missing boroughs as 'Unknown')
    zone_to_boro = (
        lookup_df.set_index("LocationID")["Borough"]
        .fillna("Unknown")
        .to_dict()
        )

    all_zones = list(range(1, 266))

    boro_matrix = df.groupby(["pickup_borough", "dropoff_borough"])["trip_distance"].mean().unstack()

    # Construct full borough fallback grid
    boro_fallback = pd.DataFrame(index=all_zones, columns=all_zones, dtype=float)
    for pu in all_zones:
        for do in all_zones:
            pu_boro, do_boro = zone_to_boro.get(pu), zone_to_boro.get(do)
            if pu_boro in boro_matrix.index and do_boro in boro_matrix.columns:
                boro_fallback.loc[pu, do] = boro_matrix.loc[pu_boro, do_boro]

    distance_matrix = distance_matrix.fillna(boro_fallback)

    return distance_matrix
