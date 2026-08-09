"""
The file contains functions that engineer features from the data. These include:
- calculating the duration of the trip, hour of pickup and day of pickup
- add an airport flag which have flat rates (JFK = $70, Newark = $**)
- categories the rates which are unknown/null
These decisions are based on the EDA conducted on the dataset (../notebooks/eda.ipynb).
"""


RATECODE_MAP = {
    1: "standard",
    2: "jfk",
    3: "newark",
    4: "nassau_westchester",
    5: "negotiated",
    6: "other",
    99: "unknown",
}

import pandas as pd

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
    df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
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