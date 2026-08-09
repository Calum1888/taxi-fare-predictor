"""
The file contains functions that engineer features from the data. These include:
- calculating the duration of the trip, hour of pickup and day of pickup
- add an airport flag which have flat rates (JFK = $70, Newark = $**)
- categories the rates which are unknown/null
These decisions are based on the EDA conducted on the dataset (../notebooks/eda.ipynb).
"""
import pandas as pd

def date_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a dataframe and calculates the duration of a taxi trip, the hour of pickup/dropoff and the day of pickup/dropoff.

    Arguments:
        df (pd.DataFrame): A dataframe that contains columns with date information on taxi trips.
    Returns:
        df (pd.DataFrame): A datafram with new duration, hour and day columns.

    """
    # duartion of trip column
    df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['pickup_dayofweek'] = df['tpep_pickup_datetime'].dt.dayofweek  # 0 = Monday, ... , 6 = Sunday

    return df
