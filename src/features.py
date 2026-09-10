"""
The file contains functions that engineer features from the data. These include:
- calculating the duration of the trip, hour of pickup and day of pickup
- add an airport flag which have flat rates (JFK = $70, Newark = $**)
- categories the rates which are unknown/null
These decisions are based on the EDA conducted on the dataset (../notebooks/eda.ipynb).
"""

import pandas as pd

from pathlib import Path

TIME_CONVERSION = 60 # dividing by 60 gives journey time in minutes
INTRA_ZONE_DEFAULTS = {
    'trip_distance': 0.8,      # miles, adjust to your existing value
    'trip_duration': 8.0,      # minutes, adjust as appropriate
}

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

def load_zone_lookup() -> pd.DataFrame:
    base_dir = Path(__file__).resolve().parent
    path = base_dir / ".." / "data" / "taxi_zone_lookup.csv"
    return pd.read_csv(path)


def load_zone_lookup() -> pd.DataFrame:
    base_dir = Path(__file__).resolve().parent
    path = base_dir / ".." / "data" / "taxi_zone_lookup.csv"
    return pd.read_csv(path)


def create_zone_matrix(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Builds a PU x DO zone matrix of mean values for the given column.

    Arguments:
        df (pd.DataFrame): Trip data with PULocationID, DOLocationID and value_col.
        value_col (str): Column to aggregate, e.g. 'trip_distance' or 'trip_duration'.

    Returns:
        pd.DataFrame: Zone x zone matrix of mean value_col.
    """
    lookup = (
        df.groupby(["PULocationID", "DOLocationID"])[value_col]
        .mean()
        .reset_index()
        .rename(columns={value_col: f"mean_{value_col}"})
    )

    matrix = lookup.pivot(
        index="PULocationID", columns="DOLocationID", values=f"mean_{value_col}"
    )

    return matrix


def transpose_fillna(matrix: pd.DataFrame) -> pd.DataFrame:
    matrix = matrix.combine_first(matrix.T)
    return matrix


def intra_zone_fillna(matrix: pd.DataFrame, value_col: str) -> pd.DataFrame:
    default = INTRA_ZONE_DEFAULTS[value_col]
    diag_indices = matrix.index.intersection(matrix.columns)
    for idx in diag_indices:
        if pd.isna(matrix.loc[idx, idx]):
            matrix.loc[idx, idx] = default
    return matrix


def borough_mean_fillna(df: pd.DataFrame, matrix: pd.DataFrame, lookup_ref: pd.DataFrame, value_col: str) -> pd.DataFrame:

    zone_to_boro = (
        lookup_ref.set_index("LocationID")["Borough"]
        .fillna("Unknown")
        .to_dict()
    )

    all_zones = list(range(1, 266))

    boro_matrix = df.groupby(["pickup_borough", "dropoff_borough"])[value_col].mean().unstack()

    zone_boro = pd.Series(all_zones, index=all_zones).map(zone_to_boro)
    boro_fallback = boro_matrix.reindex(index=zone_boro, columns=zone_boro)
    boro_fallback.index = all_zones
    boro_fallback.columns = all_zones

    matrix = matrix.fillna(boro_fallback)

    return matrix


def build_zone_lookup(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Builds a fully-filled PU/DO zone matrix for the given value column.

    Pipeline:
        1. Mean historical value per PU/DO zone pair.
        2. Fill missing pairs using the reverse (DO->PU) trip if available.
        3. Fill missing same-zone (intra-zone) trips with a default value.
        4. Fill any remaining gaps using borough-level mean.
        5. Fill any final gaps with the global mean as a last resort.

    Arguments:
        df (pd.DataFrame): Trip data with PULocationID, DOLocationID, value_col,
            pickup_borough and dropoff_borough columns.
        value_col (str): Column to aggregate, e.g. 'trip_distance' or 'trip_duration'.

    Returns:
        pd.DataFrame: A fully-filled zone x zone matrix, with no remaining NaN values.
    """
    lookup_ref = load_zone_lookup()

    matrix = create_zone_matrix(df, value_col)
    matrix = transpose_fillna(matrix)
    matrix = intra_zone_fillna(matrix, value_col)
    matrix = borough_mean_fillna(df, matrix, lookup_ref, value_col)

    global_mean = matrix.stack().mean()
    matrix = matrix.fillna(global_mean)

    assert matrix.isna().sum().sum() == 0, f"{value_col} matrix still contains NaNs after fallback chain"

    return matrix


def add_zone_features(df: pd.DataFrame, zone_lookup: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    """
    Merges a precomputed PU/DO zone lookup onto a trip dataframe.

    Arguments:
        df (pd.DataFrame): Trip data containing PULocationID and DOLocationID.
        zone_lookup (pd.DataFrame): Zone x zone matrix from build_zone_lookup().
        feature_name (str): Name for the resulting merged column.

    Returns:
        pd.DataFrame: df with the merged feature column added.
    """
    return df.merge(
        zone_lookup.stack().rename(feature_name),
        left_on=['PULocationID', 'DOLocationID'],
        right_index=True,
        how='left'
    )

def time_sorted_split_df(df: pd.DataFrame, sort_index: str, test_proportion: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sorts a dataframe by date/time and splits it into train/test row sets,
    without selecting or encoding any features.
    """
    df_sorted = df.sort_values(sort_index).reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_proportion))
    df_train = df_sorted.iloc[:split_idx]
    df_test = df_sorted.iloc[split_idx:]
    return df_train, df_test