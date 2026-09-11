"""
features.py

Feature engineering for the NYC taxi fare predictor.

Covers two stages of feature building:

1. Per-trip features derived from the raw pickup/dropoff timestamps and
   rate code — trip duration, pickup hour/day of week, a human-readable
   rate category, and an airport-flat-rate flag. These decisions are based
   on the EDA conducted on the dataset (../notebooks/eda.ipynb).

2. PU/DO zone-pair lookup tables (mean historical trip distance and
   duration between each pickup/dropoff zone), used both as candidate
   training features and, more importantly, as a stand-in for real
   trip_distance/trip_duration at inference time — since a user requesting
   a fare estimate can only supply pickup/dropoff zones, not an actual
   measured trip. See build_zone_lookup() for the fallback chain used to
   ensure every zone pair has a value.

Also provides time_sorted_split_df(), a shared chronological train/test
row-split helper used both when building model features and when building
zone lookups restricted to the training period (to avoid leaking
test-period trips into the lookup tables).
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
    Runs the full per-trip feature engineering pipeline: date/time features,
    rate category mapping, and the airport flag. Does not build or attach
    the zone-pair distance/duration lookups — see build_zone_lookup() and
    add_zone_features() for that separate step.

    Arguments:
        df (pd.DataFrame): Raw (cleaned) trip dataframe.

    Returns:
        pd.DataFrame: df with trip_duration, pickup_hour, pickup_dayofweek,
            rate_category, and is_airport_trip columns added.
    """
    df = date_time_features(df)
    df = add_rate_category(df)
    df = add_airport_flag(df)

    return df

def load_zone_lookup() -> pd.DataFrame:
    """
    Loads the raw taxi zone lookup table (LocationID, Borough, Zone, ...)
    from taxi_zone_lookup.csv.

    Returns:
        pd.DataFrame: The taxi zone lookup table, unfiltered.
    """
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
    """
    Fills missing PU->DO zone-pair values using the reverse DO->PU value,
    where available (e.g. if zone A->B was never observed but B->A was,
    A->B is filled with B->A as a reasonable approximation for a roughly
    symmetric quantity like distance).

    Arguments:
        matrix (pd.DataFrame): A PU x DO zone matrix, as from create_zone_matrix().

    Returns:
        pd.DataFrame: The matrix with reverse-direction values filled in
            where the forward direction was missing.
    """
    matrix = matrix.combine_first(matrix.T)
    return matrix


def intra_zone_fillna(matrix: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Fills missing same-zone (PU == DO) values on the matrix diagonal with a
    fixed default from INTRA_ZONE_DEFAULTS, since a same-zone trip has no
    "reverse trip" to fall back on via transpose_fillna().

    Arguments:
        matrix (pd.DataFrame): A PU x DO zone matrix.
        value_col (str): The column this matrix represents ('trip_distance'
            or 'trip_duration'), used to select the right default from
            INTRA_ZONE_DEFAULTS.

    Returns:
        pd.DataFrame: The matrix with any missing diagonal (same-zone)
            entries filled with the default value.
    """
    default = INTRA_ZONE_DEFAULTS[value_col]
    diag_indices = matrix.index.intersection(matrix.columns)
    for idx in diag_indices:
        if pd.isna(matrix.loc[idx, idx]):
            matrix.loc[idx, idx] = default
    return matrix


def borough_mean_fillna(df: pd.DataFrame, matrix: pd.DataFrame, lookup_ref: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Fills any remaining missing zone-pair values using the mean value_col
    for that pair's boroughs (e.g. any still-missing Queens->Brooklyn zone
    pair is filled with the overall Queens->Brooklyn mean), as a broader
    fallback for zone pairs with no direct or reverse-direction historical
    data.

    Arguments:
        df (pd.DataFrame): Trip data with pickup_borough, dropoff_borough,
            and value_col columns.
        matrix (pd.DataFrame): A PU x DO zone matrix, after transpose_fillna()
            and intra_zone_fillna() have already been applied.
        lookup_ref (pd.DataFrame): The taxi zone lookup table (LocationID,
            Borough, ...), used to map each zone ID to its borough.
        value_col (str): Column to aggregate for the borough-level fallback,
            e.g. 'trip_distance' or 'trip_duration'.

    Returns:
        pd.DataFrame: The matrix with remaining gaps filled from
            borough-level means.
    """
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

    Shares the same sort-and-split logic as time_sorted_split() (in
    train_and_tune.py), but returns the raw, unencoded rows rather than
    a model-ready feature matrix — used where the original columns
    (e.g. PULocationID, DOLocationID) are needed, such as building
    zone lookups restricted to the training period only.

    Arguments:
        df (pd.DataFrame): Trip dataframe to split.
        sort_index (str): Column to sort chronologically by, e.g. a
            pickup datetime column.
        test_proportion (float): Proportion of rows (by chronological
            position, not randomly) to hold out as the test set.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (df_train, df_test), covering
            the same rows as time_sorted_split() would for the same
            sort_index/test_proportion, but with all original columns intact.
    """
    df_sorted = df.sort_values(sort_index).reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_proportion))
    df_train = df_sorted.iloc[:split_idx]
    df_test = df_sorted.iloc[split_idx:]
    return df_train, df_test