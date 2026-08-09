"""
End-to-end data preparation pipeline: load -> clean -> engineer features.

Usage:
    from pipeline import build_dataset
    df = build_dataset()
"""

import pandas as pd

from cleaning import load_and_clean
from features import engineer_features


def build_dataset(trips_path: str = None, taxi_zones_path: str = None) -> pd.DataFrame:
    kwargs = {}
    if trips_path:
        kwargs["trips_path"] = trips_path
    if taxi_zones_path:
        kwargs["zones_path"] = taxi_zones_path

    df = load_and_clean(**kwargs)
    df = engineer_features(df)
    return df


if __name__ == "__main__":
    df = build_dataset()
    print(f"Final shape: {df.shape}")
    print(df.dtypes)