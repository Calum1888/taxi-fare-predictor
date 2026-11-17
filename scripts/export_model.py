"""
export_model.py

One-off script: exports the production model from the local MLflow registry
to models/xgb_model.json, so it can be committed and loaded by src/predict.py
without needing the MLflow tracking database.

Re-run this whenever a new model is promoted to the 'production' alias.

Usage (from the repository root):
    python scripts/export_model.py
    python scripts/export_model.py --tracking-uri sqlite:///path/to/mlflow.db
"""

import argparse
from pathlib import Path

import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
MODEL_URI = "models:/taxi-fare-xgboost@production"
OUTPUT_PATH = PROJECT_ROOT / "models" / "xgb_model.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--model-uri", default=MODEL_URI)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    model = mlflow.sklearn.load_model(args.model_uri)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(args.output)
    print(f"Saved {type(model).__name__} to {args.output}")


if __name__ == "__main__":
    main()