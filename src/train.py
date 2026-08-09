import mlflow
mlflow.set_tracking_uri("sqlite:///C:/Dev/taxi-fare-predictor/mlflow.db")
mlflow.set_experiment("taxi-fare-prediction")

import sys
sys.path.append('../src')

import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from pipeline import build_dataset

mlflow.set_experiment("taxi-fare-prediction")

TIME_SORT_INDEX = 'tpep_pickup_datetime'

TARGET = 'fare_amount'
NUMERICAL_FEATURES = ['trip_duration', 'passenger_count', 'trip_distance']
CATEGORICAL_FEATURES = ['pickup_borough', 'dropoff_borough', 'rate_category', 'pickup_hour', 'pickup_dayofweek']


# XGBoost Parameters

N_ESTIMATORS = 300,
XGB_LEARNING_RATE = 0.05,
XGB_MAX_DEPTH = 6,
SUBSAMPLE=0.8,
COLSAMPLE_BYTREE=0.8,
OBJECTIVE = 'reg:squarederror'
N_JOBS = -1
RANDOM_STATE = 42

# Random Forest Parameters 

RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = None       
RF_MIN_SAMPLES_LEAF = 1
RF_MAX_FEATURES = "sqrt"


def time_sorted_split(df: pd.DataFrame, 
                      num_features: list, 
                      cat_features: list, 
                      target: str, 
                      sort_index: str,
                      test_proportion: float):
    df_sorted = df.sort_values(TIME_SORT_INDEX).reset_index(drop=True)
    X = pd.get_dummies(df_sorted[num_features + cat_features], columns=cat_features, drop_first=True)
    y = df_sorted[target]

    split_idx = int(len(df_sorted) * (1-test_proportion))

    X_train = X.iloc[:split_idx]
    X_test  = X.iloc[split_idx:]

    y_train = y.iloc[:split_idx]
    y_test  = y.iloc[split_idx:]

    return X_train, y_train, X_test, y_test

def train_linear_regession(X_train: pd.DataFrame, 
                           y_train: pd.Series, 
                           X_test: pd.DataFrame, 
                           y_test: pd.Series):

    with mlflow.start_run(run_name='LinearRegression'):
        model = LinearRegression()
        model.fit(X_train, y_train)
    
        preds = model.predict(X_test)
     
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
     
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
     
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
     
        mlflow.sklearn.log_model(model, 'model')
     
        print(f"RMSE: {rmse:.3f} | MAE: {mae:.3f} | R2: {r2:.3f}")

    return model, rmse, mae, r2

def train_XGBoost(X_train: pd.DataFrame,
                   y_train: pd.Series,
                   X_test: pd.DataFrame,
                   y_test: pd.Series,
                   n_estimators: int = N_ESTIMATORS,
                   learning_rate: float = XGB_LEARNING_RATE,
                   max_depth: int = XGB_MAX_DEPTH,
                   subsample: float = SUBSAMPLE,
                   colsample_bytree: float = COLSAMPLE_BYTREE,
                   objective: str = OBJECTIVE,
                   n_jobs: int = N_JOBS,
                   random_state: int = RANDOM_STATE):

    with mlflow.start_run(run_name='XGBoost'):
        model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective=objective,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        mlflow.log_param("model_type", "XGBRegressor")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("subsample", subsample)
        mlflow.log_param("colsample_bytree", colsample_bytree)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        mlflow.xgboost.log_model(model, "model")

        print(f"RMSE: {rmse:.3f} | MAE: {mae:.3f} | R2: {r2:.3f}")

    return model, rmse, mae, r2

def train_RandomForest(X_train: pd.DataFrame,
                        y_train: pd.Series,
                        X_test: pd.DataFrame,
                        y_test: pd.Series,
                        n_estimators: int = RF_N_ESTIMATORS,
                        max_depth: int = RF_MAX_DEPTH,
                        min_samples_leaf: int = RF_MIN_SAMPLES_LEAF,
                        max_features: str = RF_MAX_FEATURES):

    with mlflow.start_run(run_name='RandomForest'):
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=-1,
            random_state=42,
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(model, "model")

        print(f"RMSE: {rmse:.3f} | MAE: {mae:.3f} | R2: {r2:.3f}")

    return model, rmse, mae, r2