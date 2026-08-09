import mlflow
mlflow.set_tracking_uri("sqlite:///C:/Dev/taxi-fare-predictor/mlflow.db")
mlflow.set_experiment("taxi-fare-prediction")

import sys
sys.path.append('../src')

import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from pipeline import build_dataset

mlflow.set_experiment("taxi-fare-prediction")

TIME_SORT_INDEX = 'tpep_pickup_datetime'

TARGET = 'fare_amount'
NUMERICAL_FEATURES = ['trip_duration', 'passenger_count', 'trip_distance']
CATEGORICAL_FEATURES = ['pickup_borough', 'dropoff_borough', 'rate_category', 'pickup_hour', 'pickup_dayofweek']

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

def train_linear_regession(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame):

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
     
        mlflow.sklearn.log_model(model, "model")
     
        print(f"RMSE: {rmse:.3f} | MAE: {mae:.3f} | R2: {r2:.3f}")

    return model, rmse, mae, r2

def train_XGBoost():

    with mlflow.start_run(run_name='XGBoost_test'):
        model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror'
        )

def train_RandomForest():