# TODO: Write docstring for this file 

# data malnipulation and computation
import numpy as np
import pandas as pd

# experiment tracking
import mlflow

# models
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# cross validation
from sklearn.model_selection import RandomizedSearchCV

# parameters
N_JOBS = -1
RANDOM_STATE = 42

def time_sorted_split(df: pd.DataFrame, 
                      num_features: list, 
                      cat_features: list, 
                      target: str, 
                      sort_index: str,
                      test_proportion: float) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Sorts the taxi trips by date and time. The data is then split into train and test sets.
    Test data is always later than training data for accurate prediction of future fares.

    Arguments:
        df (pd.DataFrame) A dataframe of features and target column.
        num_features (list) A list of numerical feature column names.
        cat_features (list) A list of categorical feature column names.
        target (str) A string of the name if the target column ('fare_amount').
        sort_index (str) The way the dataframe is sorted (by date/time).
        test_proportion (float) The proportion of the data used for testing.
    Returns:
        X_train (pd.DataFrame) The dataframe of features for training.
        y_train (pd.Series) The series of target column for training.
        X_test (pd.DataFrame) The dataframe of features for testing.
        y_test (pd.Series) The series of the target column for testing.

    """
    # sort taxi trips based on date
    df_sorted = df.sort_values(sort_index).reset_index(drop=True)

    # split into feature dataframe and target column
    X = pd.get_dummies(df_sorted[num_features + cat_features], columns=cat_features, drop_first=True)
    y = df_sorted[target]

    # index to split into test and train
    split_idx = int(len(df_sorted) * (1-test_proportion))

    # train, test plit
    X_train = X.iloc[:split_idx]
    X_test  = X.iloc[split_idx:]

    y_train = y.iloc[:split_idx]
    y_test  = y.iloc[split_idx:]

    return X_train, y_train, X_test, y_test

class BaseTrainer():
    # TODO: write docstrings for methods and class
    def __init__(self, model, model_name: str):
        self.model = model
        self.model_name = model_name
        self.best_model = model
        self.best_params_ = None
        self.cv_results_ = None
        self.metrics = {}

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.best_model = self.model
        

    def evaluate(self, X_train, y_train, X_test, y_test):
        train_preds = self.best_model.predict(X_train)
        test_preds = self.best_model.predict(X_test)

        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))

        self.metrics = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': mean_absolute_error(y_train, train_preds),
            'test_mae': mean_absolute_error(y_test, test_preds),
            'train_r2': r2_score(y_train, train_preds),
            'test_r2': r2_score(y_test, test_preds),
            'rmse_gap': test_rmse - train_rmse
            }

        return self.metrics

        
    
    def log_to_mlflow(self, params: dict):
        mlflow.log_param("model_type", self.model_name)

        for k, v in params.items():
            mlflow.log_param(k, v)

        for k, v in self.metrics.items():
            mlflow.log_metric(k, v)

        mlflow.sklearn.log_model(self.best_model, name="model",
                                 skops_trusted_types=[
            "xgboost.core.Booster",
            "xgboost.sklearn.XGBRegressor"
        ])

    def print_results(self):
        print(
            f"Train RMSE: {self.metrics['train_rmse']:.3f} | "
            f"Test RMSE: {self.metrics['test_rmse']:.3f} | "
            f"Gap: {self.metrics['rmse_gap']:.3f}"
        )
        print(
            f"Train R2: {self.metrics['train_r2']:.3f} | "
            f"Test R2: {self.metrics['test_r2']:.3f}"
        )

class TunableTrainer(BaseTrainer):
    # TODO: write docstrings for methods and class
    def tune(self, X_train, y_train, param_distributions, n_iter, cv, scoring):
        search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE
        )

        search.fit(X_train, y_train)

        self.best_model = search.best_estimator_
        self.best_params_ = search.best_params_
        self.cv_results_ = search.cv_results_

        return self.best_params_

class LinearRegressionTrainer(TunableTrainer):
    # TODO: write docstrings for methods and class
    def __init__(self):
        super().__init__(LinearRegression(), "LinearRegression")

class XGBTrainer(TunableTrainer):
    # TODO: write docstrings for methods and class
    def __init__(self, **kwargs):
        super().__init__(XGBRegressor(**kwargs), "XGBRegressor")

class RandomForestTrainer(TunableTrainer):
    # TODO: write docstrings for methods and class
    def __init__(self, **kwargs):
        super().__init__(RandomForestRegressor(**kwargs), "RandomForestRegressor")


