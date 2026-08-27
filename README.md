# NYC Taxi Fare Predictor

## Description

This project aims to predict the fare of a taxi trip in NYC based of historic taxi data.

## Work done so far

### EDA and Feature Engineering (`notebooks/eda.ipynb`)

I have conducted explorotory data analysis on the taxi trip dataset, analysing distributions of features such as fare amount, trip distance and duration of trips. This EDA is contained and documented in `notebooks/eda.ipynb`.

A data loading and cleaning file (`src/cleaning.py`) contains a function `load_and_clean()` which takes the file paths for trip data and taxi zone codes and creates an SQL query that filters data based on the criteria which was decided in the EDA.

I have performed feature engineering (`src/features.py`) which includes calculating the duration of trips, the hour and day of pickups, as well as flagging airport trips to JFK and Newark which are flat rates. Columns that need to be scaled will also be treated accordingly.

### Initial Training (`notebooks/model_comparison.ipynb`)

Training classes and methods have been created which allows for an initial training run for any model as well as a training run for a randomised cross validation search to tune the hyperparamters of the model (`src/train_and_tune.py`).  The traning and model comparison is conducted and documented in `notebooks/model_comparison.ipynb`.

Initial training runs of linear regression, XGBoost regressor and Random Forest regressor models have been conducted and their root mean squared error (RMSE) and R2 score have been computed in order to compare the performance of the models. The following graphs compare the models.

![R2 Comparison](Graphs/Initial_R2_by_Model.png)
**Figure 1:** R2 comparison across models.

![RMSE Comparison](Graphs/Initial_RMSE_by_Model.png)
**Figure 2:** RMSE comparison across models.

Clearly, XGBoost is the best performing model out of the three with the highest R2 score and lowest RMSE. We will see if this trend carries over after hyperparameter tuning.

### Hyperparameter Tuning

Using `RandomisedCVSearch`, the hyperparameters for the XGBoost and Random Forest models were tuned by randomly searching over a grid of parameters to find the best model. The following graphs show the best performing XGBoost and Random Forest models during tuning.

![R2 Comparison](Graphs/XGB_RF_R2.png)
**Figure 3:** R2 comparison across models.

![RMSE Comparison](Graphs/XGB_RF_RMSE.png)
**Figure 4:** RMSE comparison across models.

We will chose the XGBoost regression model. Despite having similar training RMSE scores, the graph indicates that the XGBoost has significantly lower test RMSE which suggests it generalises better to unseen data that the RandomForest regressor.

### Fare Prediction

A function (`predict_fare()` in `src/predict.py`) was built to take in the users parameters and make a prediction of the base fare for the journey. The production model which was chosen is loaded using MlFlow and then the prediction is made. Here is an example:

```python
predict_fare(trip_duration=34.0,
    passenger_count=2,
    trip_distance=8.3,
    pickup_borough='Manhattan',
    dropoff_borough='Manhattan',
    rate_category='standard',
    pickup_hour=11,
    pickup_dayofweek=4)
```

### User Interface

![First Browser Page](Graphs/first_browser_page.png)
**Figure 5:** First simple browser page to demostrate the concept.

## Project Structure 
```
TAXI-FARE-PREDICTOR/
├── data/                     # Raw datasets (CSV, Parquet)
│   ├── taxi_zone_lookup.csv
│   └── yellow_tripdata_2026-01.parquet
│
├── Graphs/                   # Saved model performance plots
│   ├── Initial_R2_by_Model.png
│   └── Initial_RMSE_by_Model.png
|   |__ XGB_RF_R2.png
|   |__ XGB_RF_RMSE.png
│
├── notebooks/                # Jupyter notebooks for EDA & model comparison
│   ├── eda.ipynb
│   └── model_comparison.ipynb
│
├── src/                      # Source code for data processing & training
│   ├── cleaning.py           # Data cleaning utilities
│   ├── features.py           # Feature engineering
│   ├── pipeline.py           # ML pipeline construction
│   └── train_and_tune.py     # Model training & hyperparameter tuning
│
├── tests/                    # Unit tests for project modules
│   ├── test_cleaning.py
│   ├── test_features.py
│   └── test_train.py
│
├── mlflow.db                 # Local MLflow tracking database
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
└── .gitignore                # Git ignore rules
```

## Future work

The next step is to tune the hyperparamters and settle in a final model to build the production pipeline around. The experiments will be tracked with MLflow and the aim is to produce a full API/pipeline where users can input their loaction, destination etc and get an accurate quote of the fare. 

Tests and documentation will also be provided throughout.