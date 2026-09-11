# NYC Taxi Fare Predictor

## Description

This project aims to predict the fare of a taxi trip in NYC based of historic taxi data. The raw data can be found at this link https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page.

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

### User Interface

![First Browser Page](Graphs/first_browser_page.png)
**Figure 5:** First simple browser page to demostrate the concept.

### Adding Tips

![New Browser Page](Graphs/browser_with_tip_amount.png)
**Figure 6:** Demonstration of the add custom tip feature.

## Change in Modelling Approach

### Area IDs

So far in the cleaning, feature enginerring and training of the models, we have only conidered the pickup and dropoff locations as one or more of the five boroughs of NYC. This made for simple encoding during the feature engineering stage but limits the usablitiy of the model for customers wanting to estimate their fare accurately.

By using the 5 boroughs, the model cannot disingusih between travelling 100 meters along one block in Manhatten or travelling the full length of Manhatten island. Furthermore, a user might not know their trip duration or distance beforehand, leabing these inputs useless.

To fix this, we created lookup tables which describe the average trip distance and duration between pick up and drop off zones in NYC. The process to create these tables is described below.

### Building the Zone Lookup Tables

At inference time a user can only give a pickup zone and a dropoff zone, not a measured trip distance or duration. Since the model was trained on real per-trip `trip_distance`/`trip_duration`, we need a stand-in for those two features that can be derived from just the zone pair. The fix is a pair of 256x256 PU (pickup) x DO (dropoff) zone matrices — one for mean trip distance, one for mean trip duration — built once from historical data and then looked up by zone ID at prediction time.

The matrices are built in `src/features.py`, restricted to the training period only (via `time_sorted_split_df()`) so that no test-period trips leak into the lookup values. The build runs as a fallback chain, since not every one of the ~65,000 possible zone pairs has been observed in the data:

1. **`create_zone_matrix(df, value_col)`** — groups trips by `PULocationID`/`DOLocationID` and pivots the mean of `value_col` (`trip_distance` or `trip_duration`) into a PU x DO matrix. Most zone pairs are filled here, but many are still missing.
2. **`transpose_fillna(matrix)`** — fills a missing A→B value using the observed B→A value, where one exists. Reasonable for something roughly symmetric like distance or duration between two fixed points.
3. **`intra_zone_fillna(matrix, value_col)`** — same-zone (PU == DO) trips have no reverse trip to borrow from, so these diagonal entries are filled with a fixed default from `INTRA_ZONE_DEFAULTS` (0.8 miles / 8.0 minutes).
4. **`borough_mean_fillna(df, matrix, lookup_ref, value_col)`** — any zone pairs still missing after steps 2–3 are filled using the mean `value_col` for that pair's boroughs (e.g. a specific Queens→Brooklyn zone pair with no data falls back to the overall Queens→Brooklyn average), using the zone→borough mapping from `taxi_zone_lookup.csv`.
5. Any final gaps are filled with the global mean across the whole matrix.

`build_zone_lookup(df, value_col)` runs this whole chain and asserts no `NaN` values remain. The two resulting matrices are saved out as `distance_lookup.csv` and `duration_lookup.csv`, and merged onto the training data as candidate features via `add_zone_features()`.

```python
from features import time_sorted_split_df, build_zone_lookup

df_train, df_test = time_sorted_split_df(df, sort_index='tpep_pickup_datetime', test_proportion=0.2)

distance_lookup = build_zone_lookup(df_train, value_col='trip_distance')
duration_lookup = build_zone_lookup(df_train, value_col='trip_duration')

distance_lookup.to_csv('data/distance_lookup.csv')
duration_lookup.to_csv('data/duration_lookup.csv')
```

### Fare Prediction

`predict_fare()` in `src/predict.py` takes the user-facing trip details (no real distance/duration required), looks up the pickup/dropoff zones' historical mean distance and duration from the two lookup tables above, encodes the request the same way the training data was encoded, and predicts the base fare using the production XGBoost model (loaded via MLflow). It then adds an itemised set of NYC taxi surcharges on top, based on TLC's published fare rules — the MTA State Surcharge, Improvement Surcharge, rush hour and overnight surcharges, the NYS Congestion Surcharge (Manhattan-related trips), and the MTA Congestion/CBD fee (an exact, data-derived list of CBD zone IDs, `CBD_ZONE_IDS`).

```python
from predict import predict_fare

predict_fare(
    passenger_count=2,
    pickup_zone='JFK Airport',
    dropoff_zone='Midtown Center',
    rate_category='standard',
    pickup_hour=14,
    pickup_dayofweek=2
)
```

This returns a dictionary with the model's predicted base fare, an itemised surcharge breakdown, and the total:

```python
{
    'base_fare': ...,
    'surcharges': {
        'mta_state_surcharge': 0.50,
        'improvement_surcharge': 1.00,
        'rush_hour_surcharge': 0.0,
        'overnight_surcharge': 0.0,
        'nys_congestion_surcharge': 2.50,
        'cbd_congestion_fee': 0.75,
        'total_surcharges': ...
    },
    'fare_with_surcharges': ...
}
```

An optional tip can then be added on top via `optional_tip(fare, tip_percentage)`, which applies a simple percentage to whichever fare figure (usually `fare_with_surcharges`) is passed in.

**Known limitation:** because the model was trained on real, noisy per-trip distance/duration but predicts from smoothed zone-pair averages, there's a train/inference distribution mismatch — real-world accuracy is expected to be somewhat lower than the training/test metrics computed on real trip data.

### New User Interface

![New Browser Page](Graphs/Browser_with_surcharges.png)
**Figure 7:** Demostration of the new surcharges applied.


## Project Structure 
```
TAXI-FARE-PREDICTOR/
├── data/                     # Raw datasets (CSV, Parquet)
│   ├── taxi_zone_lookup.csv
│   ├── distance_lookup.csv
│   ├── duration_lookup.csv
│   ├── model_training_columns.json
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
│   ├── features.py           # Feature engineering & zone lookup tables
│   ├── pipeline.py           # ML pipeline construction
│   ├── predict.py            # Inference & surcharge calculation
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