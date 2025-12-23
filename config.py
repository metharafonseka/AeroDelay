#Configuration file for AeroDelay project.

#This module contains all configuration parameters for data preprocessing,
#model training, and prediction.

import os

# PROJECT PATHS
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Data file paths
RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, "flight_delays.csv")
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "flight_delays_preprocessed.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "flight_delay_model.pkl")


# DATA PREPROCESSING CONFIG

# Columns to drop
COLUMNS_TO_DROP = [
    'FlightID',
    'FlightNumber',
    'TailNumber',
    'DelayReason',
    'ActualDeparture',
    'ActualArrival',
    'Cancelled',
    'Diverted'
]

# Datetime columns
DATETIME_COLUMNS = [
    'ScheduledDeparture',
    'ScheduledArrival'
]

# Categorical columns for encoding
CATEGORICAL_COLUMNS = [
    'Airline',
    'Origin',
    'Destination',
    'AircraftType'
]

# Numeric columns for scaling
NUMERIC_COLUMNS_TO_SCALE = [
    'Distance',
    'ScheduledDuration',
    'DepartureHour'
]

# Target variable
TARGET_COLUMN = 'DelayMinutes'

# MODEL TRAINING CONFIG

# Train-test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Random Forest baseline parameters
RF_BASELINE_PARAMS = {
    'n_estimators': 200,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# Hyperparameter tuning grid
PARAM_GRID = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# GridSearchCV parameters
CV_FOLDS = 3
SCORING_METRIC = 'neg_mean_absolute_error'


# PREDICTION CONFIG

# Delay categories (in minutes)
DELAY_CATEGORIES = {
    'on_time': 15,
    'minor': 30,
    'moderate': 60
}

# Feature names (after preprocessing)
EXPECTED_FEATURES = [
    'Distance',
    'ScheduledDuration',
    'DepartureHour',
    'DepartureDay',
    'DepartureMonth',
    'IsWeekend',
    'Airline_Encoded',
    'Origin_Encoded',
    'Destination_Encoded',
    'AircraftType_Encoded'
]


# VISUALIZATION CONFIG
# Plot settings
PLOT_STYLE = 'seaborn-v0_8'
FIGURE_SIZE = (10, 6)
DPI = 100
# Top N features to display
TOP_N_FEATURES = 10


# LOGGING CONFIG
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


# WEB APP CONFIG
# Streamlit app settings
APP_TITLE = "Flight Delay Predictor"
APP_ICON = "✈️"
APP_LAYOUT = "wide"
# Default values for input fields
DEFAULT_VALUES = {
    'distance': 500,
    'scheduled_duration': 120,
    'departure_hour': 12,
    'departure_month': 6
}
