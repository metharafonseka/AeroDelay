import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def load_preprocessed_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Preprocessed data not found: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"Loaded preprocessed data shape: {df.shape}")
    return df


def prepare_train_test_split(
    df: pd.DataFrame, 
    target_col: str = 'DelayMinutes',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test


def train_baseline_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 200,
    random_state: int = 42
) -> RandomForestRegressor:
    print(f"\nTraining baseline Random Forest with {n_estimators} estimators...")
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("Baseline model training complete")
    
    return model


def evaluate_model(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model"
) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"\n{model_name} Performance:")
    print(f"  MAE  : {mae:.2f} minutes")
    print(f"  MSE  : {mse:.2f}")
    print(f"  RMSE : {rmse:.2f} minutes")
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
    }


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Dict = None,
    cv: int = 3,
    random_state: int = 42
) -> GridSearchCV:
    if param_grid is None:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }
    
    print(f"\nStarting hyperparameter tuning with {cv}-fold CV...")
    print(f"Parameter grid: {param_grid}")
    
    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=random_state),
        param_grid=param_grid,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best CV Score (MAE): {-grid_search.best_score_:.2f}")
    
    return grid_search


def get_feature_importance(
    model: RandomForestRegressor,
    feature_names: list,
    top_n: int = 10
) -> pd.Series:
    feature_importances = pd.Series(
        model.feature_importances_, 
        index=feature_names
    ).sort_values(ascending=False)
    
    print(f"\nTop {top_n} Important Features:")
    for i, (feature, importance) in enumerate(feature_importances.head(top_n).items(), 1):
        print(f"  {i}. {feature:25s}: {importance:.4f}")
    
    return feature_importances


def save_model(model: RandomForestRegressor, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    print(f"\nModel saved to: {output_path}")


def train_and_evaluate_model(
    data_path: str,
    model_output_path: str,
    tune_hyperparams: bool = True
) -> RandomForestRegressor:
    print("="*60)
    print("FLIGHT DELAY MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Load data
    df = load_preprocessed_data(data_path)
    
    # Prepare train-test split
    X_train, X_test, y_train, y_test = prepare_train_test_split(df)
    
    # Train baseline model
    baseline_model = train_baseline_model(X_train, y_train)
    evaluate_model(baseline_model, X_test, y_test, "Baseline Model")
    
    # Hyperparameter tuning
    if tune_hyperparams:
        grid_search = tune_hyperparameters(X_train, y_train)
        best_model = grid_search.best_estimator_
        evaluate_model(best_model, X_test, y_test, "Tuned Model")
    else:
        best_model = baseline_model
    
    # Feature importance
    get_feature_importance(best_model, X_train.columns)
    
    # Save model
    save_model(best_model, model_output_path)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    return best_model


if __name__ == "__main__":
    # Configuration
    DATA_PATH = "../data/processed/flight_delays_preprocessed.csv"
    MODEL_PATH = "../models/flight_delay_model.pkl"
    
    # Train model
    model = train_and_evaluate_model(
        data_path=DATA_PATH,
        model_output_path=MODEL_PATH,
        tune_hyperparams=True
    )
