import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple
import os


def load_raw_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"Loaded data shape: {df.shape}")
    return df


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [
        'FlightID',      
        'FlightNumber',  
        'TailNumber',    
        'DelayReason',
        'ActualDeparture',  
        'ActualArrival',    
        'Cancelled',        
        'Diverted'          
    ]
    
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=columns_to_drop)
    
    print(f"Dropped {len(columns_to_drop)} columns")
    return df


def convert_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    datetime_cols = ["ScheduledDeparture", "ScheduledArrival"]
    
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Fix negative delays (early arrivals - set to 0)
    if 'DelayMinutes' in df.columns:
        df["DelayMinutes"] = df["DelayMinutes"].apply(lambda x: max(x, 0))
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    print(f"Data cleaned. Shape: {df.shape}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Scheduled flight duration in minutes
    if 'ScheduledArrival' in df.columns and 'ScheduledDeparture' in df.columns:
        df['ScheduledDuration'] = (
            df['ScheduledArrival'] - df['ScheduledDeparture']
        ).dt.total_seconds() / 60
    
    # Extract datetime features from ScheduledDeparture
    if 'ScheduledDeparture' in df.columns:
        df['DepartureHour'] = df['ScheduledDeparture'].dt.hour
        df['DepartureDay'] = df['ScheduledDeparture'].dt.dayofweek 
        df['DepartureMonth'] = df['ScheduledDeparture'].dt.month
        df['IsWeekend'] = (df['ScheduledDeparture'].dt.dayofweek >= 5).astype(int)
    
    # Drop datetime columns (features extracted)
    datetime_cols = ['ScheduledDeparture', 'ScheduledArrival']
    df = df.drop(columns=[col for col in datetime_cols if col in df.columns])
    
    print(f"Features engineered. Shape: {df.shape}")
    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    label_encode_cols = ['Airline', 'Origin', 'Destination', 'AircraftType']
    
    le = LabelEncoder()
    for col in label_encode_cols:
        if col in df.columns:
            df[f'{col}_Encoded'] = le.fit_transform(df[col])
    
    # Drop original categorical columns
    df = df.drop(columns=[col for col in label_encode_cols if col in df.columns])
    
    print(f"Categorical features encoded")
    return df


def scale_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = ['Distance', 'ScheduledDuration', 'DepartureHour']
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    if numeric_cols:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        print(f"Scaled {len(numeric_cols)} numeric features")
    
    return df


def preprocess_data(input_path: str, output_path: str = None) -> pd.DataFrame:
    print("Starting preprocessing pipeline...")
    
    # Load data
    df = load_raw_data(input_path)
    
    # Drop unnecessary columns
    df = drop_unnecessary_columns(df)
    
    # Convert datetime columns
    df = convert_datetime_columns(df)
    
    # Clean data
    df = clean_data(df)
    
    # Engineer features
    df = engineer_features(df)
    
    # Encode categorical features
    df = encode_categorical_features(df)
    
    # Scale numeric features
    df = scale_numeric_features(df)
    
    print(f"\nPreprocessing complete!")
    print(f"Final shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Preprocessed data saved to: {output_path}")
    
    return df


if __name__ == "__main__":
    # Example usage
    input_file = "../data/raw/flight_delays.csv"
    output_file = "../data/processed/flight_delays_preprocessed.csv"
    
    df = preprocess_data(input_file, output_file)
    print("\nFirst few rows:")
    print(df.head())
