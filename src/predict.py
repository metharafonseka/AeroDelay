import pandas as pd
import numpy as np
import joblib
import os
from typing import Union, List, Dict


class FlightDelayPredictor:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = joblib.load(model_path)
        self.model_path = model_path
        print(f"Model loaded from: {model_path}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        
        predictions = self.model.predict(X)
        return predictions
    
    def predict_single(self, flight_data: Dict) -> float:
        # Ensure correct feature order matching training data
        feature_order = [
            'Distance', 'ScheduledDuration', 'DepartureHour', 
            'DepartureDay', 'DepartureMonth', 'IsWeekend',
            'Airline_Encoded', 'Origin_Encoded', 
            'Destination_Encoded', 'AircraftType_Encoded'
        ]
        
        # Create ordered dictionary
        ordered_data = {key: flight_data[key] for key in feature_order}
        df = pd.DataFrame([ordered_data])
        
        prediction = self.predict(df)[0]
        return prediction
    
    @staticmethod
    def _categorize_delay(minutes: float) -> str:
        if minutes < 15:
            return "On Time"
        elif minutes < 30:
            return "Minor Delay"
        elif minutes < 60:
            return "Moderate Delay"
        else:
            return "Major Delay"
    
    def get_model_info(self) -> Dict:
        info = {
            'model_type': type(self.model).__name__,
            'model_path': self.model_path,
        }
        
        # Add Random Forest specific info
        if hasattr(self.model, 'n_estimators'):
            info['n_estimators'] = self.model.n_estimators
        if hasattr(self.model, 'max_depth'):
            info['max_depth'] = self.model.max_depth
        if hasattr(self.model, 'n_features_in_'):
            info['n_features'] = self.model.n_features_in_
            
        return info


def load_model(model_path: str) -> FlightDelayPredictor:
    return FlightDelayPredictor(model_path)


if __name__ == "__main__":
    MODEL_PATH = "../models/flight_delay_model.pkl"
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print("Error: Model not found. Please train the model first using train.py")
        exit(1)
    
    # Load predictor
    predictor = load_model(MODEL_PATH)
    
    # Display model info
    print("\nModel loaded successfully!")
    print("\nModel Information:")
    for key, value in predictor.get_model_info().items():
        print(f"  {key}: {value}")