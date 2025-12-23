# ✈️ AeroDelay - Flight Delay Prediction System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A comprehensive machine learning system for predicting flight delays using Random Forest Regression. This project demonstrates end-to-end ML workflow including data preprocessing, feature engineering, model training, hyperparameter tuning, and deployment through a web interface.

## 🎯 Project Overview

Flight delays cost airlines and passengers billions annually. This project builds a predictive model to forecast flight delays based on historical data, enabling proactive decision-making for airlines, airports, and travelers.

**Key Features:**
- 🔄 Robust data preprocessing pipeline
- 🎛️ Feature engineering from temporal and categorical data
- 🌲 Random Forest model with hyperparameter tuning
- 📊 Comprehensive model evaluation and visualization
- 🚀 Interactive web application using Streamlit

## 📁 Project Structure

```
AeroDelay/
├── data/
│   ├── raw/                    # Original flight data
│   └── processed/              # Preprocessed data ready for modeling
├── models/                     # Saved trained models
├── notebooks/
│   ├── preprocessing.ipynb           # Data exploration and preprocessing
│   └── model_training_and_evaluation.ipynb  # Model training and evaluation
├── src/
│   ├── preprocess.py          # Data preprocessing functions
│   ├── train.py               # Model training pipeline
│   └── predict.py             # Prediction module with FlightDelayPredictor class
├── app.py                     # Streamlit web application
├── requirements.txt           # Project dependencies
└── README.md
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd AeroDelay
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Data Preprocessing

Preprocess raw flight data using the preprocessing module:

```bash
python src/preprocess.py
```

Or explore the preprocessing notebook:
```bash
jupyter notebook notebooks/preprocessing.ipynb
```

**Preprocessing Steps:**
- Remove unnecessary columns (IDs, data leakage features)
- Handle missing values and duplicates
- Feature engineering (flight duration, temporal features, weekend indicator)
- Encode categorical variables (Airline, Origin, Destination, Aircraft Type)
- Scale numeric features using StandardScaler

### 2. Model Training

Train the Random Forest model with hyperparameter tuning:

```bash
python src/train.py
```

Or use the training notebook:
```bash
jupyter notebook notebooks/model_training_and_evaluation.ipynb
```

**Training Process:**
- Train/test split (80/20)
- Baseline Random Forest model
- GridSearchCV for hyperparameter optimization
- Model evaluation with multiple metrics
- Feature importance analysis
- Model persistence using joblib

### 3. Make Predictions

Use the trained model to predict flight delays:

```python
from src.predict import FlightDelayPredictor

# Load model
predictor = FlightDelayPredictor("models/flight_delay_model.pkl")

# Make predictions
predictions = predictor.predict(X_test)

```

Or run the example:
```bash
python src/predict.py
```

### 4. Web Application

Launch the interactive Streamlit app:

```bash
streamlit run app.py
```

### 📊 Top Important Features

1. **ScheduledDuration** - Flight duration in minutes
2. **Distance** - Flight distance
3. **DepartureHour** - Hour of departure
4. **Airline_Encoded** - Airline carrier
5. **Origin_Encoded** - Origin airport

## 🔬 Technical Details

### Machine Learning Pipeline

1. **Data Preprocessing**
   - Label encoding for categorical features
   - Standard scaling for numerical features
   - Time-based feature extraction

2. **Model Architecture**
   - Algorithm: Random Forest Regressor
   - Hyperparameters: Optimized via GridSearchCV
   - Cross-validation: 3-fold CV

3. **Evaluation Metrics**
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
   - Residual analysis

### Delay Categories

The system categorizes predictions into severity levels:
- 🟢 **On Time**: < 15 minutes
- 🟡 **Minor Delay**: 15-30 minutes
- 🟠 **Moderate Delay**: 30-60 minutes
- 🔴 **Major Delay**: > 60 minutes

## 📈 Visualizations

The notebooks include comprehensive visualizations:
- Distribution of delay minutes
- Average delay by airline
- Feature importance bar charts
- Residual plots
- Predicted vs Actual scatter plots

## 🧪 Testing the System

Run a quick test to ensure everything works:

```bash
# Test preprocessing
python -c "from src.preprocess import preprocess_data; print('✓ Preprocessing module OK')"

# Test prediction
python src/predict.py
```

## 👤 Author

**Methara Fonseka**
- GitHub: [@metharafonseka](https://github.com/metharafonseka)
- LinkedIn: [Methara Fonseka](https://linkedin.com/in/methara-fonseka)
