import streamlit as st
import pandas as pd
import numpy as np
import joblib
from src.predict import FlightDelayPredictor
import os


# Page configuration
st.set_page_config(
    page_title="AeroDelay",
    page_icon="✈️",
    layout="wide"
)

# Title and description
st.markdown("<h1 style='color:#0844a1;'>✈️ Welcome to AeroDelay</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='color:#555555;'>Predict flight delays in advance so you can plan your travel smarter and avoid unexpected waiting times.</h5>", unsafe_allow_html=True)

# Separation line
st.markdown("<hr style='border:0; border-top:1px solid #e0e0e0; margin:10px 0;'>", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    model_path = "models/flight_delay_model.pkl"
    if os.path.exists(model_path):
        return FlightDelayPredictor(model_path)
    return None

predictor = load_model()

if predictor is None:
    st.error("Model not found.")
    st.stop()

st.subheader("Enter Flight Details")

col1, col2 = st.columns(2)

AIRLINES = {
    "American Airlines": 0,
    "Delta": 1,
    "Southwest": 2,
    "United": 3
}

AIRCRAFT_TYPES = {
    "Airbus A320": 0,
    "Boeing 737": 1,
    "Boeing 777": 2
}

ORIGIN_AIRPORTS = {
    "ATL - Hartsfield-Jackson Atlanta": "ATL",
    "DFW - Dallas/Fort Worth": "DFW",
    "JFK - John F. Kennedy New York": "JFK",
    "LAX - Los Angeles": "LAX",
    "ORD - Chicago O'Hare": "ORD"
}
ORIGIN_CODES = {"ATL": 0, "DFW": 1, "JFK": 2, "LAX": 3, "ORD": 4}

DESTINATION_AIRPORTS = {
    "BOS - Boston Logan": "BOS",
    "JFK - John F. Kennedy New York": "JFK",
    "MIA - Miami": "MIA",
    "SEA - Seattle-Tacoma": "SEA",
    "SFO - San Francisco": "SFO"
}
DESTINATION_CODES = {"BOS": 0, "JFK": 1, "MIA": 2, "SEA": 3, "SFO": 4}

with col1:
    st.markdown("**Flight Information**")
    distance = st.number_input("Distance (miles)", min_value=0, max_value=5000, value=500)
    scheduled_duration = st.number_input("Scheduled Duration (minutes)", min_value=0, max_value=1000, value=120)
    
    airline = st.selectbox("Airline", list(AIRLINES.keys()))
    aircraft_type = st.selectbox("Aircraft Type", list(AIRCRAFT_TYPES.keys()))

with col2:
    st.markdown("**Schedule Information**")
    departure_hour = st.slider("Departure Hour", 0, 23, 12)
    departure_day = st.selectbox("Day of Week", 
                                 ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    departure_month = st.slider("Month", 1, 12, 6)
    
    origin = st.selectbox("Origin Airport", list(ORIGIN_AIRPORTS.keys()))
    destination = st.selectbox("Destination Airport", list(DESTINATION_AIRPORTS.keys()))

if st.button("🔮 Predict Delay", type="primary"):
    try:
        day_index = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(departure_day)
        
        is_weekend = 1 if day_index >= 5 else 0
        
        origin_code = ORIGIN_AIRPORTS[origin]
        destination_code = DESTINATION_AIRPORTS[destination]
        
        # Create feature dictionary
        flight_data = {
            'Distance': distance,
            'ScheduledDuration': scheduled_duration,
            'DepartureHour': departure_hour,
            'DepartureDay': day_index,
            'DepartureMonth': departure_month,
            'IsWeekend': is_weekend,
            'Airline_Encoded': AIRLINES[airline],
            'Origin_Encoded': ORIGIN_CODES[origin_code],
            'Destination_Encoded': DESTINATION_CODES[destination_code],
            'AircraftType_Encoded': AIRCRAFT_TYPES[aircraft_type],
        }
        
        delay_minutes = predictor.predict_single(flight_data)
        category = predictor._categorize_delay(delay_minutes)
        
        st.success("✅ Prediction Complete!")
        
        result_col1, result_col2 = st.columns(2)
        with result_col1:
            st.metric("Predicted Delay", f"{delay_minutes:.1f} minutes", 
                     delta=f"{delay_minutes - 15:.1f} min from avg" if delay_minutes > 15 else "On time")
        with result_col2:
            if category == "On Time":
                st.success(f"Status: **{category}**")
            elif category == "Minor Delay":
                st.info(f"Status: **{category}**")
            elif category == "Moderate Delay":
                st.warning(f"Status: **{category}**")
            else:
                st.error(f"Status: **{category}**")
                
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.info("Make sure your input features match the model's expected features.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p style="text-align:center; color:#555555;">&copy; 2025 AeroDelay. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)
