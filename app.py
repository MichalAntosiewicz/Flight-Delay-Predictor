import streamlit as st
import pandas as pd
import joblib
import numpy as np
from xgboost import XGBClassifier

st.set_page_config(page_title = "Flight Delay Prediction", layout = "wide")

@st.cache_resource
def load_model_assets():
    model = joblib.load("flight_model.pkl")
    encoders = joblib.load("encoders.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, encoders, feature_names

model, encoders, feature_names = load_model_assets()
months = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June", 
          7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}
days = {1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday", 5: "Thursday", 6: "Friday", 7: "Saturday"}

st.title("Flight Delay Prediction")
st.markdown("This app predicts whether a flight will be delayed based on various input features and historical data.")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Flight Details")
    carrier = st.selectbox("Airline Carrier", sorted(encoders['CARRIER_NAME'].classes_))
    airport = st.selectbox("Origin Airport", sorted(encoders['DEPARTING_AIRPORT'].classes_))
    time_blk = st.selectbox("Departure Time Block", sorted(encoders['DEP_TIME_BLK'].classes_))
    month = st.select_slider("Month", options=[f"{i}: {months[i]}" for i in months.keys()])
    day_of_week = st.select_slider("Day of Week", options=[f"{i}: {days[i]}" for i in days.keys()])
with col2:
    st.subheader("Weather Conditions")
    tmax = st.number_input("Max Temperature (°F)", value = 70.0, step = 1.0)
    awnd = st.slider("Average Wind Speed (mph)", 0, 50, 10)
    prcp = st.slider("Precipitation (inches)", 0.0, 5.0, 0.0, step = 0.1)
    snow = st.slider("Snowfall (inches)", 0.0, 10.0, 0.0, step = 0.1)
with col3:
    st.subheader("Statistics and Logistics")
    concurrent = st.number_input("Concurrent Flights", value = 5)
    seats = st.number_input("Available Seats", value = 150)

    st.info("Model also includes historical data for the selected carrier and airport")


if st.button("Analyze Flight Delay Risk", use_container_width=True, type="primary"):

    clean_month = int(month.split(":")[0])
    clean_day = int(day_of_week.split(":")[0])

    
    input_data = pd.DataFrame([{
        'MONTH': clean_month,
        'DAY_OF_WEEK': clean_day,
        'CONCURRENT_FLIGHTS': concurrent,
        'NUMBER_OF_SEATS': seats,
        'PRCP': prcp,
        'SNOW': snow,
        'TMAX': tmax,
        'AWND': awnd,
        'CARRIER_HISTORICAL': 0.18, # Średnie wartości
        'DEP_AIRPORT_HIST': 0.18,
        'DAY_HISTORICAL': 0.18,
        'DEP_BLOCK_HIST': 0.18,
        'CARRIER_NAME': carrier,
        'DEPARTING_AIRPORT': airport,
        'DEP_TIME_BLK': time_blk
    }])

    for col in ['CARRIER_NAME', 'DEPARTING_AIRPORT', 'DEP_TIME_BLK']:
        input_data[col] = encoders[col].transform(input_data[col].astype(str))

    input_data = input_data[feature_names]

    prob = model.predict_proba(input_data)[0][1]

    st.divider()

    m1, m2 = st.columns(2)
    m1.metric("Probability of Delay", f"{prob*100:.1f}%")

    threshold = 0.5

    status = "Delayed" if prob >= threshold else "On Time"

    if prob > threshold:
        st.error(f"High risk of delay: {status}")
    else:
        st.success(f"Low risk of delay: {status}")
