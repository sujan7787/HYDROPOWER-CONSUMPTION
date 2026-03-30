import streamlit as st
import pandas as pd
import pickle

st.title(" Energy Prediction App")

# Load trained model
@st.cache_resource
def load_model():
    with open("prophet_model_with_regressors.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()


st.subheader("Enter Input Features")

global_intensity = st.slider(
    "Global Intensity (1.0 - 22.4)",
    min_value=1.0,
    max_value=22.4,
    value=6.8,
    step=0.1
)

sub_metering_3 = st.slider(
    "Sub Metering 3 (0.0 - 30.0)",
    min_value=0.0,
    max_value=30.0,
    value=19.0,
    step=0.1
)

voltage = st.slider(
    "Voltage (235.65 - 246.65)",
    min_value=235.65,
    max_value=246.65,
    value=244.0,
    step=0.01
)

hour = st.slider(
    "Hour (0-23)",
    min_value=0,
    max_value=23,
    value=14
)

day_of_week = st.slider(
    "Day of Week (0=Mon, 6=Sun)",
    min_value=0,
    max_value=6,
    value=3
)

month = 10  # fixed, because  training data is only October

if st.button(" Predict"):

    # Use a ds close to training dates to avoid extreme extrapolation cause i got unexpected error when i didnot used that...
    future = pd.DataFrame({
        'ds': pd.to_datetime(['2010-10-03 14:00:00']),  
        'Global_intensity': [global_intensity],
        'Sub_metering_3': [sub_metering_3],
        'Voltage': [voltage],
        'hour': [hour],
        'day_of_week': [day_of_week],
        'month': [month]
    })

    # Make prediction
    forecast = model.predict(future)

    # Extract predicted power and clip negatives to 0
    predicted_power = max(forecast['yhat'].values[0], 0)
    yhat_lower = max(forecast['yhat_lower'].values[0], 0)
    yhat_upper = max(forecast['yhat_upper'].values[0], 0)

   
    st.subheader("⚡ Predicted Global Active Power")
    st.success(f"{predicted_power:.3f} kW")

    st.write(f"Lower Bound: {yhat_lower:.3f} kW")
    st.write(f"Upper Bound: {yhat_upper:.3f} kW")
