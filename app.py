import streamlit as st
import joblib
import numpy as np

import pandas as pd

# Load Model and Scaler
model = joblib.load('water_potability_model.pkl')
scaler = joblib.load('scaler.pkl')

# Page Title
st.title("💧 Water Potability Prediction System")
st.write(
    "Predict whether water is safe for consumption based on water quality parameters."
)

# Sample Data
potable_sample = {
    "ph": 9.445130,
    "Hardness": 145.805402,
    "Solids": 13168.529156,
    "Chloramines": 9.444471,
    "Sulfate": 310.583374,
    "Conductivity": 592.659021,
    "Organic_carbon": 8.606397,
    "Trihalomethanes": 77.577460,
    "Turbidity": 3.875165
}

non_potable_sample = {
    "ph": 5.1,
    "Hardness": 320.0,
    "Solids": 45000.0,
    "Chloramines": 12.0,
    "Sulfate": 500.0,
    "Conductivity": 700.0,
    "Organic_carbon": 25.0,
    "Trihalomethanes": 120.0,
    "Turbidity": 8.0
}

# Default Values
defaults = {
    "ph": 7.0,
    "Hardness": 200.0,
    "Solids": 20000.0,
    "Chloramines": 7.0,
    "Sulfate": 330.0,
    "Conductivity": 420.0,
    "Organic_carbon": 14.0,
    "Trihalomethanes": 70.0,
    "Turbidity": 4.0
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Demo Buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("Load Potable Sample"):
        for key, value in potable_sample.items():
            st.session_state[key] = value

with col2:
    if st.button("Load Non-Potable Sample"):
        for key, value in non_potable_sample.items():
            st.session_state[key] = value

st.subheader("📝 Enter Water Quality Parameters")

st.divider()

# Inputs
ph = st.number_input("pH", key="ph")
hardness = st.number_input("Hardness", key="Hardness")
solids = st.number_input("Solids", key="Solids")
chloramines = st.number_input("Chloramines", key="Chloramines")
sulfate = st.number_input("Sulfate", key="Sulfate")
conductivity = st.number_input("Conductivity", key="Conductivity")
organic_carbon = st.number_input("Organic Carbon", key="Organic_carbon")
trihalomethanes = st.number_input(
    "Trihalomethanes",
    key="Trihalomethanes"
)
turbidity = st.number_input(
    "Turbidity",
    key="Turbidity"
)

# Prediction
if st.button("Predict Water Potability"):


    data = pd.DataFrame({
        "ph": [ph],
        "Hardness": [hardness],
        "Solids": [solids],
        "Chloramines": [chloramines],
        "Sulfate": [sulfate],
        "Conductivity": [conductivity],
        "Organic_carbon": [organic_carbon],
        "Trihalomethanes": [trihalomethanes],
        "Turbidity": [turbidity]
    })

    st.subheader("Input Summary")

    st.dataframe(data)

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

    probability = model.predict_proba(data_scaled)

    confidence = max(probability[0])

    st.subheader("📊 Prediction Result")

    if prediction[0] == 1:
        st.success("✅ Water is Potable")
    else:
        st.error("❌ Water is Not Potable")

    st.progress(float(confidence))

    st.write(f"Confidence Score: {confidence*100:.2f}%")



st.markdown("---")

st.markdown("""
### Prediction Labels

- ✅ **Potable** → Water is considered safe for drinking.
- ❌ **Non-Potable** → Water is not considered safe for drinking.
""")


st.markdown("---")

st.markdown(
    "<center><small>Developed by Umar | Machine Learning & AI Diploma | BIA Institute</small></center>",
    unsafe_allow_html=True
)