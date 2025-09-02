# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# =====================
# 1. Load trained pipeline
# =====================
@st.cache_resource
def load_model():
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "house_price_pipeline.pkl")
    return joblib.load(MODEL_PATH)

model = load_model()

# =====================
# 2. App Title
# =====================
st.set_page_config(page_title="🏠 House Price Prediction", layout="centered")
st.title("🏠 House Price Prediction App")
st.write("Enter the details of the house and get an estimated price.")

# =====================
# 3. User Inputs
# =====================
st.sidebar.header("House Features")

# Numeric features
area = st.sidebar.number_input("Area (sq ft)", min_value=500, max_value=10000, value=2000, step=50)
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 5, 2)
stories = st.sidebar.slider("Stories", 1, 5, 2)
parking = st.sidebar.slider("Parking spaces", 0, 5, 1)

# Categorical features
mainroad = st.sidebar.selectbox("Main Road Access", ["yes", "no"])
guestroom = st.sidebar.selectbox("Guest Room", ["yes", "no"])
basement = st.sidebar.selectbox("Basement", ["yes", "no"])
hotwaterheating = st.sidebar.selectbox("Hot Water Heating", ["yes", "no"])
airconditioning = st.sidebar.selectbox("Air Conditioning", ["yes", "no"])
prefarea = st.sidebar.selectbox("Preferred Area", ["yes", "no"])
furnishingstatus = st.sidebar.selectbox("Furnishing Status", ["furnished", "semi-furnished", "unfurnished"])

# =====================
# 4. Prediction
# =====================
input_data = pd.DataFrame([{
    "area": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "stories": stories,
    "parking": parking,
    "mainroad": mainroad,
    "guestroom": guestroom,
    "basement": basement,
    "hotwaterheating": hotwaterheating,
    "airconditioning": airconditioning,
    "prefarea": prefarea,
    "furnishingstatus": furnishingstatus
}])

if st.sidebar.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Price: **₹{prediction:,.0f}**")
