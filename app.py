import streamlit as st
import pandas as pd
import joblib

# Set page layout configuration
st.set_page_config(page_title="Churn Predictor", page_icon="🏦", layout="centered")

# Load saved artifacts safely
@st.cache_resource # Cache the model so it doesn't reload on every button click
def load_assets():
    model = joblib.load('customer_churn_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
    return model, model_columns

try:
    model, model_columns = load_assets()
except FileNotFoundError:
    st.error("⚠️ Model files not found! Please run `python train.py` first to generate them.")
    st.stop()

# Title and App Header Description
st.title("🏦 Customer Churn Risk Predictor")
st.markdown("""
This AI model calculates the probability of a subscription customer canceling their service based on operational metrics.
""")

st.write("---")

# Layout columns for inputs
col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Customer Tenure (Months)", min_value=1, max_value=72, value=12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=15.0, max_value=200.0, value=65.0)

with col2:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

# Prediction Logic triggered by button
if st.button("Analyze Churn Risk", type="primary"):
    # 1. Capture user inputs into a structured dataframe
    input_data = pd.DataFrame([{
        'Tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'Contract': contract
    }])
    
    # 2. Match the exact One-Hot Encoding format used during training
    input_encoded = pd.get_dummies(input_data, columns=['Contract'], drop_first=True)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
    
    # 3. Calculate probabilities
    prediction_proba = model.predict_proba(input_encoded)[0][1] # Probability of class 1 (Churn)
    prediction = model.predict(input_encoded)[0]
    
    st.write("---")
    # Display beautiful UI results based on thresholds
    if prediction == 1:
        st.error(f"🚨 **High Risk of Churn!** The model estimates a **{prediction_proba * 100:.1f}%** chance this customer will leave.")
    else:
        st.success(f"✅ **Low Risk.** The model estimates only a **{prediction_proba * 100:.1f}%** chance of churn. Customer appears stable.")