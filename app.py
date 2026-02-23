import streamlit as st
import numpy as np
import joblib
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# -----------------------------
# Load Model (Dynamic Path)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model (2).pkl")

try:
    model = joblib.load(model_path)
except Exception as e:
    st.error("❌ Model file not found. Make sure 'model.pkl' is in the same folder.")
    st.stop()

# -----------------------------
# App Title
# -----------------------------
st.title("💳 Credit Card Fraud Detection System")
st.markdown("Enter transaction details below to check if the transaction is **Fraudulent** or **Legitimate**.")

st.divider()

# -----------------------------
# Input Section
# -----------------------------
st.subheader("📥 Transaction Details")

col1, col2 = st.columns(2)

features = []

with col1:
    time = st.number_input("Time (seconds)", min_value=0.0, step=1.0)
    features.append(time)

    for i in range(1, 15):
        value = st.number_input(f"V{i}", format="%.5f")
        features.append(value)

with col2:
    for i in range(15, 29):
        value = st.number_input(f"V{i}", format="%.5f")
        features.append(value)

    amount = st.number_input("Amount", min_value=0.0, step=0.01)
    features.append(amount)

features = np.array([features])

st.divider()

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🔍 Predict Transaction"):

    try:
        prediction = model.predict(features)[0]

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(features)[0][1]
        else:
            probability = None

        st.subheader("📊 Prediction Result")

        if prediction == 1:
            if probability is not None:
                st.error(f"⚠️ Fraudulent Transaction Detected!\n\nRisk Score: {probability:.2%}")
            else:
                st.error("⚠️ Fraudulent Transaction Detected!")
        else:
            if probability is not None:
                st.success(f"✅ Legitimate Transaction\n\nFraud Probability: {probability:.2%}")
            else:
                st.success("✅ Legitimate Transaction")

    except Exception as e:
        st.warning("⚠️ Input feature mismatch. Ensure model was trained with 30 features.")
        st.text(str(e))

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption("⚠️ This application is for educational purposes only.")