import streamlit as st
import pandas as pd
import numpy as np
import joblib
import socket
import struct
import re
from tensorflow.keras.models import load_model

# Load global models
encoder = load_model("saved_models/encoder.h5")
isf = joblib.load("saved_models/isolation_forest.pkl")
xgb = joblib.load("saved_models/xgboost_model.pkl")
scaler = joblib.load("saved_models/minmax_scaler.pkl")

# Streamlit UI
st.set_page_config(page_title="Federated Anomaly Detection", layout="wide")
st.title("Federated IIoT Anomaly Detection Dashboard")
st.markdown("Upload a **client-side CSV** to predict anomalies using globally trained models.")

# Utilities
def is_valid_ipv4(ip):
    pattern = re.compile(r'^(\d{1,3}\.){3}\d{1,3}$')
    return pattern.match(ip) is not None

def ip_to_int(ip):
    return struct.unpack("!I", socket.inet_aton(ip))[0]

def preprocess_dataframe(df):
    df = df[df['SrcAddr'].apply(is_valid_ipv4)]
    df = df[df['DstAddr'].apply(is_valid_ipv4)]
    df['SrcAddr'] = df['SrcAddr'].apply(ip_to_int)
    df['DstAddr'] = df['DstAddr'].apply(ip_to_int)

    if df["Proto"].dtype == object:
        df["Proto"] = df["Proto"].astype("category").cat.codes

    feature_cols = ["SrcAddr", "DstAddr", "Dur", "TotPkts", "TotBytes", "SrcJitter", "DstJitter", "Proto"]
    features = df[feature_cols]
    features_scaled = scaler.transform(features)

    return df, features_scaled

def predict_anomalies(features_scaled):
    encoded = encoder.predict(features_scaled)
    isf_preds = isf.predict(encoded)
    isf_preds = np.where(isf_preds == -1, 1, 0)
    xgb_input = np.column_stack((encoded, isf_preds))
    preds = xgb.predict(xgb_input)
    return preds

# === Upload client-side CSV ===
uploaded_file = st.file_uploader("Upload client CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        original_df, features_scaled = preprocess_dataframe(df)
        preds = predict_anomalies(features_scaled)

        original_df["Prediction"] = np.where(preds == 1, "Anomaly", "Normal")

        st.subheader(" Prediction Results")
        st.dataframe(original_df)

        # Anomaly Summary
        st.subheader(" Anomaly Summary")
        total = len(preds)
        anomalies = sum(preds)
        normal = total - anomalies

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", total)
        col2.metric("Normal", normal)
        col3.metric("Anomalies", anomalies)

        #Download part
        csv_out = original_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Prediction CSV", csv_out, "client_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"Error: {str(e)}")
