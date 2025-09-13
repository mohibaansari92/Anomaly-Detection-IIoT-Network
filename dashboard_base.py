import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import time
from feature_extractor import get_event

# Load models
autoencoder = load_model("C:/Users/DELL/BASE_MODEL/model/autoencoder_model.h5", compile=False)
isf = joblib.load("C:/Users/DELL/BASE_MODEL/model/isolation_forest_model.pkl")
xgb_model = joblib.load("C:/Users/DELL/BASE_MODEL/model/xgboost_model.pkl")
scaler = joblib.load("C:/Users/DELL/BASE_MODEL/model/scaler.pkl")

feature_cols = ['SrcAddr', 'DstAddr', 'Dur', 'TotPkts', 'TotBytes', 'SrcJitter', 'DstJitter', 'Proto']

st.set_page_config(page_title="IIoT Real-Time Dashboard", layout="wide")
st.title("IIoT Real-Time Anomaly Detection Dashboard")
st.markdown("Live anomaly detection using Autoencoder, Isolation Forest, and XGBoost.")

placeholder = st.empty()
result_log = []

def preprocess_features(event):
    df = pd.DataFrame([event])[feature_cols]
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    return scaler.transform(df)

def predict(event):
    X = preprocess_features(event)
    X_pred = autoencoder.predict(X)
    reconstruction_error = np.mean(np.square(X - X_pred), axis=1).reshape(-1, 1)
    isf_pred = isf.predict(reconstruction_error)
    isf_pred = np.where(isf_pred == -1, 1, 0)
    xgb_input = np.column_stack((reconstruction_error, isf_pred))
    return xgb_model.predict(xgb_input)[0]

if st.button("Start Monitoring"):
    st.success("Monitoring Started!")
    while True:
        event = get_event()
        if not event:
            st.warning("No packet detected.")
            time.sleep(1)
            continue

        pred = predict(event)
        event["Prediction"] = "Anomaly" if pred == 1 else "Normal"
        result_log.append(event)
        result_log = result_log[-10:]

        df = pd.DataFrame(result_log)
        placeholder.dataframe(df)
        time.sleep(1)
