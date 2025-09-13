import pandas as pd
import numpy as np
import ipaddress
import joblib
from tensorflow.keras.models import load_model
from feature_extractor import get_event
import socket
import struct
import re

# === Constants ===
INPUT_DIM = 8
THRESHOLD = 0.7

# === Load models ===
encoder = load_model("saved_models/encoder.h5")
isf = joblib.load("saved_models/isolation_forest.pkl")
xgb = joblib.load("saved_models/xgboost_model.pkl")
scaler = joblib.load("saved_models/minmax_scaler.pkl")

# === Helper functions ===
def is_valid_ipv4(ip):
    pattern = re.compile(r'^(\d{1,3}\.){3}\d{1,3}$')
    return pattern.match(ip) is not None

def ip_to_int(ip):
    return struct.unpack("!I", socket.inet_aton(ip))[0]

def preprocess_features(event):
    event["SrcAddr"] = ip_to_int(event["SrcAddr"]) if is_valid_ipv4(event["SrcAddr"]) else None
    event["DstAddr"] = ip_to_int(event.get("DstAddr", "0.0")) if is_valid_ipv4(event.get("DstAddr", "0.0")) else None
    event["Proto"] = int(event.get("Proto", 0))

    feature_cols = ["SrcAddr", "DstAddr", "Dur", "TotPkts", "TotBytes", "SrcJitter", "DstJitter", "Proto"]
    df = pd.DataFrame([event])[feature_cols]
    return scaler.transform(df)

def predict(event, threshold=THRESHOLD):
    X = preprocess_features(event)
    X_encoded = encoder.predict(X)
    isf_preds = isf.predict(X_encoded)
    isf_preds = np.where(isf_preds == -1, 1, 0)
    xgb_input = np.column_stack((X_encoded, isf_preds))
    prob = xgb.predict_proba(xgb_input)[0][1]
    prediction = 1 if prob > threshold else 0
    return prediction, prob

# === Live Monitor ===
if __name__ == "__main__":
    print(" Real-time Anomaly Detection Running...\n")
    print(f" Using threshold: {THRESHOLD:.2f}\n")

    while True:
        event = get_event()
        if not event:
            print(" No packet detected. Waiting...\n")
            continue

        result, score = predict(event)
        print(f" Monitored Event: {event}")
        print(f" Anomaly Score: {score:.4f}")

        if result == 1:
            print(" ALERT: Anomaly Detected!\n")
        else:
            print("Normal activity.\n")