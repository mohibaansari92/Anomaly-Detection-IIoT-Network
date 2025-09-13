import flwr as fl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#from utils import create_autoencoder, extract_encoder, train_isolation_forest, train_xgboost
from sklearn.metrics import classification_report, accuracy_score

class FLClient(fl.client.NumPyClient):
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
        self.encoder = None
        self.isf = None
        self.xgb = None

    def get_data(self):
        df = pd.read_csv(self.data_path)
        X = df.drop("label", axis=1)  # Replace 'label' if your column has another name
        y = df["label"]

        # Normalize
        X = (X - X.min()) / (X.max() - X.min())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def fit(self, parameters, config):
        X_train, X_test, y_train, y_test = self.get_data()
        input_dim = X_train.shape[1]

        # 1. Autoencoder training
        autoencoder = create_autoencoder(input_dim)
        if parameters:
            autoencoder.set_weights(parameters)

        autoencoder.fit(X_train, X_train, epochs=5, batch_size=32, verbose=0)
        self.model = autoencoder

        # 2. Encode features
        self.encoder = extract_encoder(autoencoder)
        X_train_encoded = self.encoder.predict(X_train)

        # 3. Isolation Forest
        self.isf, isf_scores, isf_preds = train_isolation_forest(X_train_encoded)

        # 4. XGBoost on encoded + ISF output
        self.xgb = train_xgboost(X_train_encoded, isf_preds, y_train)

        return self.model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):
        X_train, X_test, y_train, y_test = self.get_data()
        self.model.set_weights(parameters)

        # Encode test data
        X_test_encoded = self.encoder.predict(X_test)

        # Use ISF to get test anomaly preds
        isf_preds = self.isf.predict(X_test_encoded)
        isf_preds = np.where(isf_preds == -1, 1, 0)

        # XGBoost prediction
        xgb_input = np.column_stack((X_test_encoded, isf_preds))
        xgb_preds = self.xgb.predict(xgb_input)

        # Evaluate
        acc = accuracy_score(y_test, xgb_preds)
        print("🔍 Classification Report:\n", classification_report(y_test, xgb_preds))
        return float(acc), len(X_test), {"accuracy": float(acc)}

    def get_parameters(self, config):
        return self.model.get_weights()
