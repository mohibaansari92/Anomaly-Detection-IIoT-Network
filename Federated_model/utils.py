from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import IsolationForest
import xgboost as xgb
import numpy as np
from collections import Counter
from sklearn.utils import shuffle

# Input dimension (must match your dataset features)
input_dim = 8

# === Autoencoder ===
def create_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dense(16, activation='relu')(encoded)
    
    # Use slightly tighter bottleneck for generalization
    bottleneck = Dense(4, activation='relu', name='bottleneck')(encoded)

    decoded = Dense(16, activation='relu')(bottleneck)
    decoded = Dense(32, activation='relu')(decoded)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(128, activation='relu')(decoded)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    return autoencoder

# === Encoder extractor ===
def extract_encoder(autoencoder):
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer("bottleneck").output)
    return encoder

# === Isolation Forest Training ===
def train_isolation_forest(X_encoded, y_train=None):
    contamination = 0.05
    if y_train is not None:
        num_anomalies = sum(y_train == 1)
        ratio = num_anomalies / len(y_train)
        contamination = max(0.01, min(0.2, ratio))  # Keep in sane range

    isf = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    isf.fit(X_encoded)
    scores = isf.decision_function(X_encoded)
    preds = isf.predict(X_encoded)
    preds = (preds == -1).astype(int)
    return isf, scores, preds

# === XGBoost Training ===
def train_xgboost(X_encoded, isf_preds, y_train):
    xgb_input = np.column_stack((X_encoded, isf_preds))
    xgb_input, y_train = shuffle(xgb_input, y_train, random_state=42)
    counter = Counter(y_train)
    weight_ratio = counter[0] / counter[1]

    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=weight_ratio,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(xgb_input, y_train)
    return model