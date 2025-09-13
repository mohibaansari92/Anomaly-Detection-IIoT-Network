import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os

# Load the full dataset
df = pd.read_csv("wustl_iiot_2021.csv")

# Shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Features + target
X = df.drop("Target", axis=1)
y = df["Target"]

# Use StratifiedKFold to split data into 5 balanced sets
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
output_dir = "federated_model"
os.makedirs(output_dir, exist_ok=True)

for i, (_, test_idx) in enumerate(skf.split(X, y)):
    client_df = df.iloc[test_idx]
    client_df.to_csv(f"{output_dir}/client_{i}.csv", index=False)

print("Stratified data split done for 5 clients.")