import flwr as fl
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

# Global model and other necessary variables should be defined here
global_model = None  # Placeholder for the global model
encoder = None  # Placeholder for the encoder
isf = None  # Placeholder for the isolation forest model

def start_server():
    strategy = fl.server.strategy.FedAvg(
        evaluate_metrics_aggregation_fn=lambda metrics: {
            "accuracy": float(np.mean([m[1]["accuracy"] for m in metrics])),
            "precision": float(np.mean([m[1]["precision"] for m in metrics])),
            "recall": float(np.mean([m[1]["recall"] for m in metrics])),
            "f1": float(np.mean([m[1]["f1"] for m in metrics]))
        },
        # Removed on_fit_complete_fn as it may not be supported in your version
    )

    fl.server.start_server(
        server_address="localhost:9092",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )

def weighted_aggregation(results):
    client_updates = []
    client_accuracies = []
    client_precisions = []
    client_recalls = []
    client_f1_scores = []

    for result in results:
        weights = result[1]["weights"]
        accuracy = result[1]["accuracy"]
        precision = result[1]["precision"]
        recall = result[1]["recall"]
        f1 = result[1]["f1"]
        
        client_updates.append(weights)
        client_accuracies.append(accuracy)
        client_precisions.append(precision)
        client_recalls.append(recall)
        client_f1_scores.append(f1)

    # Calculate weights based on accuracy, precision, recall, and F1 score
    total_weight = sum(client_accuracies)  # Using accuracy for normalization
    normalized_weights = [accuracy / total_weight for accuracy in client_accuracies]

    # Perform weighted aggregation
    global_model_weights = np.zeros_like(client_updates[0])  # Assuming all updates have the same shape
    for update, weight in zip(client_updates, normalized_weights):
        global_model_weights += update * weight

    # Update the global model with the aggregated weights
    global_model.set_weights(global_model_weights)

    # Evaluate the global model and print the classification report
    X_test, y_test = load_test_data()  # Load your test data
    evaluate_global_model(global_model, X_test, y_test)

def evaluate_global_model(global_model, X_test, y_test):
    # Get predictions
    X_test_encoded = encoder.predict(X_test)  # Ensure encoder is defined and loaded
    isf_preds_test = isf.predict(X_test_encoded)  # Ensure isf is defined and loaded
    isf_preds_test = np.where(isf_preds_test == -1, 1, 0)
    xgb_input = np.column_stack((X_test_encoded, isf_preds_test))
    y_pred = global_model.predict(xgb_input)

    # Print classification report
    print(classification_report(y_test, y_pred))

def load_test_data():
    # Load your test data here
    test_data_path = "data/test_data.csv"  # Update this path to your actual test data file
    test_data = pd.read_csv(test_data_path)  # Load the test data
    X_test = test_data.drop("Target", axis=1)  # Adjust according to your dataset
    y_test = test_data["Target"]
    return X_test, y_test

if __name__ == "__main__":
    start_server()