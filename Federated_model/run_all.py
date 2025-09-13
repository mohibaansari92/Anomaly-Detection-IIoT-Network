import subprocess
import time
import os
import seaborn as sns

# Ensure output directory exists
os.makedirs("results", exist_ok=True)

# Paths to client datasets
client_datasets = [
    "federated_model/client_0.csv",
    "federated_model/client_1.csv",
    "federated_model/client_2.csv",
    "federated_model/client_3.csv",
    "federated_model/client_4.csv"
]

# Start the Flower server
print("Starting Flower server...")
server_process = subprocess.Popen(["python", "server.py"])
time.sleep(5)  # Let the server initialize

# Start all clients
client_processes = []
for idx, dataset in enumerate(client_datasets):
    print(f" Starting client {idx} with dataset: {dataset}")
    client = subprocess.Popen(["python", "client.py", dataset])
    client_processes.append(client)
    time.sleep(1)  # Small delay between clients

# Wait for all clients to finish
for client in client_processes:
    client.wait()

# Kill the server after clients finish
server_process.terminate()
print(" Federated learning completed.")

# Optional: Display results
print("\nResults Summary:")
for idx in range(len(client_datasets)):
    report_file = f"classification_report_{idx}.txt"
    if os.path.exists(report_file):
        print(f"\n Classification Report - Client {idx}:")
        with open(report_file, "r") as f:
            print(f.read())
    else:
        print(f" No report found for client {idx}")