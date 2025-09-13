# IIoT Anomaly Detection

This project is about finding unusual behavior (anomalies) in "Industrial IoT (IIoT) networks".  
I have built it in two ways:
1. Centralized Model – where all data is collected and trained in one place.  
2. Federated Model – where training happens on multiple devices (clients) without sharing raw data, using the Flower framework.

---

#Project Structure
- centralized_model/ → contains the centralized approach (Isolation Forest + Autoencoder, optimized with XGBoost).  
- federated_model/ → contains the federated learning approach using Flower.   

---

#Dataset
The dataset is too large (390 MB) to upload here.  
You can download it from Google Drive:  

[Download Dataset](https://drive.google.com/file/d/1bI51UJo5U-wGDxi1MXfHUNYPW78Kqo5w/view?usp=drive_link)

