# =====================================================
# Electricity Theft Detection - Prediction + SOM Plot
# =====================================================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from minisom import MiniSom

# -------------------------------
# 0️⃣ Load Model Artifacts
# -------------------------------
MODEL_DIR = "som_artifacts"

scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
som = joblib.load(f"{MODEL_DIR}/som_model.pkl")
u_matrix_norm = joblib.load(f"{MODEL_DIR}/u_matrix_norm.pkl")
threshold = joblib.load(f"{MODEL_DIR}/risk_threshold.pkl")

# -------------------------------
# 1️⃣ Create Test Data (5 Samples)
# -------------------------------
test_data = pd.DataFrame([
    [464,3,3,7,25,0,1,43,25,1],
    [392,3,2,6,8,1,0,26,11,0],
    [199,3,0,4,10,0,1,50,36,1],
    [250,3,1,6,24,0,1,21,72,1],
    [67,1,0,1,22,0,1,27,1,0]
], columns=[
    "Usage (kWh)","TimeOfDay","VoltageFluctuations","NumberOfResidents",
    "ApplianceCount","IndustrialAreaNearby","PreviousTheftHistory",
    "AverageDailyUsage","BillPaymentDelay (days)","UnusualUsageSpike"
])

# -------------------------------
# 2️⃣ Feature Engineering
# -------------------------------
test_data["Usage_per_Appliance"] = test_data["Usage (kWh)"] / (test_data["ApplianceCount"] + 1)
test_data["Usage_per_Person"] = test_data["Usage (kWh)"] / (test_data["NumberOfResidents"] + 1)
test_data["Spike_to_Average_Ratio"] = test_data["Usage (kWh)"] / (test_data["AverageDailyUsage"] + 1)

test_data["Time_sin"] = np.sin(2 * np.pi * test_data["TimeOfDay"] / 24)
test_data["Time_cos"] = np.cos(2 * np.pi * test_data["TimeOfDay"] / 24)

features = [
    'Usage (kWh)',
    'VoltageFluctuations',
    'NumberOfResidents',
    'ApplianceCount',
    'IndustrialAreaNearby',
    'PreviousTheftHistory',
    'AverageDailyUsage',
    'BillPaymentDelay (days)',
    'UnusualUsageSpike',
    'Usage_per_Appliance',
    'Usage_per_Person',
    'Spike_to_Average_Ratio',
    'Time_sin',
    'Time_cos'
]

X_test = test_data[features].values

# -------------------------------
# 3️⃣ Scale Features
# -------------------------------
X_scaled = scaler.transform(X_test)

# -------------------------------
# 4️⃣ Compute Anomaly Scores
# -------------------------------
def compute_anomaly_score(sample):
    winner = som.winner(sample)
    weight = som.get_weights()[winner]
    qe = np.linalg.norm(sample - weight)
    u_score = u_matrix_norm[winner]
    return 0.6 * qe + 0.4 * u_score

anomaly_scores = np.array([compute_anomaly_score(x) for x in X_scaled])
risk_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min() + 1e-8)

test_data["Theft_Risk_Score"] = risk_scores

# -------------------------------
# 5️⃣ Apply Threshold
# -------------------------------
test_data["Theft_Risk_Label"] = np.where(risk_scores > threshold, "High Risk", "Normal")
test_data["Confidence"] = np.clip((risk_scores - threshold) / (1 - threshold + 1e-8), 0, 1)

# -------------------------------
# 6️⃣ Show Results
# -------------------------------
print("✅ Prediction completed for 5 test samples:\n")
print(test_data[["Usage (kWh)", "Theft_Risk_Score", "Theft_Risk_Label", "Confidence"]])

# -------------------------------
# 7️⃣ Map Samples on SOM U-Matrix
# -------------------------------
plt.figure(figsize=(8, 8))
plt.imshow(u_matrix_norm.T, cmap="Reds", origin='lower')
plt.colorbar(label="Normalized Distance")

# Prepare mapping for visualization
results = []
for i, sample in enumerate(X_scaled):
    winner = som.winner(sample)
    label = test_data.loc[i, "Theft_Risk_Label"]
    name = f"{i+1}"  # Sample name
    results.append((name, winner, label))

# Plot each sample on the U-matrix
for name, winner, label in results:
    color = "red" if label == "High Risk" else "green"
    plt.scatter(winner[0], winner[1], c=color, s=120, edgecolors='k')
    plt.text(winner[0]+0.1, winner[1]+0.1, name, fontsize=9)

plt.title("SOM Customer Mapping (Red = High Risk, Green = Normal)")
plt.xlabel("SOM X")
plt.ylabel("SOM Y")
plt.tight_layout()
plt.show()
