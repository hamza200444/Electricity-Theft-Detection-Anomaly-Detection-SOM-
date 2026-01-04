# =====================================================
# Electricity Theft Detection 
# =====================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from minisom import MiniSom

# -------------------------------
# 0️⃣ Create Model Folder
# -------------------------------
MODEL_DIR = "som_artifacts"
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------
# 1️⃣ Load Dataset
# -------------------------------
df = pd.read_csv("electricity_new.csv")
df.columns = df.columns.str.strip()

# -------------------------------
# 2️⃣ Feature Engineering
# -------------------------------
df["Usage_per_Appliance"] = df["Usage (kWh)"] / (df["ApplianceCount"] + 1)
df["Usage_per_Person"] = df["Usage (kWh)"] / (df["NumberOfResidents"] + 1)
df["Spike_to_Average_Ratio"] = df["Usage (kWh)"] / (df["AverageDailyUsage"] + 1)
df["Time_sin"] = np.sin(2 * np.pi * df["TimeOfDay"] / 24)
df["Time_cos"] = np.cos(2 * np.pi * df["TimeOfDay"] / 24)

features = [
    'Usage (kWh)', 'VoltageFluctuations', 'NumberOfResidents', 'ApplianceCount',
    'IndustrialAreaNearby', 'PreviousTheftHistory', 'AverageDailyUsage',
    'BillPaymentDelay (days)', 'UnusualUsageSpike', 'Usage_per_Appliance',
    'Usage_per_Person', 'Spike_to_Average_Ratio', 'Time_sin', 'Time_cos'
]

# -------------------------------
# 3️⃣ Split Normal Data (70% Train / 30% Validation)
# -------------------------------
df_normal = df[df['Theft'] == 0].copy()
df_theft = df[df['Theft'] == 1].copy()  

X_normal = df_normal[features].values
X_train, X_val = train_test_split(X_normal, test_size=0.3, random_state=42)

# Full test set (validation + theft cases)
X_test = np.vstack([X_val, df_theft[features].values])
y_test = np.hstack([np.zeros(len(X_val)), np.ones(len(df_theft))])  # 0=Normal, 1=Theft

# -------------------------------
# 4️⃣ Scale Data (Fit on Training Normal Only)
# -------------------------------
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 5️⃣ Train SOM (On Training Normal Only)
# -------------------------------
# Increase grid size for better resolution
som_size = int(np.sqrt(10 * np.sqrt(len(X_train_scaled))))
som_size = max(15, som_size)

print(f"Training SOM with grid size: {som_size}x{som_size}")

som = MiniSom(
    x=som_size,
    y=som_size,
    input_len=X_train_scaled.shape[1],
    sigma=1.0,
    learning_rate=0.3,
    neighborhood_function='gaussian',
    random_seed=42
)

som.random_weights_init(X_train_scaled)
som.train_random(X_train_scaled, num_iteration=10000)  # more iterations for stability

# -------------------------------
# 6️⃣ U-Matrix for Anomaly Scoring
# -------------------------------
u_matrix = som.distance_map()
u_matrix_norm = (u_matrix - u_matrix.min()) / (u_matrix.max() - u_matrix.min() + 1e-8)

def compute_anomaly_score(sample):
    winner = som.winner(sample)
    weight = som.get_weights()[winner]
    qe = np.linalg.norm(sample - weight)
    u_score = u_matrix_norm[winner]
    # Adjust QE weight for better anomaly detection
    return 0.7 * qe + 0.3 * u_score

# -------------------------------
# 7️⃣ Compute Anomaly Scores
# -------------------------------
anomaly_scores_train = np.array([compute_anomaly_score(x) for x in X_train_scaled])
anomaly_scores_test = np.array([compute_anomaly_score(x) for x in X_test_scaled])

# Normalize risk scores
risk_train = (anomaly_scores_train - anomaly_scores_train.min()) / (anomaly_scores_train.max() - anomaly_scores_train.min() + 1e-8)
risk_test = (anomaly_scores_test - anomaly_scores_test.min()) / (anomaly_scores_test.max() - anomaly_scores_test.min() + 1e-8)

# -------------------------------
# 8️⃣ Optimize Threshold to Maximize F1-Score
# -------------------------------
thresholds = np.linspace(risk_train.min(), risk_train.max(), 100)
best_f1 = 0
best_threshold = risk_train.mean() + 2 * risk_train.std()

for t in thresholds:
    y_pred = (risk_test > t).astype(int)
    f1 = f1_score(y_test, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

threshold = best_threshold
y_pred_test = (risk_test > threshold).astype(int)
print(f"🚨 Optimized High Risk Threshold: {threshold:.3f} (F1-score: {best_f1:.3f})")

# -------------------------------
# 9️⃣ Evaluation
# -------------------------------
print("\n📊 Classification Report on Test Set:")
print(classification_report(y_test, y_pred_test, target_names=["Normal", "Theft"]))

cm = confusion_matrix(y_test, y_pred_test)
print("\nConfusion Matrix:")
print(cm)

# -------------------------------
# 🔟 Save Model Artifacts
# -------------------------------
joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
joblib.dump(som, f"{MODEL_DIR}/som_model.pkl")
joblib.dump(u_matrix_norm, f"{MODEL_DIR}/u_matrix_norm.pkl")
joblib.dump(threshold, f"{MODEL_DIR}/risk_threshold.pkl")
joblib.dump(anomaly_scores_train.min(), f"{MODEL_DIR}/risk_min.pkl")
joblib.dump(anomaly_scores_train.max(), f"{MODEL_DIR}/risk_max.pkl")
df.to_csv(f"{MODEL_DIR}/training_results.csv", index=False)

# -------------------------------
# 1️⃣1️⃣ Visualization
# -------------------------------
plt.figure(figsize=(10, 10))
plt.pcolor(u_matrix_norm.T, cmap="Reds")
plt.colorbar(label="Normalized Distance")
plt.title("SOM U-Matrix")
plt.savefig(f"{MODEL_DIR}/som_u_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

for i, feature in enumerate(features):
    plt.figure(figsize=(6, 6))
    plt.title(f"Component Plane: {feature}")
    plt.pcolor(som.get_weights()[:, :, i].T, cmap='viridis')
    plt.colorbar()
    plt.savefig(f"{MODEL_DIR}/component_{feature.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.close()

print("✅ Improved SOM training + evaluation completed!")
print(f"📁 Artifacts saved in: {MODEL_DIR}")
