# ⚡ Electricity Theft Detection using SOM (Self-Organizing Maps)

This project detects electricity theft using an unsupervised machine learning approach based on **Self-Organizing Maps (SOM)**. It analyzes customer usage behavior and identifies abnormal consumption patterns indicating possible theft.

---

## 🚀 Project Overview

- Uses **SOM (MiniSom)** for anomaly detection
- Applies **feature engineering** on electricity consumption data
- Generates **risk scores** for each user
- Classifies users into **Normal / High Risk (Theft)** categories
- Includes **visualization using U-Matrix and component planes**
- Optimized using **F1-score based threshold tuning**

---

## 🧠 Machine Learning Approach

- Self-Organizing Maps (Unsupervised Learning)
- MinMax Scaling for normalization
- Anomaly scoring using:
  - Quantization Error
  - U-Matrix distance
- Threshold optimization using F1-score

---

## 📊 Features Used

- Electricity usage patterns (kWh)
- Appliance count & residents
- Voltage fluctuations
- Time-based features (sin/cos encoding)
- Bill delay patterns
- Previous theft history
- Usage ratios & spike detection

---

## 📁 Project Structure

```
som_artifacts/
├── scaler.pkl
├── som_model.pkl
├── u_matrix_norm.pkl
├── risk_threshold.pkl
├── training_results.csv
├── som_u_matrix.png
├── component_planes/
```

---

## ⚙️ Installation

```bash
pip install numpy pandas matplotlib scikit-learn minisom joblib
```

---

## ▶️ Run Project

### Train Model
```bash
python electricity_theft_detection.py
```

### Test / Prediction
```bash
python prediction_script.py
```

---

## 📈 Output

- Risk Score per customer
- Normal / Theft classification
- SOM U-Matrix visualization
- Component plane analysis
- Confusion matrix + F1-score evaluation
## 📊 Results

### SOM U-Matrix
<img width="2388" height="2504" alt="som_u_matrix" src="https://github.com/user-attachments/assets/1c2777fd-a41d-451f-90de-4311f893a3e8" />




## 📌 Key Highlights

- Real-world inspired electricity fraud detection system
- Combines unsupervised ML + anomaly detection
- Fully explainable SOM visualization
- Optimized threshold selection for best performance

---

## 🔮 Future Improvements

- Real-time smart meter integration
- Deep learning-based anomaly detection
- Web dashboard for utility companies
- Streaming data processing (IoT-based)

---

## 👨‍💻 Author

Muhammad Hamza Shahzad
