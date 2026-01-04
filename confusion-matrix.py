import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Confusion matrix values
cm = np.array([[2168, 878],
               [748, 8100]])

# Plot
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Normal", "Theft"],
            yticklabels=["Normal", "Theft"])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")

plt.show()
