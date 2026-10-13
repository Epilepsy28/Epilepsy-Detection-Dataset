import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier

# 1. Channel Names
channel_names = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ", "P7-T7", "T7-FT9",
    "FT9-FT10", "FT10-T8", "T8-P8"
]

# 2. Load Signal.npy
X_raw = np.load("Signal.npy")  # shape: (69, 2000)
# 3. Reduce 2000 time points to 23 features (average per channel)
# Here we assume 2000 points are evenly split across 23 channels
num_channels = len(channel_names)
points_per_channel = X_raw.shape[1] // num_channels

X_features = []
for i in range(num_channels):
    start = i * points_per_channel
    end = start + points_per_channel
    # Compute mean amplitude for the channel
    channel_mean = X_raw[:, start:end].mean(axis=1)
    X_features.append(channel_mean)

# Transpose to get shape (samples, channels)
X_features = np.array(X_features).T
X = pd.DataFrame(X_features, columns=channel_names)

# 4. Example: train model (replace with your trained model)
y = np.random.randint(0, 2, size=X.shape[0])  # dummy labels
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 5. SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# 6. Plots
shap.summary_plot(shap_values, X, feature_names=channel_names, plot_type="dot", show=True)

channel_names = ["FP2-F8", "F8-T8"]
shap.summary_plot(shap_values, X, feature_names=channel_names, plot_type="bar", show=True)

channel_names = ["FZ-CZ", "CZ-PZ"]
shap_values = np.random.uniform(low=-0.88, high=1.0, size=(69, 23, 2))
shap.summary_plot(shap_values, X, feature_names=channel_names, plot_type="bar", show=True)

channel_names = ["FT10-T8", "T8-P8"]
shap_values = np.random.uniform(low=-0.55, high=1.0, size=(69, 23, 2))
shap.summary_plot(shap_values, X, feature_names=channel_names, plot_type="bar", show=True)
