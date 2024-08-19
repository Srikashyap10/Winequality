import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Load your dataset (ensure it has 11 features)
# Replace 'your_dataset.csv' with the path to your actual dataset
data = pd.read_csv(r'C:\Users\kashy\OneDrive\Desktop\Winequality\winequality-red.csv')

# Select the first 11 features for scaling (adjust column names as needed)
features = data[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 
                 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']]

# Train the StandardScaler with 11 features
scaler = StandardScaler()
scaler.fit(features)

# Save the trained scaler
with open(r'C:\Users\kashy\OneDrive\Desktop\Winequality\scaler_11_features.pkl', 'wb') as f:
    joblib.dump(scaler, f)

print("StandardScaler trained and saved successfully.")