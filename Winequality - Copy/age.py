import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib

# Load the wine dataset
data = pd.read_csv('winequality-red.csv')

# Generate synthetic age data
np.random.seed(42)
data['age'] = (data['fixed acidity'] * 0.3 +
               data['volatile acidity'] * 0.2 +
               data['citric acid'] * 0.1 +
               data['residual sugar'] * 0.2 +
               data['chlorides'] * 0.05 +
               data['free sulfur dioxide'] * 0.1 +
               data['total sulfur dioxide'] * 0.1 +
               data['density'] * 0.1 +
               data['pH'] * 0.1 +
               data['sulphates'] * 0.2 +
               data['alcohol'] * 0.3 +
               np.random.normal(0, 0.5, len(data)))  # Adding some noise

# Define features and target
X = data.drop(columns=['age', 'quality'])
y = data['age']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a regression model to predict age
age_model = LinearRegression()
age_model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = age_model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae:.2f}')

# Save the model and scaler
joblib.dump(age_model, 'age_model.pkl')
joblib.dump(scaler, 'age_scaler.pkl')