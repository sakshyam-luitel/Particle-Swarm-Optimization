import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# Load the reshaped data from CSV
df = pd.read_csv('reshaped_fish_data.csv')

# Define the features (X) and the target (y)
X = df[['Position X', 'Position Y', 'Velocity X', 'Velocity Y', 'Best Position X', 'Best Position Y']]
y = df['Fitness Value']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (optional but can improve model performance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model using joblib
joblib.dump(model, 'trained_model.joblib')

# Optionally, save the scaler as well
joblib.dump(scaler, 'scaler.joblib')

print("Model and scaler have been saved successfully.")
