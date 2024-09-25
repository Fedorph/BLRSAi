import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

# 1. Load your dataset
try:
    # Update the path to your CSV file accordingly
    data = pd.read_csv('static/data/locations.csv')  
except FileNotFoundError as e:
    print("Error: The file was not found. Please check the path.")
    raise e

# Assuming your dataset has these columns
features = ['price', 'target_audience', 'foot_traffic', 'affordability', 'resources', 
            'revenue_potential', 'competitors', 'business_category', 'income', 
            'budget', 'nearness_to_market', 'sustainability', 'road_connection', 
            'business_growth', 'government_policies']
target = 'business_success'  # The column you want to predict

# Check if all features are in the dataset
missing_features = [feature for feature in features if feature not in data.columns]
if missing_features:
    raise ValueError(f"Missing features in dataset: {missing_features}")

# 2. Prepare the data
X = data[features]
y = data[target]

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 5. Evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 6. Save the model to a file
try:
    with open('model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    print("Model has been saved successfully to 'model.pkl'!")
except Exception as e:
    print("Error saving the model:", e)

# 7. Save the scaler to a file
try:
    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    print("Scaler has been saved successfully to 'scaler.pkl'!")
except Exception as e:
    print("Error saving the scaler:", e)
