import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix, average_precision_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib
import os
import logging
from collections import Counter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure the 'models' directory exists
os.makedirs(os.path.join('static', 'models'), exist_ok=True)

# Load the dataset
data_path = os.path.join('static', 'data', 'locations.csv')  # Adjust the path if needed
logger.info("Loading dataset...")
df = pd.read_csv(data_path)

# Handle missing values
logger.info("Handling missing values...")
df.ffill(inplace=True)

# Encode categorical columns
logger.info("Encoding categorical columns...")
categorical_columns = ['type', 'business_size', 'target_audience', 'affordability', 'competitors']
encoders = {}
for col in categorical_columns:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col])

# Encode target variable 'foot_traffic'
logger.info("Encoding target variable 'foot_traffic'...")
foot_traffic_encoder = LabelEncoder()
df['foot_traffic'] = foot_traffic_encoder.fit_transform(df['foot_traffic'])

# Scale numerical columns
logger.info("Scaling numerical columns...")
scaler = StandardScaler()
df['budget_scaled'] = scaler.fit_transform(df[['budget']])

# Define features and target
logger.info("Selecting features and target variable...")
X = df[['type', 'business_size', 'target_audience', 'affordability', 'competitors', 'budget_scaled']]
y = df['foot_traffic']

# Split the data into training and testing sets
logger.info("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Adjust SMOTE k_neighbors based on class distribution
class_counts = Counter(y_train)
min_class_samples = min(class_counts.values())
k_neighbors = min(5, min_class_samples - 1) if min_class_samples > 1 else 1
logger.info(f"Setting SMOTE k_neighbors to {k_neighbors} based on smallest class sample size.")

# Set class weights if 'high foot traffic' is the most important class
class_weights = {i: 1 for i in range(len(np.unique(y)))}  # Equal weights initially
# Update class weights for 'high foot traffic' (assuming it's the class with the highest label)
# Replace 'high_class_label' with the actual label for high foot traffic
high_class_label = 2  # Example class label for high foot traffic
class_weights[high_class_label] = 5  # Give more weight to high foot traffic

pipeline = Pipeline(steps=[
    ('smote', SMOTE(random_state=42, k_neighbors=k_neighbors)),
    ('model', RandomForestClassifier(random_state=42, class_weight=class_weights))
])

# Train the model
logger.info("Training the model with pipeline...")
pipeline.fit(X_train, y_train)

# Evaluate with cross-validation
logger.info("Evaluating with cross-validation...")
cv_pipeline = Pipeline(steps=[
    ('model', RandomForestClassifier(random_state=42, class_weight=class_weights))
])
cv_scores = cross_val_score(cv_pipeline, X, y, cv=5, scoring='accuracy')
logger.info(f"Cross-validation accuracy: {cv_scores.mean():.2f}")

# Evaluate the model on the test set
logger.info("Evaluating the model on test set...")
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
logger.info(f"Model Accuracy: {accuracy:.2f}")

f1 = f1_score(y_test, y_pred, average='weighted')
logger.info(f"F1 Score: {f1:.2f}")

# Confusion Matrix and Classification Report
conf_matrix = confusion_matrix(y_test, y_pred)
logger.info(f"Confusion Matrix:\n{conf_matrix}")
logger.info("Classification Report:\n" + classification_report(y_test, y_pred))

# Analyze feature importance
logger.info("Analyzing feature importance...")
model = pipeline.named_steps['model']
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)
logger.info(f"Feature Importance:\n{feature_importance}")

# Save the trained model and encoders
logger.info("Saving the model and encoders...")
joblib.dump(pipeline, os.path.join('static', 'models', 'location_recommender_model_rf.pkl'))
joblib.dump(scaler, os.path.join('static', 'models', 'budget_scaler.pkl'))

# Save encoders for categorical variables
for col, encoder in encoders.items():
    joblib.dump(encoder, os.path.join('static', 'models', f'{col}_encoder.pkl'))

# Save the foot_traffic encoder
joblib.dump(foot_traffic_encoder, os.path.join('static', 'models', 'foot_traffic_encoder.pkl'))

logger.info("Model and encoders have been saved to files.")
