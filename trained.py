import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix, precision_recall_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os
import logging
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure the 'models' directory exists
os.makedirs(os.path.join('static', 'models'), exist_ok=True)

# Load the dataset
data_path = os.path.join('static', 'data', 'locations.csv')  # Adjust the path if needed
logger.info("Loading dataset...")
df = pd.read_csv(data_path)

# Inspect the first few rows of the dataset
logger.info(f"Data Preview:\n{df.head()}")

# Preprocessing
logger.info("Handling missing values...")
df.fillna(method='ffill', inplace=True)  # Forward fill missing values

# Encode categorical columns (e.g., 'type', 'business_size', 'target_audience', 'affordability', 'competitors')
categorical_columns = ['type', 'business_size', 'target_audience', 'affordability', 'competitors']
logger.info("Encoding categorical columns...")

# Initialize label encoders for categorical features
encoders = {}
for col in categorical_columns:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col])

# Encode target variable 'foot_traffic' (which is categorical)
logger.info("Encoding target variable 'foot_traffic'...")
foot_traffic_encoder = LabelEncoder()
df['foot_traffic'] = foot_traffic_encoder.fit_transform(df['foot_traffic'])

# Standardize the budget column
logger.info("Scaling budget column...")
scaler = StandardScaler()
df['budget_scaled'] = scaler.fit_transform(df[['budget']])

# Select features (X) and target (y)
logger.info("Selecting features and target variable...")
X = df[['type', 'business_size', 'target_audience', 'affordability', 'competitors', 'budget_scaled']]  # Features
y = df['foot_traffic']  # Target variable (encoded as numeric)

# Split the data into training and testing sets
logger.info("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance using SMOTE
logger.info("Balancing classes using SMOTE...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Train the model
logger.info("Training the model...")
model = XGBClassifier(random_state=42, eval_metric='mlogloss')
model.fit(X_train_balanced, y_train_balanced)

# Evaluate with cross-validation
logger.info("Evaluating with cross-validation...")
cv_scores = cross_val_score(model, X, y, cv=5)
logger.info(f"Cross-validation accuracy: {cv_scores.mean():.2f}")

# Evaluate the model on the test set
logger.info("Evaluating the model on test set...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
logger.info(f"Model Accuracy: {accuracy:.2f}")

f1 = f1_score(y_test, y_pred, average='weighted')
logger.info(f"F1 Score: {f1:.2f}")

# Multiclass Precision-Recall AUC
logger.info("Calculating Precision-Recall AUC for each class...")
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])  # Binarize the true labels
y_pred_prob = model.predict_proba(X_test)  # Get probability estimates

# Calculate Precision-Recall AUC for each class
pr_auc = {}
for i in range(3):  # For each class
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_pred_prob[:, i])
    pr_auc[i] = auc(recall, precision)
    logger.info(f"Precision-Recall AUC for class {i}: {pr_auc[i]:.2f}")

# Plot Precision-Recall curve for each class
plt.figure(figsize=(10, 6))
for i in range(3):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_pred_prob[:, i])
    plt.plot(recall, precision, label=f'Class {i} (AUC = {pr_auc[i]:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Multiclass)')
plt.legend(loc='lower left')
plt.show()

# Confusion Matrix and Classification Report
conf_matrix = confusion_matrix(y_test, y_pred)
logger.info(f"Confusion Matrix:\n{conf_matrix}")
logger.info("Classification Report:\n" + classification_report(y_test, y_pred))

# Feature Importance
logger.info("Analyzing feature importance...")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)
logger.info(f"Feature Importance:\n{feature_importance}")

# Save the trained model and encoders
logger.info("Saving the model and encoders...")
joblib.dump(model, os.path.join('static', 'models', 'location_recommender_model.pkl'))
joblib.dump(scaler, os.path.join('static', 'models', 'budget_scaler.pkl'))

# Save encoders for categorical variables
for col, encoder in encoders.items():
    joblib.dump(encoder, os.path.join('static', 'models', f'{col}_encoder.pkl'))

# Save the foot_traffic encoder
joblib.dump(foot_traffic_encoder, os.path.join('static', 'models', 'foot_traffic_encoder.pkl'))

logger.info("Model and encoders have been saved to files.")
