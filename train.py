import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTEENN
import matplotlib.pyplot as plt
import joblib
import os

# File paths
DATA_PATH = "static/data/locations.csv"
MODEL_PATH = "static/models/location_recommender.pkl"

def train_model():
    """
    Trains a model to recommend locations based on budget and other features.
    Saves the trained model to the specified path.
    """
    print("Loading dataset...")
    data = pd.read_csv(DATA_PATH)

    # Preprocess the data
    print("Preprocessing dataset...")

    # Drop unnecessary columns if they exist
    if 'name' in data.columns:
        data = data.drop(columns=['name'])

    # Check data types
    print("Data types of columns:")
    print(data.dtypes)

    # Convert categorical columns to numeric using one-hot encoding
    X = pd.get_dummies(data.drop(columns=['affordability']), drop_first=True)
    y = data['affordability']

    # Print class distribution before resampling
    print("Class distribution before resampling:")
    print(y.value_counts())

    # Handle class imbalance using SMOTE-ENN
    smote_enn = SMOTEENN(random_state=42)
    X_resampled, y_resampled = smote_enn.fit_resample(X, y)

    # Print class distribution after resampling
    print("Class distribution after resampling:")
    print(y_resampled.value_counts())

    # Split the data into training and testing sets
    print("Splitting dataset into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Initialize and train XGBoost model with hyperparameter tuning
    print("Tuning hyperparameters...")
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 6, 10, 15],
        'min_child_weight': [1, 5, 10],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.7, 1.0],
        'gamma': [0, 1, 5],
        'scale_pos_weight': [1, 2, 3]
    }

    xgb_model = XGBClassifier(
        objective='multi:softmax',  # Multi-class classification
        num_class=3,  # Adjust based on the number of classes in your problem
        use_label_encoder=False,  # To avoid warnings
        eval_metric='mlogloss',  # Log loss for multi-class
        n_jobs=-1  # Use all cores for parallel processing
    )

    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        n_iter=50,  # Number of random combinations to try
        cv=5,
        scoring='accuracy',
        n_jobs=-1,  # Use all CPU cores
        random_state=42
    )

    random_search.fit(X_train, y_train)
    print("Best parameters found: ", random_search.best_params_)

    # Train the model with the best parameters
    print("Training the model...")
    best_model = random_search.best_estimator_
    best_model.fit(X_train, y_train)

    # Evaluate the model
    print("Evaluating the model...")
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature importance plot
    print("Feature importance plot...")
    feature_importances = best_model.feature_importances_
    features = X.columns
    sorted_idx = feature_importances.argsort()

    plt.figure(figsize=(10, 6))
    plt.barh(features[sorted_idx], feature_importances[sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance in Model')
    plt.show()

    # Save the trained model
    print("Saving the model...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"Error during training: {e}")
