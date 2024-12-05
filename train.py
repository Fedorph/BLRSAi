import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# File paths
DATA_PATH = "static/data/locations.csv"
MODEL_PATH = "static/models/location_recommender.pkl"

def train_model():
    """
    Trains a Random Forest model to recommend locations based on budget and other features.
    Saves the trained model to the specified path.
    """
    # Load the dataset
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

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Print class distribution after resampling
    print("Class distribution after resampling:")
    print(y_resampled.value_counts())

    # Split the data into training and testing sets
    print("Splitting dataset into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Hyperparameter tuning with GridSearchCV (using parallel processing)
    print("Tuning hyperparameters...")
    param_grid = {
        'n_estimators': [100, 200],  # Reduce number of estimators
        'max_depth': [None, 10],  # Limit depth for faster training
        'min_samples_split': [2, 5],  # Reduce splits
        'min_samples_leaf': [1, 2],  # Limit leaf nodes
        'class_weight': ['balanced', None]  # Adjust class weights
    }
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),  # Use parallel processing
        param_grid=param_grid,
        cv=3,  # Reduce cross-validation folds for speed
        scoring='accuracy',
        n_jobs=-1  # Parallelize grid search itself
    )
    grid_search.fit(X_train, y_train)
    print("Best parameters found: ", grid_search.best_params_)

    # Train the best model
    print("Training the model...")
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    # Evaluate the model
    print("Evaluating the model...")
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))

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
