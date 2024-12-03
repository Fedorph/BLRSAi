import os
import joblib
import pandas as pd

# Paths to model and encoders
MODEL_PATH = os.path.join('static', 'models', 'location_model.pkl')
ENCODER_PATH = os.path.join('static', 'models', 'location_encoders.pkl')

# Load the model and encoders
model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODER_PATH)

# Load the dataset for recommendations
DATA_PATH = os.path.join('static', 'data', 'locations.csv')
locations_df = pd.read_csv(DATA_PATH)


def preprocess_input(user_input, encoders):
    """
    Encodes user input using the label encoders.
    
    Args:
        user_input (dict): Dictionary with user input for prediction.
        encoders (dict): Preloaded label encoders.
    
    Returns:
        dict: Encoded input.
    """
    encoded_input = {}
    for column, value in user_input.items():
        if column in encoders:
            encoded_input[column] = encoders[column].transform([value])[0]
        else:
            encoded_input[column] = value
    return encoded_input


def recommend_locations(budget, user_input):
    """
    Recommend locations based on budget and user preferences.
    
    Args:
        budget (int): Maximum budget for location.
        user_input (dict): User input for recommendation.
    
    Returns:
        pd.DataFrame: Filtered locations meeting the criteria.
    """
    # Preprocess user input for model prediction
    processed_input = preprocess_input(user_input, encoders)
    input_df = pd.DataFrame([processed_input])

    # Predict the foot traffic for the given user input
    foot_traffic_pred = model.predict(input_df)[0]

    # Reverse transform the foot traffic prediction to human-readable form
    foot_traffic_label = encoders['foot_traffic'].inverse_transform([foot_traffic_pred])[0]

    # Filter locations within the budget and matching the predicted foot traffic
    filtered_locations = locations_df[
        (locations_df['budget'] <= budget) & 
        (locations_df['foot_traffic'] == foot_traffic_label)
    ]

    return filtered_locations


if __name__ == "__main__":
    # Example usage
    # User input example
    user_input = {
        "target_audience": "medium",
        "affordability": "moderate",
        "competitors": "low",
        "business_size": "small",
        "type": "traditional"
    }
    user_budget = 5000000  # User's maximum budget

    # Generate recommendations
    recommendations = recommend_locations(user_budget, user_input)

    if not recommendations.empty:
        print("Recommended Locations:")
        print(recommendations[['name', 'type', 'budget', 'foot_traffic']])
    else:
        print("No locations found within the budget and criteria.")
