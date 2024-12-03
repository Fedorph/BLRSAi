import pandas as pd
import joblib
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# File paths
MODEL_PATH = 'static/models/location_recommender_model.pkl'
ENCODER_PATHS = {
    'target_audience': 'static/models/target_audience_encoder.pkl',
    'foot_traffic': 'static/models/foot_traffic_encoder.pkl',
    'affordability': 'static/models/affordability_encoder.pkl',
    'competitors': 'static/models/competitors_encoder.pkl',
}
LOCATIONS_CSV_PATH = 'static/data/cleaned_location.csv'

# Verify files exist
for name, path in {**ENCODER_PATHS, 'model': MODEL_PATH, 'locations': LOCATIONS_CSV_PATH}.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name.capitalize()} file not found at {path}")

# Load models and encoders
logger.info("Loading model and encoders...")
model = joblib.load(MODEL_PATH)
target_audience_encoder = joblib.load(ENCODER_PATHS['target_audience'])
foot_traffic_encoder = joblib.load(ENCODER_PATHS['foot_traffic'])
affordability_encoder = joblib.load(ENCODER_PATHS['affordability'])
competitors_encoder = joblib.load(ENCODER_PATHS['competitors'])

# Load locations data once at the beginning to avoid loading it each time
logger.info("Loading locations data...")
locations_df = pd.read_csv(LOCATIONS_CSV_PATH)

def get_top_recommendation(restaurant_type, business_size, budget, state):
    """
    Predicts and returns the top location recommendation based on input parameters.
    The locations should have a budget equal to or above the user's input.
    """
    try:
        # Normalize the 'state' column and filter based on state and budget (>= user input)
        logger.info(f"Filtering locations for state: {state} and budget >= {budget}")
        locations_df['state'] = locations_df['state'].str.strip().str.lower()
        state = state.strip().lower()

        # Filter locations based on state and budget constraint (greater than or equal to input)
        filtered_df = locations_df.query("state == @state and budget >= @budget")

        if filtered_df.empty:
            logger.info("No locations match the given criteria.")
            return [{"error": "No locations match the given criteria."}]

        # Encode restaurant type and business size
        restaurant_type_encoded = 1 if restaurant_type.lower() == 'traditional' else 2
        business_size_encoded = 1 if business_size.lower() == 'small' else 2

        # Prepare input data and predict scores
        logger.info("Predicting location scores...")
        filtered_df['restaurant_type_encoded'] = restaurant_type_encoded
        filtered_df['business_size_encoded'] = business_size_encoded

        # Define the features for the prediction
        features = ['restaurant_type_encoded', 'business_size_encoded', 'budget']
        filtered_df['score'] = model.predict(filtered_df[features])

        # Get the top recommendation based on the score
        top_location = filtered_df.sort_values(by='score', ascending=False).iloc[0]
        logger.info(f"Top location: {top_location['name']} with score: {top_location['score']}")

        # Return the top location details
        return [
            {
                "name": top_location['name'],
                "lat": top_location['lat'],
                "lng": top_location['lng'],
                "target_audience": target_audience_encoder.inverse_transform([int(top_location['target_audience'])])[0],
                "foot_traffic": foot_traffic_encoder.inverse_transform([int(top_location['foot_traffic'])])[0],
                "affordability": affordability_encoder.inverse_transform([int(top_location['affordability'])])[0],
                "competitors": competitors_encoder.inverse_transform([int(top_location['competitors'])])[0],
                "accuracy": round(top_location['score'] * 100, 2),
            }
        ]

    except Exception as e:
        logger.error(f"Error during recommendation: {e}")
        raise RuntimeError(f"Error processing recommendation: {e}")

if __name__ == "__main__":
    try:
        # Get user input for parameters
        restaurant_type = input("Enter the type of restaurant (e.g., 'Traditional'): ").strip()
        business_size = input("Enter the business size (e.g., 'Small'): ").strip()
        budget = float(input("Enter the budget: ").strip())
        state = input("Enter the state (e.g., 'Kaduna'): ").strip()

        # Get recommendation based on user input
        recommendation = get_top_recommendation(
            restaurant_type=restaurant_type,
            business_size=business_size,
            budget=budget,
            state=state
        )
        print("Recommendation:", recommendation)
    except ValueError as ve:
        print(f"Invalid input: {ve}")
    except RuntimeError as err:
        print(f"Error during processing: {err}")
