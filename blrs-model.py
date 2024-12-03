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
LOCATIONS_CSV_PATH = 'static/data/locations.csv'

# Verify files exist
for name, path in {**ENCODER_PATHS, 'model': MODEL_PATH, 'locations': LOCATIONS_CSV_PATH}.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name.capitalize()} file not found at {path}")

# Load files
logger.info("Loading model and encoders...")
model = joblib.load(MODEL_PATH)

# Use the correct key from ENCODER_PATHS
target_audience_encoder = joblib.load(ENCODER_PATHS['target_audience'])
foot_traffic_encoder = joblib.load(ENCODER_PATHS['foot_traffic'])
affordability_encoder = joblib.load(ENCODER_PATHS['affordability'])
competitors_encoder = joblib.load(ENCODER_PATHS['competitors'])

def get_top_recommendation(restaurant_type, business_size, budget, state):
    """
    Predicts and returns the top location recommendation based on input parameters.
    """
    try:
        # Load locations data
        logger.info("Loading locations data...")
        locations_df = pd.read_csv(LOCATIONS_CSV_PATH)

        # Filter data
        filtered_df = locations_df[
            (locations_df['state'].str.lower() == state.lower()) &
            (locations_df['budget'] <= float(budget))
        ]

        if filtered_df.empty:
            return [{"error": "No locations match the given criteria."}]

        # Prepare input features
        input_data = filtered_df.copy()
        input_data['restaurant_type_encoded'] = 1 if restaurant_type.lower() == 'traditional' else 2
        input_data['business_size_encoded'] = 1 if business_size.lower() == 'small' else 2

        features = ['restaurant_type_encoded', 'business_size_encoded', 'budget']
        input_data['score'] = model.predict(input_data[features])

        # Get top recommendation
        top_location = input_data.sort_values(by='score', ascending=False).iloc[0]
        return [
            {
                "name": top_location['location_name'],
                "lat": top_location['latitude'],
                "lng": top_location['longitude'],
                "target_audience": target_audience_encoder.inverse_transform(
                    [int(top_location['target_audience'])]
                )[0],
                "foot_traffic": foot_traffic_encoder.inverse_transform(
                    [int(top_location['foot_traffic'])]
                )[0],
                "affordability": affordability_encoder.inverse_transform(
                    [int(top_location['affordability'])]
                )[0],
                "competitors": competitors_encoder.inverse_transform(
                    [int(top_location['competitors'])]
                )[0],
                "accuracy": round(top_location['score'] * 100, 2),
            }
        ]
    except Exception as e:
        logger.error(f"Error during recommendation: {e}")
        raise RuntimeError(f"Error processing recommendation: {e}")

if __name__ == "__main__":
    try:
        recommendation = get_top_recommendation(
            restaurant_type="Traditional",
            business_size="Small",
            budget=50000,
            state="Kaduna"
        )
        print("Recommendation:", recommendation)
    except RuntimeError as err:
        print(err)
