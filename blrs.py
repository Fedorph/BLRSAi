# model.py
from flask import Flask, render_template, request, jsonify
import random
import pandas as pd
import os
import joblib
import numpy as np
  # Import function to get location data

app = Flask(__name__)

# Define the path to the data file and model files
locations_file_path = os.path.join('static', 'data', 'location.csv')
model_path = os.path.join('static', 'models', 'location_recommender_model.pkl')
target_audience_encoder_path = os.path.join('static', 'models', 'target_audience_encoder.pkl')
foot_traffic_encoder_path = os.path.join('static', 'models', 'foot_traffic_encoder.pkl')
affordability_encoder_path = os.path.join('static', 'models', 'affordability_encoder.pkl')
competitors_encoder_path = os.path.join('static', 'models', 'competitors_encoder.pkl')

# Load the trained model and label encoders
model = joblib.load(model_path)
target_audience_encoder = joblib.load(target_audience_encoder_path)
foot_traffic_encoder = joblib.load(foot_traffic_encoder_path)
affordability_encoder = joblib.load(affordability_encoder_path)
competitors_encoder = joblib.load(competitors_encoder_path)

def get_recommendations(restaurant_type, business_size, budget, state):
    """
    Function to predict and return location recommendations based on input parameters using machine learning.
    """
    # Load the location data
    locations_df = pd.read_csv(locations_file_path)

    # Filter the dataset based on restaurant type, business size, budget, and state
    recommended_locations = locations_df[locations_df['type'] == restaurant_type]
    
    if business_size:
        recommended_locations = recommended_locations[recommended_locations['business_size'] == business_size]
    
    if budget:
        recommended_locations = recommended_locations[recommended_locations['budget'] <= budget]
    
    if state:
        recommended_locations = recommended_locations[recommended_locations['state'] == state]

    # Prepare input features for prediction: encode categorical features
    recommendations = []
    
    for _, location in recommended_locations.iterrows():
        # Extract the features for the current location
        lat = location['lat']
        lng = location['lng']
        affordability = location['affordability']
        target_audience = location['target_audience']
        competitors = location['competitors']

        # Encode the categorical features
        affordability_encoded = affordability_encoder.transform([affordability])[0]
        target_audience_encoded = target_audience_encoder.transform([target_audience])[0]
        competitors_encoded = competitors_encoder.transform([competitors])[0]

        # Prepare the input data for prediction
        input_data = np.array([[affordability_encoded, target_audience_encoded, competitors_encoded]])
        
        # Make the prediction using the trained model
        foot_traffic_pred = model.predict(input_data)
        
        # Decode the prediction to get the original category
        foot_traffic = foot_traffic_encoder.inverse_transform(foot_traffic_pred)[0]
        
        # Add the prediction to the location data
        recommendations.append({
            'location': location['location'],  # assuming there's a 'location' column
            'foot_traffic': foot_traffic,
            'affordability': affordability,
            'target_audience': target_audience,
            'competitors': competitors
        })

    # Sort the recommendations by predicted foot traffic (you can adjust this logic as needed)
    sorted_recommendations = sorted(recommendations, key=lambda x: x['foot_traffic'], reverse=True)

    # Return the top 5 recommended locations
    return sorted_recommendations[:5]

# Real Estate Agencies Data
real_estate_agencies = {
    "Ahmadu Bello Way": [
        {"name": "Kaduna Property Consultants", "phone": "+2348012345678", "email": "info@kaduna-property.com"},
        {"name": "Prime Real Estates", "phone": "+2348098765432", "email": "contact@prime-realestate.com"}
    ],
    "Sultan Road": [
        {"name": "Sultan Realty", "phone": "+2347012345678", "email": "sales@sultanrealty.com"},
        {"name": "City Estates", "phone": "+2348098765432", "email": "info@cityestates.com"}
    ],
    "Rigachikun": [
        {"name": "Residential Property Group", "phone": "+2348034567890", "email": "admin@residentialgroup.com"}
    ]
}

# Route for the homepage
@app.route('/')
def index():
    return render_template('recommendation.html')

# Situational Intelligence (SI) Model for recommending business locations
def recommend_locations(restaurant_type, business_size, budget, state):
    # Filter location data based on the specified business type (restaurant_type)
    recommended_locations = location_df[location_df['business_category'] == restaurant_type]

    # Adjust recommendations based on business size and budget
    if business_size == "large" and budget > 5000000:
        recommended_locations = recommended_locations[~recommended_locations['name'].isin(["Rigachikun", "Murtala Square Stadium"])]
    elif business_size == "medium" and budget == 5000000:
        recommended_locations = recommended_locations[recommended_locations['name'].isin(["Sabon Gari", "Ungwan Rimi"])]
    elif business_size == "small" and budget < 5000000:
        recommended_locations = recommended_locations[recommended_locations['name'].isin(["Ahmadu Bello Way", "Tafawa Balewa Way"])]

    # Shuffle recommendation and limit to top 1 (or more if desired)
    final_recommendations = recommended_locations.sample(n=1).to_dict('records')

    # Generate advisory message based on business and budget analysis
    advice = analyze_and_advise(restaurant_type, business_size, budget, state)

    return final_recommendations, advice

def analyze_and_advise(restaurant_type, business_size, budget, state):
    """
    Provide tailored business advice based on the business size, budget, and area.
    Also, suggest real estate agencies to be contacted based on the area.
    """
    advice = []

    # Analysis based on business size and budget
    if business_size == "large":
        if budget > 5000000:
            advice.append("As a large business with a high budget, consider premium locations for greater visibility.")
        else:
            advice.append("Consider balancing visibility with cost, as high-traffic areas might provide great ROI.")
    elif business_size == "medium":
        advice.append("As a medium business, consider promoting your business within the chosen location.")
    elif business_size == "small":
        advice.append("As a small business, focus on affordable areas to manage costs.")

    # General area-based advice
    if recommend_locations == "Ahmadu Bello Way":
        advice.append("Ahmadu Bello Way is a bustling area—ideal for fast food and high-traffic businesses.")
    elif recommend_locations == "Independence Way":
        advice.append("Independence is known for its premium clientele. A good fit for fine dining and upscale restaurants.")
    elif recommend_locations == "Ungwan Rimi":
        advice.append("Ungwan Rimi is more residential and quieter, suitable for family-style or ethnic restaurants.")

    # Include real estate agency contact details based on the area
    agencies = real_estate_agencies.get(recommend_locations, [])
    if agencies:
        advice.append("\nReal Estate Agencies for your recommended location:")
        for agency in agencies:
            advice.append(f"- {agency['name']} (Phone: {agency['phone']}, Email: {agency['email']})")

    return advice

# Endpoint to handle form submission and provide recommendations and advice
@app.route('/process_location', methods=['POST'])
def process_location():
    # Extract form data
    company_name = request.form.get('companyName')
    budget = int(request.form.get('budget'))
    restaurant_type = request.form.get('restaurantType')
    business_size = request.form.get('businessSize')
    state = request.form.get('state')  # Updated field

    # Use the AI model to get location recommendations and advice
    recommended_locations, advice = recommend_locations(
        restaurant_type=restaurant_type,
        business_size=business_size,
        budget=budget,
        state=state,  # Updated argument
    )

    # Render recommendations and advice on the results page (updated template name)
    return render_template('results.html', 
                           recommendations=recommended_locations, 
                           advice=advice, 
                           company_name=company_name)

# JSON API for recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    restaurant_type = data.get('restaurant_type')
    business_size = data.get('business_size')
    budget = data.get('budget')
    state = data.get('state')  # Updated field

    # Validate inputs
    if not all([restaurant_type, business_size, budget, state]):
        return jsonify({"error": "Missing required fields."}), 400

    recommendations, advice = recommend_locations(restaurant_type, business_size, budget, state)
    return jsonify({"recommendations": recommendations, "advice": advice})

if __name__ == '__main__':
    app.run(debug=True)