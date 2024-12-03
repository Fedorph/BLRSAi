import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Load the location dataset
location_data_path = 'static/data/location.csv'

# Load the location data into a DataFrame
locations_df = pd.read_csv(location_data_path)

# Normalize the data (important for distance-based models)
scaler = StandardScaler()
locations_df[['budget', 'foot_traffic', 'affordability', 'competitors']] = scaler.fit_transform(
    locations_df[['budget', 'foot_traffic', 'affordability', 'competitors']]
)

# Helper function to get recommendations based on user input
def fetch_recommendation(company_name, restaurant_type, business_size, budget, target_audience, foot_traffic, affordability, competitors):
    # Create a real-time user input row based on the input
    user_input = pd.DataFrame([{
        'company_name': company_name,
        'restaurant_type': restaurant_type,
        'business_size': business_size,
        'budget': budget,
        'target_audience': target_audience,
        'foot_traffic': foot_traffic,
        'affordability': affordability,
        'competitors': competitors
    }])

    # Normalize the user input to match the location data
    user_input[['budget', 'foot_traffic', 'affordability', 'competitors']] = scaler.transform(
        user_input[['budget', 'foot_traffic', 'affordability', 'competitors']]
    )

    # Calculate cosine similarity between the user's input and all locations
    similarity_scores = cosine_similarity(user_input[['budget', 'foot_traffic', 'affordability', 'competitors']], locations_df[['budget', 'foot_traffic', 'affordability', 'competitors']])

    # Get the most similar location based on cosine similarity
    most_similar_index = np.argmax(similarity_scores)
    most_similar_location = locations_df.iloc[most_similar_index]

    # Create a recommendation dictionary
    recommendation = {
        'location_name': most_similar_location['name'],
        'target_audience': most_similar_location['target_audience'],
        'foot_traffic': most_similar_location['foot_traffic'],
        'affordability': most_similar_location['affordability'],
        'competitors': most_similar_location['competitors'],
        'accuracy': similarity_scores[0][most_similar_index] * 100  # Accuracy as percentage
    }

    return recommendation

# Function to store user details and recommendation to a history file
def store_user_history(user_data, recommendation):
    # Add timestamp to the user data
    user_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Prepare the user history data
    user_data['location_recommended'] = recommendation['location_name']
    user_data['recommendation_accuracy'] = recommendation['accuracy']
    
    # Create a unique file name for each user's session to store their data
    session_file_name = f"static/data/session_{user_data['company_name']}_{user_data['timestamp'].replace(' ', '_').replace(':', '-')}.csv"

    # Save the user interaction history to a CSV file
    user_history = pd.DataFrame([user_data])

    user_history.to_csv(session_file_name, mode='w', header=True, index=False)

    # Also append the user history to a master log file for all users
    history_path = 'static/data/user_history.csv'
    history_exists = os.path.exists(history_path)
    
    if history_exists:
        user_history.to_csv(history_path, mode='a', header=False, index=False)
    else:
        user_history.to_csv(history_path, mode='w', header=True, index=False)

    return user_history

# Function to fetch user history
def get_user_history():
    history_path = 'static/data/user_history.csv'
    if os.path.exists(history_path):
        return pd.read_csv(history_path)
    else:
        return pd.DataFrame()

# Function to save session details
def save_session_data(user_data):
    # Store session data in a CSV file
    session_file_path = 'static/data/session.csv'
    session_exists = os.path.exists(session_file_path)

    session_data = pd.DataFrame([user_data])
    
    if session_exists:
        session_data.to_csv(session_file_path, mode='a', header=False, index=False)
    else:
        session_data.to_csv(session_file_path, mode='w', header=True, index=False)

    return session_data
