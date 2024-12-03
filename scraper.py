import requests
import json
import pandas as pd
import os
import osmnx as ox

# Function to fetch data from the Overpass API
def fetch_osm_data(query, overpass_url="http://overpass-api.de/api/interpreter"):
    response = requests.get(overpass_url, params={'data': query})
    if response.status_code == 200:
        return response.json()  # Parse the response as JSON
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

# Function to parse and collect data from the JSON response
def parse_osm_data(data):
    if not data:
        print("No data to parse.")
        return []

    restaurants = []
    elements = data.get('elements', [])
    for element in elements:
        if element.get('type') == 'node':
            name = element.get('tags', {}).get('name', 'Unnamed')
            restaurant_type = element.get('tags', {}).get('cuisine', 'Unknown')
            lat = element.get('lat')
            lon = element.get('lon')

            # Simulated data for other attributes
            business_size = 'small' if 'small' in name.lower() else 'medium'
            location_budget = 200000  # Example budget, adjust as needed
            state = 'Kaduna'
            target_audience = 'locals and travelers'
            foot_traffic = 'high'  # Simulated value
            affordability = 'affordable'  # Simulated value
            competitors = 'various nearby restaurants'  # Simulated value

            restaurants.append({
                'name': name,
                'restaurant_type': restaurant_type,
                'business_size': business_size,
                'location_budget': location_budget,
                'state': state,
                'target_audience': target_audience,
                'foot_traffic': foot_traffic,
                'affordability': affordability,
                'competitors': competitors,
                'latitude': lat,
                'longitude': lon
            })
    return restaurants

# Define an Overpass API query for restaurants in Kaduna
query = """
    [out:json];
    node
        (around:5000, 10.5200, 7.4500)  // Coordinates for central Kaduna (adjust as needed)
        ["amenity"~"restaurant|cafe"];
    out body;
"""

# Fetch and parse the data
osm_data = fetch_osm_data(query)
restaurant_data = parse_osm_data(osm_data)

# Create a DataFrame from the collected data
df = pd.DataFrame(restaurant_data)

# Save to CSV and JSON in the 'static/data/' directory
output_dir = 'static/data/'
os.makedirs(output_dir, exist_ok=True)

csv_path = os.path.join(output_dir, 'restaurants_locations_data.csv')
json_path = os.path.join(output_dir, 'restaurants_locations_data.json')

# Save CSV
df.to_csv(csv_path, index=False)

# Save JSON
df.to_json(json_path, orient='records', lines=True)

print(f"Data saved to {csv_path} and {json_path}")
