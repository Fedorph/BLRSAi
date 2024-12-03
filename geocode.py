import pandas as pd
import requests

# File paths
input_file = 'static/data/location.csv'  # Input CSV file
output_file = 'static/data/location_geocoded.csv'  # Output CSV file
api_key = 'e333e2c0874a4605bac5cc54f0ca5bbb
'  # Replace with your OpenCage API key

# Function to geocode a location
def geocode_location(location_name, api_key):
    base_url = 'https://api.opencagedata.com/geocode/v1/json'
    params = {
        'q': location_name,
        'key': api_key,
        'limit': 1,  # Get only the top result
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            lat = data['results'][0]['geometry']['lat']
            lng = data['results'][0]['geometry']['lng']
            return lat, lng
    return None, None

# Load the CSV file into a DataFrame
df = pd.read_csv(input_file)

# Initialize latitude and longitude columns
df['lat'] = None
df['lng'] = None

# Geocode each location and update the DataFrame
for index, row in df.iterrows():
    location_name = row['name']
    lat, lng = geocode_location(location_name, api_key)
    df.at[index, 'lat'] = lat
    df.at[index, 'lng'] = lng
    print(f"Processed: {location_name} -> lat: {lat}, lng: {lng}")

# Save the updated DataFrame to a new CSV file
df.to_csv(output_file, index=False)
print(f"Geocoded data saved to {output_file}")
