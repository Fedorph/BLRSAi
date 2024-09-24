# app.py
from flask import Flask, render_template, request, redirect, url_for, jsonify
from model import recommend_locations  # Import the recommendation function

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('location_form.html')

@app.route('/process_location', methods=['POST'])
def process_location():
    # Extract form data
    company_name = request.form.get('companyName')
    income = int(request.form.get('income'))
    business_type = request.form.get('businessType')
    business_size = request.form.get('businessSize')
    budget = int(request.form.get('budget'))
    area = request.form.get('area')
    preferences = request.form.get('preferences')
    
    # Use the AI model to get location recommendations
    recommended_locations = recommend_locations(
        business_type=business_type,
        business_size=business_size,
        income=income,
        area=area,
        preferences=preferences
    )

    # For simplicity, print recommendations to console (you can log this)
    print(f"Recommended Locations for {company_name}: {recommended_locations}")

    # Render recommendations on the form page itself or a new page
    return render_template('location_form.html', recommendations=recommended_locations)

if __name__ == '__main__':
    app.run(debug=True)
