@app.route('/recommend', methods=['POST'])
def recommend():
    company_name = request.form.get('company_name')
    restaurant_type = request.form.get('restaurant_type')
    business_size = request.form.get('business_size')
    location_budget = float(request.form.get('location_budget'))
    state = request.form.get('state')

    # Get the recommendations from the model
    recommendations = get_recommendations(restaurant_type, business_size, location_budget, state)

    return render_template('results.html', recommendations=recommendations)
