from flask import session

def save_user_inputs(company_name, restaurant_type, business_size, budget, state):
    """Save user input to session."""
    session['company_name'] = company_name
    session['restaurant_type'] = restaurant_type
    session['business_size'] = business_size
    session['budget'] = budget
    session['state'] = state

def get_user_inputs():
    """Retrieve user inputs from session."""
    return {
        'company_name': session.get('company_name'),
        'restaurant_type': session.get('restaurant_type'),
        'business_size': session.get('business_size'),
        'budget': session.get('budget'),
        'state': session.get('state')
    }

def save_recommendations(recommendations):
    """Save recommendations to session."""
    # Ensure the recommendations are serialized properly before saving if needed
    session['recommendations'] = recommendations

def get_recommendations_from_session():
    """Retrieve recommendations from session."""
    return session.get('recommendations')
