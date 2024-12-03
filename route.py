from flask import Blueprint, request, render_template
from model import get_top_recommendation
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

recommendation_route = Blueprint('recommendation_route', __name__)
results_route = Blueprint('results_route', __name__)

@recommendation_route.route('/recommendation', methods=['GET'])
def recommendation_page():
    return render_template('recommendation.html')

@results_route.route('/results', methods=['POST'])
def results_page():
    form_data = request.form

    # Validate required fields
    required_fields = ['company_name', 'restaurant_type', 'business_size', 'budget', 'state']
    missing_fields = [field for field in required_fields if field not in form_data or not form_data[field].strip()]

    if missing_fields:
        logger.warning(f"Missing fields: {missing_fields}")
        # Return with missing fields error and the form data
        return render_template(
            'recommendation.html',
            form_data=form_data,
            error=f"Missing required fields: {', '.join(missing_fields)}"
        )

    # Extract form data
    company_name = form_data['company_name']
    restaurant_type = form_data['restaurant_type']
    business_size = form_data['business_size']
    budget = form_data['budget']
    state = form_data['state']

    try:
        # Get recommendations
        recommendations = get_top_recommendation(
            restaurant_type=restaurant_type,
            business_size=business_size,
            budget=budget,
            state=state,
        )
        logger.info(f"Recommendations generated: {recommendations}")

        # Handle no recommendations
        if not recommendations or "error" in recommendations[0]:
            error_message = recommendations[0].get("error", "No recommendations available.")
            return render_template(
                'results.html',
                company_name=company_name,
                restaurant_type=restaurant_type,
                business_size=business_size,
                budget=budget,
                state=state,
                error=error_message,
                recommendations=[]
            )

        # Render recommendations
        return render_template(
            'results.html',
            company_name=company_name,
            restaurant_type=restaurant_type,
            business_size=business_size,
            budget=budget,
            state=state,
            recommendations=recommendations
        )

    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return render_template(
            'results.html',
            error="An unexpected error occurred. Please try again later.",
            recommendations=[]
        )
