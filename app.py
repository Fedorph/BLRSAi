from flask import Flask, render_template
from route import recommendation_route, results_route
import os
import logging

# Initialize Flask app
app = Flask(__name__)

# Configure app
app.config.from_mapping(
    SECRET_KEY=os.getenv('FLASK_SECRET_KEY', '356dd4ea1fa9d38d0dd96d060aa54cd4'),
    DEBUG=os.getenv('FLASK_DEBUG', 'True').lower() in ('true', '1'),
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Routes
@app.route('/', methods=['GET'])
@app.route('/recommendation', methods=['GET'])
def recommendation():
    return render_template('recommendation.html')

@app.route('/results', methods=['GET'])
def results():
    return render_template('results.html')

@app.route('/health', methods=['GET'])
def health_check():
    return {"status": "OK"}, 200

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Server error: {e}")
    return render_template('500.html'), 500

# Initialize routes from route.py
app.register_blueprint(recommendation_route)
app.register_blueprint(results_route)

# Run app
if __name__ == '__main__':
    logger.info("Starting Flask application...")
    app.run(debug=app.config['DEBUG'])
