#app.py

from flask import Flask, render_template
from route import recommendation_route, results_route
import os

app = Flask(__name__)

# Set the secret key for session handling
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', '356dd4ea1fa9d38d0dd96d060aa54cd4')

# Route to serve recommendation.html at the root URL
@app.route('/')
@app.route('/recommendation')
def recommendation():
    return render_template('recommendation.html')

# Route to serve results.html
@app.route('/results')
def results():
    return render_template('results.html')

# Initialize routes from route.py
app.register_blueprint(recommendation_route)
app.register_blueprint(results_route)

if __name__ == '__main__':
    app.run(debug=True)
