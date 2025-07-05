from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

# Set the template folder
app.template_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/api/data', methods=['POST'])
def get_data():
    data = request.get_json()
    # Process data here
    return jsonify({'received': data})

if __name__ == '__main__':
    app.run(debug=True)