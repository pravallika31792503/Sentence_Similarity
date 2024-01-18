# app.py
import pickle
from flask import Flask, request, jsonify, render_template
from similarity_functions import calculate_normalized_similarity

app = Flask(__name__)

# Load the pickled function
with open('normalized_similarity_function.pkl', 'rb') as file:
    loaded_similarity_function = pickle.load(file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_similarity', methods=['POST'])
def predict_similarity():
    try:
        # Get data from the request body
        data = request.form.to_dict()

        # Extract text1 and text2 from the request data
        text1 = data.get('text1', '')
        text2 = data.get('text2', '')

        # Use the loaded function to calculate normalized similarity score
        normalized_similarity_score = loaded_similarity_function(text1, text2)

        return render_template('home.html', prediction_text=f"Normalized similarity score: {normalized_similarity_score}")

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
