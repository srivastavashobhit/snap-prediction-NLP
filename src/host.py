from flask import Flask, request, jsonify
import pickle
import os

from utils.lstm_utils import get_most_recent_model_path

app = Flask(__name__)

@app.before_first_request
def load_model():
    global model
    
    model_path = get_most_recent_model_path()

    with open(model_path, "rb") as model_file:
        # Load the model from the file
        model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    input_data = request.json

    # TODO: Process the input data and run it through your deep learning model
    # ...
    # Set the path of the directory containing the subdirectories
    
    # print(input_data)
    input_seq = input_data['input']
    
    if 'top_n' in input_data:
        top_n = input_data['top_n']
    else:
        top_n = 1

    input_sequences = [input_seq]

    prediction = model.predict_all_top_n(input_sequences,top_n)

    # Return the prediction as a JSON object
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)