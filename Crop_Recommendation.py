from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

def train_model(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Split the data into features (X) and labels (y)
    X = df[['N', 'P', 'K', 'humidity']]
    y = df['label']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Set feature names for RandomForestClassifier
    feature_names = ['N', 'P', 'K', 'humidity']
    X_train.columns = feature_names

    # Initialize the model (Random Forest Classifier in this case)
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    return model

def predict_crop(model, value_N, value_P, value_K, value_humidity):
    # Validate input values
    if not (0 <= value_N <= 100) or not (0 <= value_P <= 100) or not (0 <= value_K <= 100) or not (0 <= value_humidity <= 100):
        return {'predicted_crop': "Invalid input. Percentage values must be between 0 and 100."}

    # Use a 2D NumPy array directly for the new input
    new_input = [[value_N, value_P, value_K, value_humidity]]

    predicted_crop = model.predict(new_input)

    return {'predicted_crop': predicted_crop[0]}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    try:
        value_N = int(request.args.get('N'))
        value_P = int(request.args.get('P'))
        value_K = int(request.args.get('K'))
        value_humidity = int(request.args.get('humidity'))
    except ValueError:
        return jsonify({'error': 'Invalid input. Please enter valid integer values for N, P, K, and humidity.'})

    # Load the trained model
    file_path = "Crop_recommendation.csv"
    trained_model = train_model(file_path)

    # Predict the crop
    result = predict_crop(trained_model, value_N, value_P, value_K, value_humidity)

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
