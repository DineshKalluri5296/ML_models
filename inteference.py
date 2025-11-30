from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/", methods=["GET"])
def home():
    return "Linear Regression Model API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    years = data["YearsExperience"]
    years = np.array(years).reshape(-1, 1)

    predictions = model.predict(years)
    return jsonify({"prediction": predictions.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
