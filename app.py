from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__, template_folder="app/templates", static_folder="app/static")

# Load model and scaler with error handling
try:
    model = joblib.load("models/model.joblib")
    scaler = joblib.load("models/scaler.joblib")
except FileNotFoundError:
    model = None
    scaler = None
    print("Warning: Model files not found. Train the model first.")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None or scaler is None:
            return "Error: Model not loaded. Please train the model first."
        
        values = [float(x) for x in request.form.values()]
        values = np.array(values).reshape(1, -1)
        values = scaler.transform(values)
        prediction = model.predict(values)[0]
        
        result = 1 if prediction == 1 else 0
        return render_template("result.html", result=result)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
