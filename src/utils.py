import joblib
import os

def save_model(model, scaler, model_dir):
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(model, os.path.join(model_dir, "model.joblib"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))

    print("Model & scaler saved successfully!")
