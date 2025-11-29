import joblib
import numpy as np

def load_artifacts(model_dir):
    model = joblib.load(f"{model_dir}/model.pkl")
    scaler = joblib.load(f"{model_dir}/scaler.pkl")
    return model, scaler

def make_prediction(data, model_dir="saved_models"):
    model, scaler = load_artifacts(model_dir)

    data = np.array(data).reshape(1, -1)
    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)[0]
    return prediction
