from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from data_loader import EEGDataLoader
from preprocessing import EEGPreprocessor
from feature_extraction import EEGFeatureExtractor
from utils import load_model_checkpoint

app = Flask(__name__)

# Load trained model 
MODEL_PATH = "cnn_crf_model.h5"
model = load_model_checkpoint(MODEL_PATH)

def predict_sleep_stage(eeg_data):
    preprocessor = EEGPreprocessor(sampling_rate=100)
    processed_data = preprocessor.preprocess(eeg_data)

    feature_extractor = EEGFeatureExtractor(sampling_rate=100)
    features = feature_extractor.extract_features(processed_data)

    predictions = np.argmax(model.predict(features), axis=-1)

    sleep_stages = ["Wake", "N1", "N2", "N3", "REM"]
    predicted_stages = [sleep_stages[p] for p in predictions]

    return predicted_stages

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    if "eeg_data" not in data:
        return jsonify({"error": "Missing EEG data"}), 400

    eeg_data = np.array(data["eeg_data"])  # Convert JSON input to NumPy array

    predicted_stages = predict_sleep_stage(eeg_data)
    
    return jsonify({"predicted_stages": predicted_stages})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
