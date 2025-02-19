import numpy as np
import tensorflow as tf
from data_loader import EEGDataLoader
from preprocessing import EEGPreprocessor
from feature_extraction import EEGFeatureExtractor
from utils import load_model_checkpoint

# Load training model
MODEL_PATH = "cnn_crf_model.h5"  
model = load_model_checkpoint(MODEL_PATH)

def predict_sleep_stage(eeg_data):
    # Preprocess EEG data
    preprocessor = EEGPreprocessor(sampling_rate=100)
    processed_data = preprocessor.preprocess(eeg_data)

    # Extract features
    feature_extractor = EEGFeatureExtractor(sampling_rate=100)
    features = feature_extractor.extract_features(processed_data)

    # Make predictions
    predictions = np.argmax(model.predict(features), axis=-1)

    # Sleep stage mapping
    sleep_stages = ["Wake", "N1", "N2", "N3", "REM"]
    predicted_stages = [sleep_stages[p] for p in predictions]

    return predicted_stages

if __name__ == "__main__":
    # Simulate a single EEG sample (300 time points, 100 features)
    dummy_eeg_data = np.random.randn(1, 3000)

    predicted_stages = predict_sleep_stage(dummy_eeg_data)
    print("Predicted Sleep Stages:", predicted_stages)
