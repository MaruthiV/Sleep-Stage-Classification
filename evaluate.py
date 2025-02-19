import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from data_loader import EEGDataLoader
from preprocessing import EEGPreprocessor
from feature_extraction import EEGFeatureExtractor

# Load test dataset
data_loader = EEGDataLoader(dataset_name="Sleep-EDF", sampling_rate=100)
raw_data, raw_labels = data_loader.prepare_data("path_to_test_dataset")

# Preprocess EEG signals
preprocessor = EEGPreprocessor(sampling_rate=100)
processed_data = preprocessor.preprocess(raw_data)

# Extract features
feature_extractor = EEGFeatureExtractor(sampling_rate=100)
features = feature_extractor.extract_features(processed_data)

# Load trained models
cnn_crf_model = tf.keras.models.load_model("cnn_crf_model.h5", compile=False)
lstm_model = tf.keras.models.load_model("lstm_model.h5", compile=False)

# Make predictions
cnn_predictions = np.argmax(cnn_crf_model.predict(features), axis=-1)
lstm_predictions = np.argmax(lstm_model.predict(features), axis=-1)

# Convert true labels to categorical format
true_labels = np.argmax(raw_labels, axis=-1)

# Evaluate CNN-CRF Model
cnn_accuracy = accuracy_score(true_labels, cnn_predictions)
cnn_report = classification_report(true_labels, cnn_predictions, target_names=["W", "N1", "N2", "N3", "REM"])
cnn_conf_matrix = confusion_matrix(true_labels, cnn_predictions)

# Evaluate LSTM Model
lstm_accuracy = accuracy_score(true_labels, lstm_predictions)
lstm_report = classification_report(true_labels, lstm_predictions, target_names=["W", "N1", "N2", "N3", "REM"])
lstm_conf_matrix = confusion_matrix(true_labels, lstm_predictions)

# Print evaluation metrics
print("\n=== CNN-CRF Model Evaluation ===")
print(f"Accuracy: {cnn_accuracy:.4f}")
print(cnn_report)

print("\n=== LSTM Model Evaluation ===")
print(f"Accuracy: {lstm_accuracy:.4f}")
print(lstm_report)

# Plot confusion matrices
def plot_confusion_matrix(matrix, title):
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["W", "N1", "N2", "N3", "REM"], 
                yticklabels=["W", "N1", "N2", "N3", "REM"])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(title)
    plt.show()

plot_confusion_matrix(cnn_conf_matrix, "CNN-CRF Confusion Matrix")
plot_confusion_matrix(lstm_conf_matrix, "LSTM Confusion Matrix")
