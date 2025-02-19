import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from data_loader import EEGDataLoader
from preprocessing import EEGPreprocessor
from feature_extraction import EEGFeatureExtractor
from cnn_crf_model import CNNCRFModel
from lstm_model import LSTMModel
from pfoa_optimizer import PufferFishOptimization

# Load and preprocess data
data_loader = EEGDataLoader(dataset_name="Sleep-EDF", sampling_rate=100)
raw_data, raw_labels = data_loader.prepare_data("data.csv")

# Preprocess EEG signals
preprocessor = EEGPreprocessor(sampling_rate=100)
processed_data = preprocessor.preprocess(raw_data)

# Extract features
feature_extractor = EEGFeatureExtractor(sampling_rate=100)
features = feature_extractor.extract_features(processed_data)

# One-hot encode labels
num_classes = 5  # Sleep stages: W, N1, N2, N3, REM
labels = to_categorical(raw_labels, num_classes=num_classes)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Hyperparameter tuning using PFOA
optimizer = PufferFishOptimization(population_size=10, num_generations=5)
best_hyperparams = optimizer.optimize()

# Train CNN-CRF Model
cnn_crf_model = CNNCRFModel(input_shape=(X_train.shape[1], X_train.shape[2]), num_classes=num_classes)
cnn_crf = cnn_crf_model.build_model()
cnn_crf.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=int(best_hyperparams['batch_size']))

# Save CNN-CRF model
cnn_crf.save("cnn_crf_model.h5")

# Train LSTM Model
lstm_model = LSTMModel(input_shape=(X_train.shape[1], X_train.shape[2]), num_classes=num_classes,
                        learning_rate=best_hyperparams['learning_rate'], dropout_rate=best_hyperparams['dropout_rate'])
lstm = lstm_model.build_model()
lstm.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=int(best_hyperparams['batch_size']))

# Save LSTM model
lstm.save("lstm_model.h5")

print("Training complete. Models saved successfully!")