import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def plot_training_history(history, title):
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{title} - Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.legend()

    plt.show()

# Load training histories
cnn_crf_model = tf.keras.models.load_model("cnn_crf_model.h5", compile=False)
lstm_model = tf.keras.models.load_model("lstm_model.h5", compile=False)

# Assuming history files were saved during training
cnn_crf_history = np.load("cnn_crf_history.npy", allow_pickle=True).item()
lstm_history = np.load("lstm_history.npy", allow_pickle=True).item()

# Plot training histories
plot_training_history(cnn_crf_history, "CNN-CRF Model")
plot_training_history(lstm_history, "LSTM Model")

# Compare Model Performance
def compare_performance(cnn_crf_metrics, lstm_metrics):
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    cnn_values = [cnn_crf_metrics['accuracy'], cnn_crf_metrics['precision'], cnn_crf_metrics['recall'], cnn_crf_metrics['f1_score']]
    lstm_values = [lstm_metrics['accuracy'], lstm_metrics['precision'], lstm_metrics['recall'], lstm_metrics['f1_score']]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, cnn_values, width, label='CNN-CRF', color='blue')
    plt.bar(x + width/2, lstm_values, width, label='LSTM', color='red')

    plt.xticks(ticks=x, labels=metrics)
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.legend()
    plt.show()

# Load evaluation metrics
cnn_crf_metrics = np.load("cnn_crf_metrics.npy", allow_pickle=True).item()
lstm_metrics = np.load("lstm_metrics.npy", allow_pickle=True).item()

# Plot comparison
compare_performance(cnn_crf_metrics, lstm_metrics)
