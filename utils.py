import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def save_training_history(history, filename):
    history_dict = history.history
    np.save(filename, history_dict)

def load_training_history(filename):
    return np.load(filename, allow_pickle=True).item()

def save_model_checkpoint(model, filename):
    model.save(filename)

def load_model_checkpoint(filename):
    return tf.keras.models.load_model(filename, compile=False)

def log_message(message, log_file="training.log"):
    with open(log_file, "a") as f:
        f.write(message + "\n")
    print(message)

def plot_eeg_signal(eeg_data, sample_index=0, title="EEG Signal"):
    plt.figure(figsize=(10, 4))
    plt.plot(eeg_data[sample_index])
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    dummy_history = {"loss": [0.5, 0.4, 0.3], "accuracy": [0.75, 0.85, 0.92]}
    save_training_history(dummy_history, "test_history.npy")

    loaded_history = load_training_history("test_history.npy")
    print("Loaded History:", loaded_history)

    log_message("Training completed successfully!")
