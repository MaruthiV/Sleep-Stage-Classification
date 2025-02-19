# Configuration file for EEG Sleep Stage Classification

CONFIG = {
    # Data Settings
    "dataset_name": "Sleep-EDF",  # Options: "Sleep-EDF", "SHHS"
    "sampling_rate": 100,  # Hz

    # Preprocessing Settings
    "low_freq": 0.5,  # Bandpass filter lower bound
    "high_freq": 40,  # Bandpass filter upper bound
    "wavelet": "db4",  # Wavelet for feature extraction
    "wavelet_levels": 4,

    # Model Hyperparameters (Tuned using PFOA)
    "learning_rate": 0.0005,
    "batch_size": 64,
    "dropout_rate": 0.3,
    "lstm_units": 64,

    # Training Settings
    "epochs": 10,
    "validation_split": 0.2,

    # File Paths
    "data_path": "path_to_dataset",
    "test_data_path": "path_to_test_dataset",
    "cnn_crf_model_path": "cnn_crf_model.h5",
    "lstm_model_path": "lstm_model.h5",
    "history_save_path": "training_history/",
    "metrics_save_path": "evaluation_metrics/"
}
