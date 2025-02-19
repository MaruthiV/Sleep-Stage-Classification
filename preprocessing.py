import numpy as np
import scipy.signal as signal
import pywt
from sklearn.preprocessing import MinMaxScaler

class EEGPreprocessor:
    def __init__(self, sampling_rate=100):
        self.sampling_rate = sampling_rate
        self.scaler = MinMaxScaler(feature_range=(0, 1))  # Normalize between 0 and 1

    def bandpass_filter(self, data, low_freq=0.5, high_freq=40):
        nyquist = 0.5 * self.sampling_rate
        low = low_freq / nyquist
        high = high_freq / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_data = signal.filtfilt(b, a, data, axis=1)
        return filtered_data

    def apply_wavelet_transform(self, data, wavelet='db4', level=4):
        transformed_data = []
        for sample in data:
            coeffs, _ = pywt.cwt(sample, scales=np.arange(1, level + 1), wavelet=wavelet)
            transformed_data.append(coeffs.flatten())  # Flatten for model input
        return np.array(transformed_data)

    def normalize_features(self, data):
        reshaped_data = data.reshape(-1, 1)
        normalized_data = self.scaler.fit_transform(reshaped_data).reshape(data.shape)
        return normalized_data

    def preprocess(self, data):
        filtered_data = self.bandpass_filter(data)
        wavelet_features = self.apply_wavelet_transform(filtered_data)
        normalized_data = self.normalize_features(wavelet_features)
        return normalized_data

if __name__ == "__main__":
    dummy_data = np.random.randn(10, 3000)  # Simulated EEG signals (10 samples, 3000 time points)
    preprocessor = EEGPreprocessor(sampling_rate=100)
    processed_data = preprocessor.preprocess(dummy_data)

    print("Processed Data Shape:", processed_data.shape)
