import numpy as np
import scipy.stats as stats
import scipy.signal as signal
import pywt

class EEGFeatureExtractor:
    def __init__(self, sampling_rate=100):
        self.sampling_rate = sampling_rate

    def time_domain_features(self, data):
        mean = np.mean(data, axis=1)
        std_dev = np.std(data, axis=1)
        skewness = stats.skew(data, axis=1)
        kurtosis = stats.kurtosis(data, axis=1)
        return np.column_stack((mean, std_dev, skewness, kurtosis))

    def frequency_domain_features(self, data):
        psd_features = []
        for sample in data:
            freqs, psd = signal.welch(sample, fs=self.sampling_rate, nperseg=256)
            psd_features.append(psd[:50])  # Taking first 50 frequency bins
        return np.array(psd_features)

    def wavelet_features(self, data, wavelet='db4', level=4):
        wavelet_energy = []
        for sample in data:
            coeffs = pywt.wavedec(sample, wavelet, level=level)
            energy = [np.sum(np.square(c)) for c in coeffs]  # Compute energy of coefficients
            wavelet_energy.append(energy)
        return np.array(wavelet_energy)

    def extract_features(self, data):
        time_features = self.time_domain_features(data)
        freq_features = self.frequency_domain_features(data)
        wavelet_features = self.wavelet_features(data)

        # Combine all features
        combined_features = np.hstack((time_features, freq_features, wavelet_features))
        return combined_features

if __name__ == "__main__":
    dummy_data = np.random.randn(10, 3000)  # Simulated EEG signals (10 samples, 3000 time points)
    extractor = EEGFeatureExtractor(sampling_rate=100)
    extracted_features = extractor.extract_features(dummy_data)

    print("Extracted Features Shape:", extracted_features.shape)
