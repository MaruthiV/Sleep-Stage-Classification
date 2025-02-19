import numpy as np
import pandas as pd
import os
import glob
import mne
from scipy.signal import resample

class EEGDataLoader:
    def __init__(self, dataset_name="Sleep-EDF", sampling_rate=100):
        self.dataset_name = dataset_name
        self.sampling_rate = sampling_rate

    def load_sleep_edf(self, data_dir):
        files = glob.glob(os.path.join(data_dir, "*.edf"))
        data = []
        labels = []

        for file in files:
            raw = mne.io.read_raw_edf(file, preload=True)
            eeg_data = raw.get_data()
            annotations = raw.annotations

            # Extract annotations (sleep stage labels)
            label_array = np.zeros(len(eeg_data[0]))  # Placeholder labels
            for i, annot in enumerate(annotations):
                onset = int(annot["onset"] * self.sampling_rate)
                duration = int(annot["duration"] * self.sampling_rate)
                label = annot["description"]

                # Map annotations to sleep stages
                stage = self.map_sleep_stage(label)
                label_array[onset:onset + duration] = stage
            
            # Resampling to uniform sampling rate
            eeg_data = resample(eeg_data, num=len(label_array), axis=1)
            data.append(eeg_data)
            labels.append(label_array)

        return np.array(data), np.array(labels)

    def load_shhs(self, data_dir):
        files = glob.glob(os.path.join(data_dir, "*.edf"))
        data = []
        labels = []

        for file in files:
            raw = mne.io.read_raw_edf(file, preload=True)
            eeg_data = raw.get_data()
            annotations = raw.annotations

            label_array = np.zeros(len(eeg_data[0]))  # Placeholder labels
            for i, annot in enumerate(annotations):
                onset = int(annot["onset"] * self.sampling_rate)
                duration = int(annot["duration"] * self.sampling_rate)
                label = annot["description"]

                stage = self.map_sleep_stage(label)
                label_array[onset:onset + duration] = stage

            eeg_data = resample(eeg_data, num=len(label_array), axis=1)
            data.append(eeg_data)
            labels.append(label_array)

        return np.array(data), np.array(labels)

    def map_sleep_stage(self, label):
        mapping = {
            "W": 0,  # Wake
            "N1": 1,  # Light Sleep
            "N2": 2,  # Deeper Sleep
            "N3": 3,  # Deep Sleep
            "REM": 4  # REM Sleep
        }
        return mapping.get(label, -1)  # Return -1 if the label is unknown

    def prepare_data(self, data_dir):
        if self.dataset_name.lower() == "sleep-edf":
            return self.load_sleep_edf(data_dir)
        elif self.dataset_name.lower() == "shhs":
            return self.load_shhs(data_dir)
        else:
            raise ValueError("Unsupported dataset name. Choose 'Sleep-EDF' or 'SHHS'.")

if __name__ == "__main__":
    data_loader = EEGDataLoader(dataset_name="Sleep-EDF", sampling_rate=100)
    data, labels = data_loader.prepare_data("path_to_your_dataset")
    print("Data Shape:", data.shape)
    print("Labels Shape:", labels.shape)
