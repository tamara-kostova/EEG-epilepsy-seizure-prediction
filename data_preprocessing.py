import os
import mne
import numpy as np
import pandas as pd
from scipy import stats
import pyeeg
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import logging


@dataclass
class EEGFeatures:
    temporal: Dict[str, float]
    spectral: Dict[str, float]
    nonlinear: Dict[str, float]


class SignalProcessor:
    def __init__(self, sampling_rate: int = 256):
        self.fs = sampling_rate
        self.freq_bands = [1, 5, 10, 15, 20, 25]

    def extract_temporal_features(self, signal: np.ndarray) -> Dict[str, float]:
        return {
            "rms": np.sqrt(np.mean(np.square(signal))),
            "variance": np.var(signal),
            "kurtosis": stats.kurtosis(signal),
            "skewness": stats.skew(signal),
            "peak_amp": np.max(np.abs(signal)),
            "peak_count": len(signal.signal.argrelextrema(signal, np.greater)[0]),
            "zero_crossings": np.sum(np.diff(np.signbit(signal).astype(int)) != 0),
        }

    def extract_spectral_features(self, signal: np.ndarray) -> Dict[str, float]:
        freqs, psd = signal.welch(signal, fs=self.fs)
        power, power_ratio = pyeeg.bin_power(signal, self.freq_bands, self.fs)

        return {
            "total_power": np.sum(psd),
            "median_freq": freqs[np.where(np.cumsum(psd) >= np.sum(psd) / 2)[0][0]],
            "peak_freq": freqs[np.argmax(psd)],
            "spectral_entropy": pyeeg.spectral_entropy(
                signal, self.freq_bands, self.fs
            ),
            **{
                f"band_power_{band}": ratio
                for band, ratio in zip(self.freq_bands[:-1], power_ratio)
            },
        }

    def extract_nonlinear_features(self, signal: np.ndarray) -> Dict[str, float]:
        mobility, complexity = pyeeg.hjorth(signal)
        return {
            "hfd": pyeeg.hfd(signal, Kmax=5),
            "pfd": pyeeg.pfd(signal),
            "hurst": pyeeg.hurst(signal),
            "hjorth_mobility": mobility,
            "hjorth_complexity": complexity,
        }

    def extract_all_features(self, signal: np.ndarray) -> EEGFeatures:
        return EEGFeatures(
            temporal=self.extract_temporal_features(signal),
            spectral=self.extract_spectral_features(signal),
            nonlinear=self.extract_nonlinear_features(signal),
        )


class EEGPreprocessor:
    def __init__(
        self, processor: SignalProcessor, epoch_length: int = 10, step_size: int = 1
    ):
        self.processor = processor
        self.epoch_length = epoch_length
        self.step_size = step_size
        self.logger = logging.getLogger(__name__)

    def load_and_filter(self, file_path: str) -> mne.io.Raw:
        raw = mne.io.read_raw_edf(file_path, preload=True)
        raw.filter(l_freq=0.25, h_freq=25)
        return raw

    def process_epoch(
        self,
        raw: mne.io.Raw,
        start_time: float,
        seizure_intervals: Optional[List[Tuple[float, float]]] = None,
    ) -> Dict:
        start, stop = raw.time_as_index([start_time, start_time + self.epoch_length])
        data = raw[:, start:stop][0]

        features = {"start_time": start_time}

        for idx, channel in enumerate(raw.ch_names):
            channel_features = self.processor.extract_all_features(data[idx])
            features.update(
                {
                    f"{channel}_{key}": value
                    for feature_type in vars(channel_features).values()
                    for key, value in feature_type.items()
                }
            )

        if seizure_intervals:
            features["seizure"] = any(
                start_time > start
                and start_time < end
                or start_time + self.epoch_length > start
                and start_time + self.epoch_length < end
                for start, end in seizure_intervals
            )
        else:
            features["seizure"] = 0

        return features

    def process_recording(
        self,
        file_path: str,
        seizure_intervals: Optional[List[Tuple[float, float]]] = None,
    ) -> pd.DataFrame:
        raw = self.load_and_filter(file_path)
        epochs = []

        start_time = 0
        while start_time <= raw.times[-1] - self.epoch_length:
            self.logger.info(f"Processing epoch starting at {start_time}s")
            epoch_features = self.process_epoch(raw, start_time, seizure_intervals)
            epochs.append(epoch_features)
            start_time += self.step_size

        return pd.DataFrame(epochs)


def main():
    logging.basicConfig(level=logging.INFO)

    data_dir = "data"
    output_dir = "processed_data"
    seizure_info = {
        "chb02_16": [[130, 212]],
        "chb05_06": [[417, 532]],
        "chb05_13": [[1086, 1196]],
        "chb05_16": [[2317, 2413]],
        "chb05_17": [[2451, 2571]],
        "chb05_22": [[2348, 2465]],
        "chb08_02": [[2670, 2841]],
        "chb08_05": [[2856, 3046]],
        "chb08_11": [[2988, 3211]],
        "chb08_13": [[2417, 2577]],
        "chb08_21": [[2083, 2347]],
    }

    signal_processor = SignalProcessor()
    eeg_processor = EEGPreprocessor(signal_processor)

    for filename in os.listdir(data_dir):
        if filename.endswith(".edf"):
            file_path = os.path.join(data_dir, filename)
            recording_id = os.path.splitext(filename)[0]

            seizure_intervals = seizure_info.get(recording_id)

            features_df = eeg_processor.process_recording(file_path, seizure_intervals)

            output_path = os.path.join(output_dir, f"{recording_id}.csv")
            features_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
