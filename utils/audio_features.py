import os
import numpy as np
import pandas as pd
import librosa
from scipy import signal
from scipy.stats import kurtosis, skew
from tqdm import tqdm  # For progress tracking

class AudioFeatureExtractor:
    """
    Class for extracting audio features from audio files
    """
    
    def __init__(self, filter_freq_range=(5000, 100000), file_extensions=None):
        """
        Initialize the audio feature extractor
        
        Args:
            filter_freq_range (tuple): Frequency range for the bandpass filter (min_hz, max_hz)
            file_extensions (list): List of file extensions to process (default: ['.wav'])
        """
        self.filter_freq_range = filter_freq_range
        self.file_extensions = file_extensions or ['.wav']
        
    def extract_from_directory(self, directory_path, verbose=True):
        """
        Extract audio features from all audio files in the specified directory
        
        Args:
            directory_path (str): Path to the directory containing audio files
            verbose (bool): Whether to show progress and print errors
            
        Returns:
            pd.DataFrame: DataFrame containing the extracted features
        """
        # Check if the directory exists
        if not os.path.isdir(directory_path):
            raise ValueError(f"'{directory_path}' is not a directory")
        
        # Get all audio files with specified extensions
        audio_files = []
        for ext in self.file_extensions:
            audio_files.extend([
                os.path.join(directory_path, f) 
                for f in os.listdir(directory_path) 
                if os.path.isfile(os.path.join(directory_path, f)) and f.lower().endswith(ext)
            ])
        
        if not audio_files:
            if verbose:
                print(f"No audio files with extensions {self.file_extensions} found in {directory_path}")
            return pd.DataFrame()
        
        audio_properties = []
        
        # Process each audio file with error handling
        for file in tqdm(audio_files, desc="Processing audio files") if verbose else audio_files:
            try:
                features = self.extract_from_file(file)
                audio_properties.append(features)
            except Exception as e:
                if verbose:
                    print(f"Error processing {file}: {e}")
            
        columns = [
            "File", "Peak Frequency", "Mean ICI (s)", "SNR (dB)", "Kurtosis", "Skewness",
            "Amplitude Mean", "Amplitude Std", "Amplitude Min", "Amplitude Max",
            "RMS Mean", "RMS Std", "RMS Min", "RMS Max",
            "Spectral Centroid Mean", "Spectral Centroid Std", "Spectral Centroid Min", "Spectral Centroid Max",
            "Spectral Bandwidth Mean", "Spectral Bandwidth Std", "Spectral Bandwidth Min", "Spectral Bandwidth Max",
            "Spectral Flatness Mean", "Spectral Flatness Std", "Spectral Flatness Min", "Spectral Flatness Max"
        ]
        
        return pd.DataFrame(audio_properties, columns=columns)
    
    def extract_from_file(self, file_path):
        """
        Extract audio features from an audio file
        
        Args:
            file_path (str): Path to the audio file
            
        Returns:
            list: List of extracted features
        """
        y, sr = librosa.load(file_path, sr=None)
        
        # Apply a bandpass filter
        y_filtered = self._apply_bandpass_filter(y, sr)
        
        # Extract all features
        amp_features = self._extract_amplitude_features(y_filtered)
        rms_features = self._extract_rms_features(y_filtered)
        spectral_features = self._extract_spectral_features(y_filtered, sr)
        peak_freq = self._find_peak_frequency(y_filtered, sr)
        mean_ici = self._calculate_ici(y_filtered, sr)
        snr = self._calculate_snr(y_filtered)
        kurt, skewness = self._calculate_spectral_shape(y_filtered)
        
        # Assemble all features
        return [file_path, peak_freq, mean_ici, snr, kurt, skewness, 
                *amp_features, *rms_features, 
                *spectral_features["centroid"], *spectral_features["bandwidth"], *spectral_features["flatness"]]
    
    def _apply_bandpass_filter(self, y, sr):
        """Apply a bandpass filter to the signal"""
        sos = signal.butter(6, self.filter_freq_range, 'bandpass', fs=sr, output='sos')
        return signal.sosfiltfilt(sos, y)
    
    def _extract_amplitude_features(self, y):
        """Extract amplitude features"""
        return [np.mean(y), np.std(y), np.min(y), np.max(y)]
    
    def _extract_rms_features(self, y):
        """Extract RMS features"""
        rms = librosa.feature.rms(y=y)
        return [np.mean(rms), np.std(rms), np.min(rms), np.max(rms)]
    
    def _extract_spectral_features(self, y, sr):
        """Extract various spectral features"""
        features = {}
        
        # Spectral Centroid
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features["centroid"] = [np.mean(centroid), np.std(centroid), np.min(centroid), np.max(centroid)]
        
        # Spectral Bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features["bandwidth"] = [np.mean(bandwidth), np.std(bandwidth), np.min(bandwidth), np.max(bandwidth)]
        
        # Spectral Flatness
        flatness = librosa.feature.spectral_flatness(y=y)
        features["flatness"] = [np.mean(flatness), np.std(flatness), np.min(flatness), np.max(flatness)]
        
        return features
    
    def _find_peak_frequency(self, y, sr):
        """Find the dominant frequency"""
        D = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        return freqs[np.argmax(np.mean(D, axis=1))]
    
    def _calculate_ici(self, y, sr):
        """Calculate the inter-click interval (ICI)"""
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        ici_values = np.diff(onset_times)
        return np.mean(ici_values) if len(ici_values) > 0 else 0
    
    def _calculate_snr(self, y):
        """Calculate the signal-to-noise ratio (SNR)"""
        signal_power = np.mean(y**2)
        noise_power = np.var(y)
        return 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0
    
    def _calculate_spectral_shape(self, y):
        """Calculate the kurtosis and skewness of the spectrum"""
        spec = np.abs(librosa.stft(y))
        spec_mean = np.mean(spec, axis=1)
        return kurtosis(spec_mean), skew(spec_mean)