#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SSVEP Classifier for EEG Data

This script provides functions to process EEG data and classify SSVEP responses
using a combination of Filter Bank Canonical Correlation Analysis (FBCCA) and
Power Spectral Density (PSD) features with SVM classification.

Input data:
- EEG time series (NumPy arrays or .mat files)
- Sampling rate: 256Hz
- Channels: CH2-CH9 (8 channels)
- Stimulus frequencies (e.g., [9, 10, 12, 15] Hz)
- Trigger/label channels (CH10: trigger, CH11: LDA label)

Processing:
1. Preprocessing: 4-45Hz bandpass filter + multiple notch filters
2. Sliding windows: 3-second epochs with 1-second stride
3. Feature extraction:
   a. FBCCA for each stimulus frequency
   b. PSD (Welch method) for 4-45Hz range
   c. Inter-channel correlation features
   d. Common spatial patterns
   e. Adaptive filter features

Classification:
- Ensemble classifier (SVM, Random Forest, Neural Network)
- 80:20 train-test split with standardization
- Outputs accuracy and confusion matrix
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.linalg import eigh
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import mat73  # For loading .mat files
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

warnings.filterwarnings('ignore')

def load_data(file_path):
    """
    Load EEG data from a .mat file or numpy array
    
    Parameters:
    -----------
    file_path : str
        Path to the .mat file
        
    Returns:
    --------
    eeg_data : ndarray
        EEG data with shape (channels, samples)
    """
    try:
        # Load .mat file
        if file_path.endswith('.mat'):
            data_dict = mat73.loadmat(file_path)
            # Assuming data is stored under 'y' based on explored notebooks
            if 'y' in data_dict:
                return data_dict['y']
            else:
                keys = list(data_dict.keys())
                return data_dict[keys[0]]
        else:
            # Load numpy array
            return np.load(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def bandpass_filter(eeg_data, lowcut=4, highcut=45, fs=256, order=6):
    """
    Apply bandpass filter to EEG data
    
    Parameters:
    -----------
    eeg_data : ndarray
        EEG data with shape (channels, samples)
    lowcut : float
        Lower cutoff frequency
    highcut : float
        Upper cutoff frequency
    fs : float
        Sampling frequency
    order : int
        Filter order
        
    Returns:
    --------
    filtered_data : ndarray
        Filtered EEG data
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # Design Butterworth filter
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Apply filter
    filtered_data = np.zeros_like(eeg_data)
    for i in range(eeg_data.shape[0]):
        filtered_data[i] = signal.filtfilt(b, a, eeg_data[i])
    
    return filtered_data

def notch_filter(eeg_data, notch_freq=50, quality_factor=30, fs=256):
    """
    Apply notch filter to remove power line noise
    
    Parameters:
    -----------
    eeg_data : ndarray
        EEG data with shape (channels, samples)
    notch_freq : float
        Frequency to remove
    quality_factor : float
        Quality factor of the notch filter
    fs : float
        Sampling frequency
        
    Returns:
    --------
    filtered_data : ndarray
        Filtered EEG data
    """
    b, a = signal.iirnotch(notch_freq, quality_factor, fs)
    
    filtered_data = np.zeros_like(eeg_data)
    for i in range(eeg_data.shape[0]):
        filtered_data[i] = signal.filtfilt(b, a, eeg_data[i])
    
    return filtered_data

def preprocess_eeg(eeg_data, apply_notch=True, fs=256):
    """
    Preprocess EEG data with bandpass and optional notch filter
    
    Parameters:
    -----------
    eeg_data : ndarray
        EEG data with shape (channels, samples)
    apply_notch : bool
        Whether to apply notch filter
    fs : float
        Sampling frequency
        
    Returns:
    --------
    preprocessed_data : ndarray
        Preprocessed EEG data
    """
    # Apply bandpass filter first (4-45 Hz)
    filtered_data = bandpass_filter(eeg_data, 4, 45, fs)
    
    # Apply multiple notch filters if needed
    if apply_notch:
        # 50 Hz (power line noise)
        filtered_data = notch_filter(filtered_data, 50, 30, fs)
        # 100 Hz (first harmonic)
        filtered_data = notch_filter(filtered_data, 100, 30, fs)
    
    return filtered_data

def create_reference_signals(freqs, fs, T):
    """
    Create reference signals for CCA analysis
    
    Parameters:
    -----------
    freqs : list of float
        Target frequencies
    fs : float
        Sampling frequency
    T : float
        Duration in seconds
        
    Returns:
    --------
    Y : dict
        Dictionary containing reference signals for each frequency
    """
    t = np.arange(0, T, 1/fs)
    Y = {}
    
    for freq in freqs:
        signals = []
        
        # Add up to 5 harmonics
        for h in range(1, 6):
            # Sine and cosine at each harmonic
            signals.append(np.sin(2 * np.pi * h * freq * t))
            signals.append(np.cos(2 * np.pi * h * freq * t))
        
        # Stack reference signals
        Y[freq] = np.vstack(signals)
    
    return Y

def canonical_correlation(X, Y):
    """
    Calculate Canonical Correlation between X and Y
    
    Parameters:
    -----------
    X : ndarray
        First dataset with shape (features1, samples)
    Y : ndarray
        Second dataset with shape (features2, samples)
        
    Returns:
    --------
    r : float
        Maximum canonical correlation coefficient
    """
    # Center the data
    X = X - X.mean(axis=1, keepdims=True)
    Y = Y - Y.mean(axis=1, keepdims=True)
    
    # Calculate covariance matrices
    Cxx = np.cov(X)
    Cyy = np.cov(Y)
    Cxy = np.cov(X, Y)[:X.shape[0], X.shape[0]:]
    
    # Add small regularization to avoid singularity issues
    Cxx += np.eye(Cxx.shape[0]) * 1e-8
    Cyy += np.eye(Cyy.shape[0]) * 1e-8
    
    # Compute canonical correlations
    Cxx_inv_sqrt = np.linalg.inv(np.linalg.cholesky(Cxx))
    Cyy_inv_sqrt = np.linalg.inv(np.linalg.cholesky(Cyy))
    
    T = Cxx_inv_sqrt @ Cxy @ Cyy_inv_sqrt
    
    # SVD to get correlations
    U, s, Vh = np.linalg.svd(T, full_matrices=False)
    
    # Return maximum correlation
    return s[0]

def apply_filter_bank(eeg_data, fs, n_bands=8):
    """
    Apply filter bank to EEG data
    
    Parameters:
    -----------
    eeg_data : ndarray
        EEG data with shape (channels, samples)
    fs : float
        Sampling frequency
    n_bands : int
        Number of filter bands
        
    Returns:
    --------
    filtered_bands : list
        List of filtered EEG data for each band
    """
    filtered_bands = []
    
    # Define filter bank with increasing cutoff frequencies
    for i in range(n_bands):
        # Each filter has passband with increasing lower cutoff
        low_cut = 6 + 3 * i  # Start from 6Hz, increase by 3Hz
        high_cut = min(90, fs/2 - 1)  # Nyquist limit
        
        # Apply bandpass filter
        b, a = signal.butter(6, [low_cut/(fs/2), high_cut/(fs/2)], btype='band')
        
        # Apply filter to each channel
        filtered_band = np.zeros_like(eeg_data)
        for ch in range(eeg_data.shape[0]):
            filtered_band[ch] = signal.filtfilt(b, a, eeg_data[ch])
        
        filtered_bands.append(filtered_band)
    
    return filtered_bands

def fbcca_features(eeg_epoch, stim_freqs, fs=256, n_bands=8):
    """
    Extract FBCCA features for SSVEP classification
    
    Parameters:
    -----------
    eeg_epoch : ndarray
        EEG epoch data with shape (channels, samples)
    stim_freqs : list
        List of stimulus frequencies
    fs : float
        Sampling frequency
    n_bands : int
        Number of filter bands
        
    Returns:
    --------
    features : ndarray
        FBCCA features (correlation coefficients for each frequency)
    """
    # Duration of the epoch in seconds
    T = eeg_epoch.shape[1] / fs
    
    # Create reference signals
    Y_ref = create_reference_signals(stim_freqs, fs, T)
    
    # Apply filter bank
    filter_bank_signals = apply_filter_bank(eeg_epoch, fs, n_bands)
    
    # Calculate weights for each band (emphasize lower frequency bands)
    weights = np.power(np.arange(1, len(filter_bank_signals) + 1), -1.5)
    
    # Calculate CCA for each frequency
    r_all = []
    
    for freq in stim_freqs:
        # Get reference signals for this frequency
        Y = Y_ref[freq]
        
        # Calculate CCA for each filter bank
        r_per_band = []
        for filtered_signal in filter_bank_signals:
            r = canonical_correlation(filtered_signal, Y)
            r_per_band.append(r)
        
        # Apply weights to band correlations
        r_weighted = np.sum(np.array(r_per_band) * weights[:len(r_per_band)])
        r_all.append(r_weighted)
        
        # Add max correlation as additional feature
        r_all.append(np.max(r_per_band))
        
        # Add variance of correlations as feature
        r_all.append(np.var(r_per_band))
    
    return np.array(r_all)

def calculate_psd(eeg_epoch, fs=256, nperseg=512, freq_range=(4, 45)):
    """
    Calculate Power Spectral Density using Welch's method
    
    Parameters:
    -----------
    eeg_epoch : ndarray
        EEG epoch data with shape (channels, samples)
    fs : float
        Sampling frequency
    nperseg : int
        Length of each segment for FFT
    freq_range : tuple
        Frequency range to extract (min_freq, max_freq)
        
    Returns:
    --------
    psd_features : ndarray
        PSD features for all channels
    """
    min_freq, max_freq = freq_range
    
    # Initialize array to store PSD features
    psd_features = []
    
    # Calculate PSD for each channel
    for ch in range(eeg_epoch.shape[0]):
        f, Pxx = signal.welch(eeg_epoch[ch], fs=fs, nperseg=nperseg)
        
        # Extract PSD in the desired frequency range
        idx = np.logical_and(f >= min_freq, f <= max_freq)
        psd_features.append(Pxx[idx])
        
        # Calculate band power
        theta = np.mean(Pxx[np.logical_and(f >= 4, f < 8)])
        alpha = np.mean(Pxx[np.logical_and(f >= 8, f < 13)])
        beta = np.mean(Pxx[np.logical_and(f >= 13, f < 30)])
        gamma = np.mean(Pxx[np.logical_and(f >= 30, f <= 45)])
        
        # Add band power features
        psd_features.append(np.array([theta, alpha, beta, gamma]))
        
        # Add band power ratios
        psd_features.append(np.array([
            beta/alpha,
            gamma/beta,
            theta/alpha,
            (beta+gamma)/(theta+alpha)
        ]))
    
    # Flatten to create feature vector
    return np.hstack(psd_features)

def extract_epochs(eeg_data, trigger_channel, epoch_duration=3, stride=1, fs=256):
    """
    Extract epochs from continuous EEG data based on trigger channel
    
    Parameters:
    -----------
    eeg_data : ndarray
        EEG data with shape (channels, samples)
    trigger_channel : ndarray
        Trigger channel data with shape (samples,)
    epoch_duration : float
        Duration of each epoch in seconds
    stride : float
        Stride (step) between epochs in seconds
    fs : float
        Sampling frequency
        
    Returns:
    --------
    epochs : list
        List of EEG epochs
    """
    epoch_samples = int(epoch_duration * fs)
    stride_samples = int(stride * fs)
    
    # Find trigger onsets (0->1 transitions)
    trigger_onsets = np.where(np.diff(np.concatenate(([0], trigger_channel))) == 1)[0]
    
    # Find trigger offsets (1->0 transitions)
    trigger_offsets = np.where(np.diff(np.concatenate((trigger_channel, [0]))) == -1)[0]
    
    # Ensure equal number of onsets and offsets
    min_len = min(len(trigger_onsets), len(trigger_offsets))
    trigger_intervals = np.vstack([trigger_onsets[:min_len], trigger_offsets[:min_len]]).T
    
    epochs = []
    labels = []
    
    # Extract epochs from each trigger interval
    for start, end in trigger_intervals:
        # Get the stimulus label from the interval
        # Assuming constant label within interval
        if end > start:  # Ensure valid interval
            label = int(eeg_data[10, start:end].mean())  # CH11 (index 10) contains labels
            
            # Extract epochs with stride
            for i in range(start, end - epoch_samples + 1, stride_samples):
                epoch = eeg_data[1:9, i:i+epoch_samples]  # CH2-CH9 (index 1-8)
                
                if epoch.shape[1] == epoch_samples:  # Ensure complete epoch
                    epochs.append(epoch)
                    labels.append(label)
    
    return epochs, labels

class SpatialFilterTransformer(BaseEstimator, TransformerMixin):
    """
    Apply spatial filtering (CSP) to EEG epochs
    """
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.filters_ = None
        
    def fit(self, X, y):
        n_epochs = len(X)
        n_channels = X[0].shape[0]
        
        # Initialize covariance matrices for each class
        classes = np.unique(y)
        covs = {c: np.zeros((n_channels, n_channels)) for c in classes}
        counts = {c: 0 for c in classes}
        
        # Compute covariance matrices for each class
        for i in range(n_epochs):
            epoch = X[i]
            c = y[i]
            
            # Compute covariance
            cov = np.cov(epoch)
            covs[c] += cov
            counts[c] += 1
        
        # Average covariance matrices
        for c in classes:
            if counts[c] > 0:
                covs[c] /= counts[c]
        
        # Compute CSP filters
        composite_cov = sum(covs.values())
        eigvals, eigvecs = np.linalg.eig(composite_cov)
        
        # Whiten data
        whitening = eigvecs @ np.diag(1.0/np.sqrt(eigvals)) @ eigvecs.T
        
        # Compute CSP
        filtered_covs = {c: whitening @ cov @ whitening.T for c, cov in covs.items()}
        
        # Joint diagonalization
        if len(classes) == 2:
            # Simple case: two classes
            eigvals, eigvecs = np.linalg.eig(filtered_covs[classes[0]] @ np.linalg.inv(filtered_covs[classes[1]]))
            idx = np.argsort(eigvals)[::-1]
            eigvecs = eigvecs[:, idx]
            
            # Select filters
            self.filters_ = eigvecs[:, :self.n_components]
        else:
            # Multi-class case: use one-vs-rest approach
            all_filters = []
            for c1 in classes:
                rest_cov = sum([cov for c2, cov in filtered_covs.items() if c2 != c1])
                
                # Find eigenvectors
                eigvals, eigvecs = np.linalg.eig(filtered_covs[c1] @ np.linalg.inv(rest_cov))
                idx = np.argsort(eigvals)[::-1]
                eigvecs = eigvecs[:, idx]
                
                # Add top and bottom filters
                half_comp = self.n_components // len(classes)
                all_filters.extend([eigvecs[:, i] for i in range(half_comp)])
                all_filters.extend([eigvecs[:, -(i+1)] for i in range(half_comp)])
            
            self.filters_ = np.column_stack(all_filters[:self.n_components])
        
        return self
    
    def transform(self, X):
        if self.filters_ is None:
            raise ValueError("Transformer not fitted")
        
        n_epochs = len(X)
        n_samples = X[0].shape[1]
        
        X_transformed = np.zeros((n_epochs, self.n_components * 2))
        
        for i in range(n_epochs):
            # Apply spatial filter
            filtered_data = self.filters_.T @ X[i]
            
            # Extract features: log variance
            X_transformed[i, :self.n_components] = np.log(np.var(filtered_data, axis=1))
            
            # Add mean power as features
            X_transformed[i, self.n_components:] = np.log(np.mean(filtered_data ** 2, axis=1))
        
        return X_transformed

def compute_interchannel_correlation(eeg_epoch):
    """
    Compute correlation between EEG channels as additional features
    
    Parameters:
    -----------
    eeg_epoch : ndarray
        EEG epoch data with shape (channels, samples)
    
    Returns:
    --------
    corr_features : ndarray
        Inter-channel correlation features
    """
    # Compute correlation matrix
    corr_matrix = np.corrcoef(eeg_epoch)
    
    # Extract upper triangle (excluding diagonal)
    mask = np.triu_indices(corr_matrix.shape[0], k=1)
    corr_features = corr_matrix[mask]
    
    return corr_features

def adaptive_filter_features(eeg_epoch, fs=256):
    """
    Extract features using adaptive filtering
    
    Parameters:
    -----------
    eeg_epoch : ndarray
        EEG epoch data with shape (channels, samples)
    fs : float
        Sampling frequency
    
    Returns:
    --------
    features : ndarray
        Adaptive filter features
    """
    # Number of channels and samples
    n_channels, n_samples = eeg_epoch.shape
    
    # FFT for each channel
    fft_features = []
    for ch in range(n_channels):
        # Compute FFT
        fft = np.fft.rfft(eeg_epoch[ch])
        fft_power = np.abs(fft)**2
        
        # Get frequencies
        freqs = np.fft.rfftfreq(n_samples, 1/fs)
        
        # Extract power at specific frequency bands
        idx_theta = np.logical_and(freqs >= 4, freqs < 8)
        idx_alpha = np.logical_and(freqs >= 8, freqs < 13)
        idx_beta = np.logical_and(freqs >= 13, freqs < 30)
        idx_gamma = np.logical_and(freqs >= 30, freqs <= 45)
        
        # Extract peaks in each band
        peak_theta = np.max(fft_power[idx_theta]) if np.any(idx_theta) else 0
        peak_alpha = np.max(fft_power[idx_alpha]) if np.any(idx_alpha) else 0
        peak_beta = np.max(fft_power[idx_beta]) if np.any(idx_beta) else 0
        peak_gamma = np.max(fft_power[idx_gamma]) if np.any(idx_gamma) else 0
        
        # Find frequency of maximum power
        idx_max = np.argmax(fft_power[1:]) + 1  # Skip DC component
        freq_max = freqs[idx_max]
        
        # Add features
        fft_features.extend([peak_theta, peak_alpha, peak_beta, peak_gamma, freq_max])
    
    return np.array(fft_features)

def extract_features(epochs, stim_freqs, fs=256, include_spatial=True):
    """
    Extract features from EEG epochs
    
    Parameters:
    -----------
    epochs : list
        List of EEG epochs
    stim_freqs : list
        List of stimulus frequencies
    fs : float
        Sampling frequency
    include_spatial : bool
        Whether to include spatial filtering features
        
    Returns:
    --------
    features : ndarray
        Feature matrix
    """
    all_features = []
    
    for epoch in epochs:
        # Preprocess epoch with enhanced methods
        processed_epoch = preprocess_eeg(epoch, apply_notch=True, fs=fs)
        
        # Extract FBCCA features
        fbcca_feats = fbcca_features(processed_epoch, stim_freqs, fs)
        
        # Extract PSD features
        psd_feats = calculate_psd(processed_epoch, fs)
        
        # Add inter-channel correlation features
        ic_feats = compute_interchannel_correlation(processed_epoch)
        
        # Add adaptive filter features
        af_feats = adaptive_filter_features(processed_epoch, fs)
        
        # Concatenate features
        combined_features = np.concatenate([fbcca_feats, psd_feats, ic_feats, af_feats])
        all_features.append(combined_features)
    
    feature_matrix = np.array(all_features)
    
    return feature_matrix

def train_svm_classifier(features, labels, test_size=0.2, random_state=42):
    """
    Train SVM classifier on extracted features
    
    Parameters:
    -----------
    features : ndarray
        Feature matrix
    labels : ndarray
        Label vector
    test_size : float
        Proportion of the dataset to include in the test split
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    model : SVC
        Trained SVM model
    X_train : ndarray
        Training features
    X_test : ndarray
        Testing features
    y_train : ndarray
        Training labels
    y_test : ndarray
        Testing labels
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train SVM model
    model = SVC(kernel='rbf', C=10, gamma='scale')
    model.fit(X_train_scaled, y_train)
    
    return model, X_train_scaled, X_test_scaled, y_train, y_test

def evaluate_model(model, X_test, y_test):
    """
    Evaluate SVM model and return metrics
    
    Parameters:
    -----------
    model : SVC
        Trained SVM model
    X_test : ndarray
        Testing features
    y_test : ndarray
        Testing labels
        
    Returns:
    --------
    accuracy : float
        Classification accuracy
    conf_matrix : ndarray
        Confusion matrix
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return accuracy, conf_matrix, y_pred

def plot_confusion_matrix(conf_matrix, class_names=None):
    """
    Plot confusion matrix
    
    Parameters:
    -----------
    conf_matrix : ndarray
        Confusion matrix
    class_names : list
        List of class names
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

def main(file_path, stim_freqs, fs=256):
    """
    Main function to run the SSVEP classification pipeline
    
    Parameters:
    -----------
    file_path : str
        Path to the EEG data file
    stim_freqs : list
        List of stimulus frequencies
    fs : float
        Sampling frequency
    """
    # 1. Load data
    print("Loading data...")
    eeg_data = load_data(file_path)
    
    if eeg_data is None:
        print("Failed to load data. Exiting.")
        return
    
    print(f"Data loaded with shape: {eeg_data.shape}")
    
    # 2. Extract epochs
    print("Extracting epochs...")
    epochs, labels = extract_epochs(eeg_data, eeg_data[9], epoch_duration=3, stride=1, fs=fs)
    
    print(f"Extracted {len(epochs)} epochs")
    
    # 3. Extract features
    print("Extracting features...")
    features = extract_features(epochs, stim_freqs, fs)
    
    print(f"Feature shape: {features.shape}")
    
    # 4. Train SVM classifier
    print("Training SVM classifier...")
    model, X_train, X_test, y_train, y_test = train_svm_classifier(features, labels)
    
    # 5. Evaluate model
    print("Evaluating model...")
    accuracy, conf_matrix, y_pred = evaluate_model(model, X_test, y_test)
    
    print(f"Classification accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    
    # 6. Plot confusion matrix
    unique_labels = np.unique(labels)
    class_names = [f"Freq {stim_freqs[i-1]} Hz" if i-1 < len(stim_freqs) else f"Class {i}" 
                   for i in unique_labels]
    
    plot_confusion_matrix(conf_matrix, class_names)
    
    # Return results
    return model, epochs, features, labels, y_test, y_pred, accuracy

if __name__ == "__main__":
    # Example usage
    file_path = "data/subject_1_fvep_led_training_1.mat"  # Replace with your data file
    stim_freqs = [15, 12, 10, 9]  # Corrected stimulus frequencies (top, right, bottom, left)
    
    main(file_path, stim_freqs) 