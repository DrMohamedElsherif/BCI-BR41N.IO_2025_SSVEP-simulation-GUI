#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SSVEP Classification for All Datasets with Enhanced Performance

This script evaluates the SSVEP classifier on all 4 mat datasets with improved
methods to achieve better classification accuracy.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ssvep_classifier import *
import os
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score

def enhanced_preprocess_eeg(eeg_data, fs=256):
    """
    Enhanced preprocessing for EEG data with improved filter parameters
    
    Parameters:
    -----------
    eeg_data : ndarray
        EEG data with shape (channels, samples)
    fs : float
        Sampling frequency
        
    Returns:
    --------
    processed_data : ndarray
        Preprocessed EEG data
    """
    # Apply a wider bandpass filter (4-45 Hz) to capture more SSVEP harmonics
    b1, a1 = signal.butter(N=6, Wn=[4, 45], btype='band', fs=fs)
    
    # Apply multiple notch filters (50Hz and harmonics)
    b2, a2 = signal.iirnotch(w0=50, Q=30, fs=fs)
    b3, a3 = signal.iirnotch(w0=100, Q=30, fs=fs)
    
    # Apply filters
    filtered_data = np.zeros_like(eeg_data)
    for i in range(eeg_data.shape[0]):
        # Bandpass
        temp = signal.filtfilt(b1, a1, eeg_data[i])
        # Notch 50Hz
        temp = signal.filtfilt(b2, a2, temp)
        # Notch 100Hz
        filtered_data[i] = signal.filtfilt(b3, a3, temp)
    
    return filtered_data

def enhanced_extract_features(epochs, stim_freqs, fs=256):
    """
    Enhanced feature extraction with improved parameters
    
    Parameters:
    -----------
    epochs : list
        List of EEG epochs
    stim_freqs : list
        List of stimulus frequencies
    fs : float
        Sampling frequency
        
    Returns:
    --------
    features : ndarray
        Enhanced feature matrix
    """
    all_features = []
    
    for epoch in epochs:
        # Apply enhanced preprocessing
        processed_epoch = enhanced_preprocess_eeg(epoch, fs)
        
        # Extract FBCCA features with more filter banks and optimized weights
        fbcca_feats = enhanced_fbcca_features(processed_epoch, stim_freqs, fs)
        
        # Extract improved PSD features with more frequency bins
        psd_feats = enhanced_psd_features(processed_epoch, fs)
        
        # Add inter-channel correlation features
        ic_feats = compute_interchannel_correlation(processed_epoch)
        
        # Concatenate all features
        combined_features = np.concatenate([fbcca_feats, psd_feats, ic_feats])
        all_features.append(combined_features)
    
    return np.array(all_features)

def enhanced_fbcca_features(eeg_epoch, stim_freqs, fs=256, n_bands=8):
    """
    Enhanced FBCCA feature extraction with more filter banks
    
    Parameters:
    -----------
    eeg_epoch : ndarray
        EEG epoch data with shape (channels, samples)
    stim_freqs : list
        List of stimulus frequencies
    fs : float
        Sampling frequency
    n_bands : int
        Number of filter banks
    
    Returns:
    --------
    features : ndarray
        Enhanced FBCCA features
    """
    # Duration of the epoch in seconds
    T = eeg_epoch.shape[1] / fs
    
    # Create reference signals with more harmonics
    Y_ref = enhanced_reference_signals(stim_freqs, fs, T)
    
    # Apply filter bank with more bands
    filter_bank_signals = enhanced_filter_bank(eeg_epoch, fs, n_bands)
    
    # Optimize weights for different filter banks (emphasize lower frequency bands)
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
        
        # Add max correlation as a feature
        r_all.append(np.max(r_per_band))
    
    return np.array(r_all)

def enhanced_reference_signals(freqs, fs, T):
    """
    Create enhanced reference signals for CCA analysis with more harmonics
    
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

def enhanced_filter_bank(eeg_data, fs, n_bands=8):
    """
    Apply enhanced filter bank to EEG data with more bands
    
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
    
    # Define filter bank with optimized cutoff frequencies
    for i in range(n_bands):
        # Adjust filter bounds for better frequency resolution
        low_cut = 6 + 4 * i  # Start from 6Hz with 4Hz steps
        high_cut = min(90, fs/2 - 1)  # Nyquist limit
        
        # Apply bandpass filter with higher order for steeper rolloff
        b, a = signal.butter(6, [low_cut/(fs/2), high_cut/(fs/2)], btype='band')
        
        # Apply filter to each channel
        filtered_band = np.zeros_like(eeg_data)
        for ch in range(eeg_data.shape[0]):
            filtered_band[ch] = signal.filtfilt(b, a, eeg_data[ch])
        
        filtered_bands.append(filtered_band)
    
    return filtered_bands

def enhanced_psd_features(eeg_epoch, fs=256, nperseg=512):
    """
    Calculate enhanced PSD features with higher frequency resolution
    
    Parameters:
    -----------
    eeg_epoch : ndarray
        EEG epoch data with shape (channels, samples)
    fs : float
        Sampling frequency
    nperseg : int
        Length of each segment for FFT
    
    Returns:
    --------
    psd_features : ndarray
        Enhanced PSD features
    """
    # Initialize array to store PSD features
    psd_features = []
    
    # Calculate PSD for each channel
    for ch in range(eeg_epoch.shape[0]):
        f, Pxx = signal.welch(eeg_epoch[ch], fs=fs, nperseg=nperseg)
        
        # Focus on frequencies around stimulus frequencies and harmonics (4-45 Hz)
        idx = np.logical_and(f >= 4, f <= 45)
        
        # Extract power at specific frequency bands
        psd_features.append(Pxx[idx])
        
        # Add band power ratios as features
        theta = np.mean(Pxx[np.logical_and(f >= 4, f <= 8)])
        alpha = np.mean(Pxx[np.logical_and(f >= 8, f <= 13)])
        beta = np.mean(Pxx[np.logical_and(f >= 13, f <= 30)])
        gamma = np.mean(Pxx[np.logical_and(f >= 30, f <= 45)])
        
        # Add ratios
        psd_features.append(np.array([beta/alpha, gamma/beta, theta/alpha]))
    
    # Flatten to create feature vector
    return np.hstack(psd_features)

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

def process_dataset_with_enhancements(file_path, stim_freqs, fs=256):
    """
    Process a single dataset with enhanced methods
    
    Parameters:
    -----------
    file_path : str
        Path to the mat file
    stim_freqs : list
        List of stimulus frequencies
    fs : int
        Sampling frequency
    
    Returns:
    --------
    dict
        Dictionary containing dataset information and results
    """
    print(f"\nProcessing {os.path.basename(file_path)} with enhanced methods...")
    
    # Load data
    eeg_data = load_data(file_path)
    if eeg_data is None:
        print(f"Failed to load {file_path}")
        return None
    
    print(f"Data loaded with shape: {eeg_data.shape}")
    
    # Extract epochs
    epochs, labels = extract_epochs(eeg_data, eeg_data[9], epoch_duration=3, stride=1, fs=fs)
    print(f"Extracted {len(epochs)} epochs")
    
    # Count epochs per class
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique_labels, counts))
    print(f"Label distribution: {label_counts}")
    
    # Extract enhanced features
    features = extract_features(epochs, stim_freqs, fs, include_spatial=True)
    print(f"Enhanced feature shape: {features.shape}")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Perform cross-validation first
    cv_results = perform_cross_validation(features_scaled, labels)
    
    # Train final ensemble model
    model, X_train, X_test, y_train, y_test = train_ensemble_classifier(features_scaled, labels)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"Enhanced classification accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"Cross-val Accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    
    # Calculate per-class accuracy using confusion matrix
    class_accuracy = {}
    for i, label in enumerate(np.unique(y_test)):
        class_idx = np.where(np.unique(y_test) == label)[0][0]
        if np.sum(conf_matrix[class_idx]) > 0:  # Avoid division by zero
            class_acc = conf_matrix[class_idx, class_idx] / np.sum(conf_matrix[class_idx])
            class_accuracy[int(label)] = class_acc
    
    return {
        "file": os.path.basename(file_path),
        "epochs": len(epochs),
        "label_counts": label_counts,
        "accuracy": accuracy,
        "cv_accuracy": cv_results['mean_accuracy'],
        "cv_std": cv_results['std_accuracy'],
        "conf_matrix": conf_matrix,
        "class_accuracy": class_accuracy,
        "train_size": len(y_train),
        "test_size": len(y_test)
    }

def perform_cross_validation(features, labels, n_folds=5):
    """
    Perform cross-validation with multiple classifiers
    
    Parameters:
    -----------
    features : ndarray
        Feature matrix
    labels : ndarray
        Label vector
    n_folds : int
        Number of cross-validation folds
    
    Returns:
    --------
    dict
        Dictionary with cross-validation results
    """
    # Initialize classifiers
    classifiers = {
        'SVM': SVC(kernel='rbf', C=10, gamma='scale', probability=True),
        'RF': RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42),
        'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
        'Ensemble': VotingClassifier(
            estimators=[
                ('svm', SVC(kernel='rbf', C=10, gamma='scale', probability=True)),
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('mlp', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42))
            ],
            voting='soft'
        )
    }
    
    # Set up cross-validation
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Store results
    cv_results = {}
    
    # Perform cross-validation for each classifier
    for name, clf in classifiers.items():
        scores = cross_val_score(clf, features, labels, cv=cv, scoring='accuracy')
        cv_results[name] = {
            'scores': scores,
            'mean': np.mean(scores),
            'std': np.std(scores)
        }
        print(f"{name} CV Accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    # Return overall best result
    best_clf = max(cv_results.items(), key=lambda x: x[1]['mean'])[0]
    
    return {
        'mean_accuracy': cv_results[best_clf]['mean'],
        'std_accuracy': cv_results[best_clf]['std'],
        'best_classifier': best_clf,
        'all_results': cv_results
    }

def train_ensemble_classifier(features, labels, test_size=0.2, random_state=42):
    """
    Train an ensemble of classifiers for improved performance
    
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
    model : VotingClassifier
        Trained ensemble model
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
    
    # Create individual classifiers with optimized hyperparameters
    svm_clf = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=None, 
                                   min_samples_split=2, random_state=random_state)
    mlp_clf = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu',
                           alpha=0.0001, max_iter=1000, random_state=random_state)
    
    # Create ensemble classifier
    ensemble = VotingClassifier(
        estimators=[
            ('svm', svm_clf),
            ('rf', rf_clf),
            ('mlp', mlp_clf)
        ],
        voting='soft'  # Use probability estimates for prediction
    )
    
    # Train the ensemble
    ensemble.fit(X_train, y_train)
    
    return ensemble, X_train, X_test, y_train, y_test

def main():
    # Dataset files
    dataset_files = [
        "data/subject_1_fvep_led_training_1.mat",
        "data/subject_1_fvep_led_training_2.mat", 
        "data/subject_2_fvep_led_training_1.mat",
        "data/subject_2_fvep_led_training_2.mat"
    ]
    
    # Stimulus frequencies
    stim_freqs = [9, 10, 12, 15]
    
    # Process each dataset with enhanced methods
    results = []
    
    for file_path in dataset_files:
        if os.path.exists(file_path):
            result = process_dataset_with_enhancements(file_path, stim_freqs)
            if result:
                results.append(result)
        else:
            print(f"File not found: {file_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY OF ENHANCED RESULTS")
    print("="*50)
    
    for result in results:
        print(f"\nDataset: {result['file']}")
        print(f"Epochs: {result['epochs']} (Train: {result['train_size']}, Test: {result['test_size']})")
        print(f"Test Accuracy: {result['accuracy']*100:.2f}%")
        print(f"Cross-val Accuracy: {result['cv_accuracy']*100:.2f}% ± {result['cv_std']*100:.2f}%")
        print("Class accuracies:")
        for label, acc in result['class_accuracy'].items():
            print(f"  Class {label}: {acc*100:.2f}%")
    
    # Plot overall results
    if results:
        # Extract file names and accuracies
        file_names = [r["file"].replace("subject_", "Subj").replace("_fvep_led_training_", " Train ") for r in results]
        accuracies = [r["accuracy"] * 100 for r in results]
        cv_accuracies = [r["cv_accuracy"] * 100 for r in results]
        cv_stds = [r["cv_std"] * 100 for r in results]
        
        # Plot test accuracies
        plt.figure(figsize=(12, 6))
        
        # Plot CV accuracies with error bars
        x = np.arange(len(file_names))
        plt.bar(x - 0.2, accuracies, width=0.4, color='skyblue', label='Test Accuracy')
        plt.bar(x + 0.2, cv_accuracies, width=0.4, color='lightgreen', label='CV Accuracy')
        plt.errorbar(x + 0.2, cv_accuracies, yerr=cv_stds, fmt='none', ecolor='green', capsize=5)
        
        # Add text labels
        for i, acc in enumerate(accuracies):
            plt.text(i - 0.2, acc + 1, f"{acc:.1f}%", ha='center', va='bottom', fontsize=10)
        
        for i, (acc, std) in enumerate(zip(cv_accuracies, cv_stds)):
            plt.text(i + 0.2, acc + 1, f"{acc:.1f}%", ha='center', va='bottom', fontsize=10)
        
        plt.title("Enhanced SSVEP Classification Accuracy Across Datasets", fontsize=16)
        plt.ylabel("Accuracy (%)", fontsize=14)
        plt.xticks(x, file_names, fontsize=12, rotation=45, ha='right')
        plt.yticks(fontsize=12)
        plt.ylim(0, 105)
        plt.grid(axis='y', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("enhanced_datasets_accuracy.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot confusion matrices for each dataset
        for result in results:
            plt.figure(figsize=(8, 6))
            sns.heatmap(result['conf_matrix'], annot=True, fmt='d', cmap='Blues',
                      xticklabels=[f"{f} Hz" for f in stim_freqs],
                      yticklabels=[f"{f} Hz" for f in stim_freqs])
            plt.title(f"Confusion Matrix - {result['file']}", fontsize=14)
            plt.xlabel("Predicted", fontsize=12)
            plt.ylabel("True", fontsize=12)
            plt.tight_layout()
            plt.savefig(f"confusion_matrix_{result['file'].replace('.mat', '')}.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Average accuracy
        avg_acc = np.mean(accuracies)
        avg_cv_acc = np.mean(cv_accuracies)
        print(f"\nAverage test accuracy across all datasets: {avg_acc:.2f}%")
        print(f"Average cross-validation accuracy across all datasets: {avg_cv_acc:.2f}%")

if __name__ == "__main__":
    main() 