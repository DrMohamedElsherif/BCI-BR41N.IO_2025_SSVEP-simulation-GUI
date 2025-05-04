#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SSVEP Visualization Script

This script visualizes all stages of SSVEP classification using the ssvep_classifier module.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ssvep_classifier import *
from scipy import signal
import os

# Create images directory if it doesn't exist
os.makedirs('images', exist_ok=True)

def visualize_raw_data(eeg_data, duration=5, fs=256):
    """Visualize raw EEG data"""
    samples = int(duration * fs)
    
    plt.figure(figsize=(15, 10))
    plt.suptitle(f"1. Raw EEG Data (First {duration} seconds)", fontsize=16)
    
    # Plot EEG channels (CH2-CH9)
    for i in range(1, 9):
        plt.subplot(9, 1, i)
        plt.plot(eeg_data[i, :samples])
        plt.ylabel(f"CH{i+1}")
        plt.xlim(0, samples)
        if i == 8:
            plt.xlabel("Samples")
    
    # Plot trigger channel
    plt.subplot(9, 1, 9)
    plt.plot(eeg_data[9, :samples], 'r')
    plt.ylabel("Trigger\n(CH10)")
    plt.xlim(0, samples)
    plt.xlabel("Samples")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig("images/1_raw_data.png", dpi=300, bbox_inches='tight')
    plt.close()

def visualize_filtering(eeg_data, fs=256):
    """Visualize filtering effects"""
    # Take a small segment from channel 2 (index 1)
    channel_idx = 1
    segment_duration = 3  # seconds
    segment = eeg_data[channel_idx, :int(segment_duration * fs)]
    
    # Apply filters
    bp_filtered = bandpass_filter(segment.reshape(1, -1), 6, 40, fs)[0]
    notch_filtered = notch_filter(bp_filtered.reshape(1, -1), 50, 30, fs)[0]
    
    plt.figure(figsize=(15, 10))
    plt.suptitle("2. Filtering Effects", fontsize=16)
    
    # Original signal
    plt.subplot(3, 1, 1)
    plt.plot(segment)
    plt.title("Original Signal (CH2)")
    plt.ylabel("Amplitude")
    
    # Bandpass filtered
    plt.subplot(3, 1, 2)
    plt.plot(bp_filtered)
    plt.title("Bandpass Filter Applied (6-40Hz)")
    plt.ylabel("Amplitude")
    
    # Notch filtered
    plt.subplot(3, 1, 3)
    plt.plot(notch_filtered)
    plt.title("Notch Filter Applied (50Hz)")
    plt.ylabel("Amplitude")
    plt.xlabel("Samples")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig("images/2_filtering.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also visualize in frequency domain
    plt.figure(figsize=(15, 10))
    plt.suptitle("2-2. Filtering Effects in Frequency Domain", fontsize=16)
    
    # Calculate PSDs
    f_orig, psd_orig = signal.welch(segment, fs, nperseg=int(fs))
    f_bp, psd_bp = signal.welch(bp_filtered, fs, nperseg=int(fs))
    f_notch, psd_notch = signal.welch(notch_filtered, fs, nperseg=int(fs))
    
    # Plot PSDs
    plt.subplot(3, 1, 1)
    plt.semilogy(f_orig, psd_orig)
    plt.title("Original Signal PSD")
    plt.ylabel("Log Power")
    plt.xlim(0, 80)
    
    plt.subplot(3, 1, 2)
    plt.semilogy(f_bp, psd_bp)
    plt.title("PSD After Bandpass Filter")
    plt.ylabel("Log Power")
    plt.xlim(0, 80)
    
    plt.subplot(3, 1, 3)
    plt.semilogy(f_notch, psd_notch)
    plt.title("PSD After Notch Filter")
    plt.ylabel("Log Power")
    plt.xlabel("Frequency (Hz)")
    plt.xlim(0, 80)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig("images/2_2_filtering_freq.png", dpi=300, bbox_inches='tight')
    plt.close()

def visualize_epochs(epochs, labels, n_epochs=3):
    """Visualize extracted epochs"""
    n_epochs = min(n_epochs, len(epochs))
    
    plt.figure(figsize=(15, 10))
    plt.suptitle("3. Extracted Epochs", fontsize=16)
    
    for e in range(n_epochs):
        epoch = epochs[e]
        label = labels[e]
        
        for i in range(8):
            plt.subplot(n_epochs, 8, e*8 + i + 1)
            plt.plot(epoch[i])
            
            if e == 0:
                plt.title(f"CH{i+2}")
            if i == 0:
                plt.ylabel(f"Epoch {e+1}\n(Label: {label})")
            
            plt.xticks([])
            if e == n_epochs-1 and i == 3:
                plt.xlabel("Time (samples)")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig("images/3_epochs.png", dpi=300, bbox_inches='tight')
    plt.close()

def visualize_filter_bank(epoch, stim_freqs, fs=256):
    """Visualize filter bank signals"""
    # Apply preprocessing
    processed_epoch = preprocess_eeg(epoch, True, fs)
    
    # Apply filter bank
    filter_bank_signals = apply_filter_bank(processed_epoch, fs)
    
    plt.figure(figsize=(15, 12))
    plt.suptitle("4. Filter Bank (FBCCA) - CH2 Signal", fontsize=16)
    
    # Plot original signal
    plt.subplot(len(filter_bank_signals)+1, 1, 1)
    plt.plot(processed_epoch[0])
    plt.title("Preprocessed Signal")
    plt.ylabel("Amplitude")
    
    # Plot each filtered signal
    for i, filtered_signal in enumerate(filter_bank_signals):
        plt.subplot(len(filter_bank_signals)+1, 1, i+2)
        plt.plot(filtered_signal[0])
        lowcut = 8 * (i + 1)
        highcut = min(90, fs/2 - 1)
        plt.title(f"Band {i+1}: {lowcut}-{highcut:.1f} Hz")
        plt.ylabel("Amplitude")
        
        if i == len(filter_bank_signals)-1:
            plt.xlabel("Time (samples)")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig("images/4_filter_bank.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also visualize in frequency domain
    plt.figure(figsize=(15, 12))
    plt.suptitle("4-2. Filter Bank Frequency Response (CH2)", fontsize=16)
    
    # Calculate and plot PSD of original signal
    f_orig, psd_orig = signal.welch(processed_epoch[0], fs, nperseg=int(fs))
    plt.subplot(len(filter_bank_signals)+1, 1, 1)
    plt.semilogy(f_orig, psd_orig)
    plt.title("PSD of Preprocessed Signal")
    plt.ylabel("Log Power")
    plt.xlim(0, 80)
    
    # Mark stimulus frequencies
    for freq in stim_freqs:
        plt.axvline(x=freq, color='r', linestyle='--', alpha=0.7)
    
    # Calculate and plot PSDs for each filtered signal
    for i, filtered_signal in enumerate(filter_bank_signals):
        f, psd = signal.welch(filtered_signal[0], fs, nperseg=int(fs))
        plt.subplot(len(filter_bank_signals)+1, 1, i+2)
        plt.semilogy(f, psd)
        lowcut = 8 * (i + 1)
        highcut = min(90, fs/2 - 1)
        plt.title(f"Band {i+1}: {lowcut}-{highcut:.1f} Hz")
        plt.ylabel("Log Power")
        plt.xlim(0, 80)
        
        # Mark stimulus frequencies
        for freq in stim_freqs:
            plt.axvline(x=freq, color='r', linestyle='--', alpha=0.7)
            
        if i == len(filter_bank_signals)-1:
            plt.xlabel("Frequency (Hz)")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig("images/4_2_filter_bank_freq.png", dpi=300, bbox_inches='tight')
    plt.close()

def visualize_fbcca_features(epoch, stim_freqs, fs=256):
    """Visualize FBCCA features for a single epoch"""
    # Preprocess epoch
    processed_epoch = preprocess_eeg(epoch, True, fs)
    
    # Get FBCCA features
    fbcca_feats = fbcca_features(processed_epoch, stim_freqs, fs)
    
    # The fbcca_features function returns 3 features per frequency
    # Let's extract just the first feature for each frequency for visualization
    n_freqs = len(stim_freqs)
    weighted_corr_features = fbcca_feats[:n_freqs]  # First feature for each frequency
    
    plt.figure(figsize=(12, 6))
    plt.suptitle("5. FBCCA Features", fontsize=16)
    
    # Plot features as bar graph
    plt.bar(range(len(stim_freqs)), weighted_corr_features)
    plt.xlabel("Stimulus Frequency Index")
    plt.ylabel("Weighted Canonical Correlation Coefficient")
    plt.xticks(range(len(stim_freqs)), [f"{freq} Hz" for freq in stim_freqs])
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig("images/5_fbcca_features.png", dpi=300, bbox_inches='tight')
    plt.close()

def visualize_psd_features(epoch, fs=256):
    """Visualize PSD features for a single epoch"""
    # Preprocess epoch
    processed_epoch = preprocess_eeg(epoch, True, fs)
    
    plt.figure(figsize=(15, 10))
    plt.suptitle("6. PSD Features (Welch Method)", fontsize=16)
    
    # Calculate and plot PSD for each channel
    for i in range(8):
        plt.subplot(4, 2, i+1)
        f, Pxx = signal.welch(processed_epoch[i], fs=fs, nperseg=256)
        idx = np.logical_and(f >= 5, f <= 40)
        plt.semilogy(f[idx], Pxx[idx])
        plt.grid(True, alpha=0.3)
        plt.title(f"CH{i+2}")
        
        if i % 2 == 0:
            plt.ylabel("Log Power")
        if i >= 6:
            plt.xlabel("Frequency (Hz)")
            
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig("images/6_psd_features.png", dpi=300, bbox_inches='tight')
    plt.close()

def visualize_combined_features(epochs, stim_freqs, fs=256):
    """Visualize combined FBCCA and PSD features for all epochs"""
    # Preprocess first epoch
    processed_epoch = preprocess_eeg(epochs[0], True, fs)
    
    # Get FBCCA features
    fbcca_feats = fbcca_features(processed_epoch, stim_freqs, fs)
    
    # The fbcca_features function returns 3 features per frequency
    # Let's extract just the first feature for each frequency for visualization
    n_freqs = len(stim_freqs)
    weighted_corr_features = fbcca_feats[:n_freqs]  # First feature for each frequency
    
    # Get PSD features
    psd_feats = psd_features(processed_epoch, stim_freqs, fs)
    
    plt.figure(figsize=(18, 12))
    plt.suptitle("7. Combined Features Visualization", fontsize=16)
    
    # Plot FBCCA features
    plt.subplot(3, 1, 1)
    plt.bar(range(len(stim_freqs)), weighted_corr_features)
    plt.title("FBCCA Features")
    plt.xlabel("Stimulus Frequency Index")
    plt.ylabel("Weighted CCA Coefficient")
    plt.xticks(range(len(stim_freqs)), [f"{freq} Hz" for freq in stim_freqs])
    plt.grid(axis='y', alpha=0.3)
    
    # Plot PSD features
    plt.subplot(3, 1, 2)
    plt.bar(range(len(stim_freqs)), psd_feats)
    plt.title("PSD Features")
    plt.xlabel("Stimulus Frequency Index")
    plt.ylabel("Power Spectral Density")
    plt.xticks(range(len(stim_freqs)), [f"{freq} Hz" for freq in stim_freqs])
    plt.grid(axis='y', alpha=0.3)
    
    # Plot combined features (simple addition for visualization)
    plt.subplot(3, 1, 3)
    
    # Normalize features to compare them
    fbcca_norm = weighted_corr_features / np.max(weighted_corr_features) if np.max(weighted_corr_features) > 0 else weighted_corr_features
    psd_norm = psd_feats / np.max(psd_feats) if np.max(psd_feats) > 0 else psd_feats
    
    # Combine features (simple weighted sum)
    combined = 0.6 * fbcca_norm + 0.4 * psd_norm
    
    plt.bar(range(len(stim_freqs)), combined)
    plt.title("Combined Features (0.6*FBCCA + 0.4*PSD)")
    plt.xlabel("Stimulus Frequency Index")
    plt.ylabel("Combined Feature Value")
    plt.xticks(range(len(stim_freqs)), [f"{freq} Hz" for freq in stim_freqs])
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig("images/7_1_combined_features.png", dpi=300, bbox_inches='tight')
    
    # Plot different epochs comparison
    if len(epochs) >= 3:
        plt.figure(figsize=(18, 12))
        plt.suptitle("7.2 Comparison of Different Epochs", fontsize=16)
        
        # Process 3 different epochs
        for i, idx in enumerate([0, len(epochs)//2, len(epochs)-1]):
            # Preprocess epoch
            processed_epoch = preprocess_eeg(epochs[idx], True, fs)
            
            # Get features
            fbcca = fbcca_features(processed_epoch, stim_freqs, fs)
            fbcca_normalized = fbcca[:n_freqs]  # First feature for each frequency
            psd = psd_features(processed_epoch, stim_freqs, fs)
            
            plt.subplot(3, 1, i+1)
            
            # Normalize for visualization
            fbcca_norm = fbcca_normalized / np.max(fbcca_normalized) if np.max(fbcca_normalized) > 0 else fbcca_normalized
            psd_norm = psd / np.max(psd) if np.max(psd) > 0 else psd
            
            # Plot both features
            x = np.arange(len(stim_freqs))
            width = 0.35
            
            plt.bar(x - width/2, fbcca_norm, width, label='FBCCA')
            plt.bar(x + width/2, psd_norm, width, label='PSD')
            
            plt.title(f"Epoch {idx+1}")
            plt.xlabel("Stimulus Frequency")
            plt.ylabel("Normalized Feature Value")
            plt.xticks(range(len(stim_freqs)), [f"{freq} Hz" for freq in stim_freqs])
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig("images/7_2_combined_features.png", dpi=300, bbox_inches='tight')
        
        # Plot 3D feature space
        plt.figure(figsize=(10, 8))
        ax = plt.axes(projection='3d')
        plt.suptitle("7.3 3D Feature Space", fontsize=16)
        
        # Get features for all epochs
        for i, epoch in enumerate(epochs):
            # Preprocess epoch
            processed_epoch = preprocess_eeg(epoch, True, fs)
            
            # Get features
            fbcca = fbcca_features(processed_epoch, stim_freqs, fs)
            fbcca_1 = fbcca[0]  # Feature for first frequency
            fbcca_2 = fbcca[1]  # Feature for second frequency
            psd_val = np.max(psd_features(processed_epoch, stim_freqs, fs))
            
            # Plot in 3D
            ax.scatter(fbcca_1, fbcca_2, psd_val, marker='o')
        
        ax.set_xlabel('FBCCA Feature 1')
        ax.set_ylabel('FBCCA Feature 2')
        ax.set_zlabel('PSD Feature')
        
        plt.tight_layout()
        plt.savefig("images/7_3_combined_features.png", dpi=300, bbox_inches='tight')
    
    plt.close('all')

def visualize_classification_results(labels, y_test, y_pred, stim_freqs):
    """Visualize classification results"""
    # Get confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Print the unique labels for debugging
    unique_test_labels = np.unique(y_test)
    print(f"Unique labels in test set: {unique_test_labels}")
    
    # Create a full confusion matrix with all classes
    all_classes = np.arange(1, len(stim_freqs) + 1)  # Class numbers 1-4 (for 15, 12, 10, 9 Hz)
    n_classes = len(all_classes)
    
    # Initialize a new confusion matrix with zeros
    full_conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    # Map the existing confusion matrix to our full one
    # We need to map class values to indices (0-based)
    for i, true_label in enumerate(unique_test_labels):
        if true_label > 0 and true_label <= n_classes:  # Skip class 0 (if present)
            true_idx = true_label - 1  # Convert class number to 0-based index
            for j, pred_label in enumerate(unique_test_labels):
                if pred_label > 0 and pred_label <= n_classes:
                    pred_idx = pred_label - 1  # Convert class number to 0-based index
                    # Find value in original confusion matrix
                    original_idx_true = np.where(unique_test_labels == true_label)[0][0]
                    original_idx_pred = np.where(unique_test_labels == pred_label)[0][0]
                    count = conf_matrix[original_idx_true, original_idx_pred]
                    full_conf_matrix[true_idx, pred_idx] = count
    
    # Create class names for all frequencies
    class_names = [f"Freq {freq} Hz" for freq in stim_freqs]
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.suptitle("8. Classification Results", fontsize=16)
    
    sns.heatmap(full_conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2%})')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig("images/8_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate overall accuracy for visualization
    plt.figure(figsize=(10, 6))
    plt.suptitle("9. Classification Accuracy", fontsize=16)
    
    plt.bar(['Overall Accuracy'], [accuracy])
    plt.text(0, accuracy + 0.01, f'{accuracy:.1%}', ha='center', va='bottom', fontsize=12)
    
    plt.ylabel('Accuracy')
    plt.title(f'SSVEP Classification Accuracy: {accuracy:.2%}')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig("images/9_accuracy.png", dpi=300, bbox_inches='tight')
    plt.close()

def preprocess_eeg(epoch, apply_filters=True, fs=256):
    """Apply preprocessing steps to an EEG epoch"""
    # Default to showing a 3-second epoch
    epoch_samples = min(3 * fs, epoch.shape[1])
    processed_epoch = epoch[:, :epoch_samples].copy()
    
    if apply_filters:
        # Apply bandpass filter between 5-45 Hz
        sos = signal.butter(4, [5, 45], 'bandpass', fs=fs, output='sos')
        for ch in range(processed_epoch.shape[0]):
            processed_epoch[ch] = signal.sosfilt(sos, processed_epoch[ch])
        
        # Apply notch filters (60 Hz, 120 Hz, etc.)
        for freq in [50, 60, 120]:
            sos = signal.butter(4, [freq-2, freq+2], 'bandstop', fs=fs, output='sos')
            for ch in range(processed_epoch.shape[0]):
                processed_epoch[ch] = signal.sosfilt(sos, processed_epoch[ch])
    
    return processed_epoch

def psd_features(epoch, stim_freqs, fs=256):
    """Calculate power spectral density features for the given frequencies"""
    # Calculate PSD using Welch's method
    f, Pxx = signal.welch(epoch, fs=fs, nperseg=512, noverlap=256)
    
    # Extract power at each stimulus frequency (average across channels)
    psd_features = []
    for freq in stim_freqs:
        # Find closest frequency bin
        idx = np.argmin(np.abs(f - freq))
        
        # Get average power across channels
        power = np.mean(Pxx[:, idx])
        psd_features.append(power)
    
    return np.array(psd_features)

def main():
    # Set the stimulus frequencies
    stim_freqs = [15, 12, 10, 9]  # Corrected stimulus frequencies (top, right, bottom, left)
    
    # Dataset files
    dataset_files = [
        "data/subject_1_fvep_led_training_1.mat",
        "data/subject_1_fvep_led_training_2.mat", 
        "data/subject_2_fvep_led_training_1.mat",
        "data/subject_2_fvep_led_training_2.mat"
    ]
    
    # Load first dataset for visualization
    print("Loading data...")
    file_path = dataset_files[0]
    eeg_data = load_data(file_path)
    
    if eeg_data is None:
        print("Failed to load data. Exiting.")
        return
    
    print(f"Data loaded with shape: {eeg_data.shape}")
    
    # 1. Visualize raw data
    print("1. Visualizing raw data...")
    visualize_raw_data(eeg_data)
    
    # 2. Visualize filtering
    print("2. Visualizing filtering effects...")
    visualize_filtering(eeg_data)
    
    # 3. Extract and visualize epochs
    print("3. Extracting and visualizing epochs...")
    epochs, labels = extract_epochs(eeg_data, eeg_data[9], epoch_duration=3, stride=1, fs=256)
    visualize_epochs(epochs, labels)
    
    # 4. Visualize filter bank
    print("4. Visualizing filter bank...")
    visualize_filter_bank(epochs[0], stim_freqs)
    
    # 5. Visualize FBCCA features
    print("5. Visualizing FBCCA features...")
    visualize_fbcca_features(epochs[0], stim_freqs)
    
    # 6. Visualize PSD features
    print("6. Visualizing PSD features...")
    visualize_psd_features(epochs[0])
    
    # 7. Visualize combined features
    print("7. Visualizing combined features...")
    visualize_combined_features(epochs, stim_freqs)
    
    # Now process all datasets to ensure all frequencies are represented
    print("Extracting features from all datasets...")
    all_epochs = []
    all_labels = []
    
    for file_path in dataset_files:
        try:
            data = load_data(file_path)
            if data is not None:
                print(f"Processing {file_path}...")
                ep, lab = extract_epochs(data, data[9], epoch_duration=3, stride=1, fs=256)
                
                # Filter out invalid labels (only keep 1-4 corresponding to our frequencies)
                valid_indices = [i for i, label in enumerate(lab) if 1 <= label <= 4]
                if valid_indices:
                    filtered_epochs = [ep[i] for i in valid_indices]
                    filtered_labels = [lab[i] for i in valid_indices]
                    
                    all_epochs.extend(filtered_epochs)
                    all_labels.extend(filtered_labels)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Extract features for all epochs
    print(f"Total epochs collected: {len(all_epochs)}")
    print(f"Unique labels: {np.unique(all_labels)}")
    
    print("Extracting features...")
    features = extract_features(all_epochs, stim_freqs)
    
    # Train classifier
    print("Training classifier...")
    model, X_train, X_test, y_train, y_test = train_svm_classifier(features, all_labels)
    
    # Evaluate model
    print("Evaluating model...")
    accuracy, conf_matrix, y_pred = evaluate_model(model, X_test, y_test)
    
    # 8 & 9. Visualize classification results
    print("8. Visualizing classification results...")
    visualize_classification_results(all_labels, y_test, y_pred, stim_freqs)
    
    # Print final results
    print("\n=== FINAL RESULTS ===")
    print(f"Total epochs: {len(all_epochs)}")
    print(f"Training set size: {len(y_train)}")
    print(f"Test set size: {len(y_test)}")
    print(f"Classification accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print("Confusion Matrix:")
    print(conf_matrix)
    
    print("\nAll visualizations completed. Check the output images.")

if __name__ == "__main__":
    main() 