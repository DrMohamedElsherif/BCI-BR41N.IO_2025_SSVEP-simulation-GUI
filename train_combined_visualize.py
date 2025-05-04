#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SSVEP Combined Training and Visualization

This script trains a classifier on all 4 datasets combined, then predicts on each dataset
separately and visualizes the results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
import os
from ssvep_classifier import load_data, preprocess_eeg, extract_features, canonical_correlation
from scipy import signal
import time
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import seaborn as sns
import pickle

# Create images directory if it doesn't exist
os.makedirs('images', exist_ok=True)
os.makedirs('videos', exist_ok=True)

# Constants
fs = 256  # Sampling rate (Hz)
stim_freqs = [15, 12, 10, 9]  # Stimulus frequencies (top, right, bottom, left)
freq_names = ["15 Hz (Top)", "12 Hz (Right)", "10 Hz (Bottom)", "9 Hz (Left)"]

def create_reference_signals(freqs, fs, T):
    """Create reference signals for FBCCA"""
    t = np.arange(0, T, 1/fs)
    Y = {}
    
    for freq in freqs:
        signals = []
        
        # Add harmonics (sine and cosine)
        for h in range(1, 6):  # Use 5 harmonics
            signals.append(np.sin(2 * np.pi * h * freq * t))
            signals.append(np.cos(2 * np.pi * h * freq * t))
        
        Y[freq] = np.vstack(signals)
    
    return Y

def find_trigger_intervals(trigger_channel):
    """Find start and end points of each trigger in the data"""
    # Find trigger onsets (0->1 transitions)
    trigger_onsets = np.where(np.diff(np.concatenate(([0], trigger_channel))) == 1)[0]
    # Find trigger offsets (1->0 transitions)
    trigger_offsets = np.where(np.diff(np.concatenate((trigger_channel, [0]))) == -1)[0]
    
    # Ensure equal number of onsets and offsets
    min_len = min(len(trigger_onsets), len(trigger_offsets))
    return np.vstack([trigger_onsets[:min_len], trigger_offsets[:min_len]]).T

def extract_dataset_features(file_path, dataset_name):
    """Extract features from a dataset"""
    print(f"\n=== Extracting features from {dataset_name} ===")
    
    # Load data
    print(f"Loading data from {file_path}...")
    eeg_data = load_data(file_path)
    
    if eeg_data is None:
        print(f"Failed to load data from {file_path}")
        return None, None
    
    # Find trigger intervals
    trigger_channel = eeg_data[9]
    trigger_intervals = find_trigger_intervals(trigger_channel)
    
    print(f"Found {len(trigger_intervals)} trigger intervals")
    
    # Expected pattern is 15Hz, 12Hz, 10Hz, 9Hz repeated 5 times
    expected_labels = np.array([1, 2, 3, 4] * 5)  # Class labels 1-4 repeated 5 times
    
    # Truncate to match actual number of intervals found
    if len(trigger_intervals) < len(expected_labels):
        print(f"Warning: Found fewer trigger intervals ({len(trigger_intervals)}) than expected ({len(expected_labels)})")
        expected_labels = expected_labels[:len(trigger_intervals)]
    elif len(trigger_intervals) > len(expected_labels):
        print(f"Warning: Found more trigger intervals ({len(trigger_intervals)}) than expected ({len(expected_labels)})")
        trigger_intervals = trigger_intervals[:len(expected_labels)]
    
    # Extract epochs and features
    all_features = []
    all_labels = []
    epoch_duration = 3  # seconds
    epoch_samples = int(epoch_duration * fs)
    
    for i, (start, end) in enumerate(trigger_intervals):
        # Skip if interval is too short
        if end - start < epoch_samples:
            continue
        
        # Get expected label (1-4)
        expected_label = expected_labels[i]
        
        # Extract EEG signals for this interval (CH2-CH9)
        eeg_signals = eeg_data[1:9, start:end]
        
        # Preprocess the signals
        preprocessed_signals = preprocess_eeg(eeg_signals, apply_notch=True, fs=fs)
        
        # Extract multiple windows from each interval
        window_step = fs // 2  # 0.5 second step
        
        for j in range(0, preprocessed_signals.shape[1] - epoch_samples + 1, window_step):
            # Extract window
            window = preprocessed_signals[:, j:j+epoch_samples]
            
            # Extract features
            features = extract_features([window], stim_freqs, fs)
            
            # Add to collection
            all_features.append(features[0])
            all_labels.append(expected_label)
    
    print(f"Extracted {len(all_features)} feature vectors")
    print(f"Labels distribution: {np.bincount(all_labels)[1:]}")
    
    return np.array(all_features), np.array(all_labels)

def train_combined_classifier():
    """Train a classifier on all 4 datasets combined"""
    # Dataset files and names
    datasets = [
        {"path": "data/subject_1_fvep_led_training_1.mat", "name": "Subject 1 Training 1"},
        {"path": "data/subject_1_fvep_led_training_2.mat", "name": "Subject 1 Training 2"},
        {"path": "data/subject_2_fvep_led_training_1.mat", "name": "Subject 2 Training 1"},
        {"path": "data/subject_2_fvep_led_training_2.mat", "name": "Subject 2 Training 2"}
    ]
    
    # Extract features from all datasets
    all_features = []
    all_labels = []
    
    for dataset in datasets:
        features, labels = extract_dataset_features(dataset["path"], dataset["name"])
        if features is not None and labels is not None:
            all_features.append(features)
            all_labels.append(labels)
    
    # Combine features and labels
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    
    print(f"\n=== Training combined classifier ===")
    print(f"Total samples: {len(X)}")
    print(f"Labels distribution: {np.bincount(y)[1:]}")
    
    # Create a pipeline with preprocessing and classifiers
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train individual classifiers
    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    
    print("Training SVM...")
    svm.fit(X_scaled, y)
    
    print("Training Random Forest...")
    rf.fit(X_scaled, y)
    
    print("Training Neural Network...")
    mlp.fit(X_scaled, y)
    
    # Evaluate each classifier
    svm_score = svm.score(X_scaled, y)
    rf_score = rf.score(X_scaled, y)
    mlp_score = mlp.score(X_scaled, y)
    
    print(f"SVM accuracy: {svm_score:.4f}")
    print(f"Random Forest accuracy: {rf_score:.4f}")
    print(f"Neural Network accuracy: {mlp_score:.4f}")
    
    # Choose the best classifier
    best_clf = None
    best_score = 0
    
    if svm_score > best_score:
        best_clf = svm
        best_score = svm_score
        best_name = "SVM"
    
    if rf_score > best_score:
        best_clf = rf
        best_score = rf_score
        best_name = "Random Forest"
    
    if mlp_score > best_score:
        best_clf = mlp
        best_score = mlp_score
        best_name = "Neural Network"
    
    print(f"Best classifier: {best_name} (accuracy: {best_score:.4f})")
    
    # Create and save model package
    model_package = {
        'scaler': scaler,
        'classifier': best_clf,
        'best_classifier_name': best_name,
        'training_accuracy': best_score
    }
    
    # Save the model
    with open('trained_combined_model.pkl', 'wb') as f:
        pickle.dump(model_package, f)
    
    print("Model saved to trained_combined_model.pkl")
    
    return model_package

def classify_epoch_with_model(eeg_epoch, model_package, fs=256):
    """Classify an epoch of EEG data using the trained model"""
    # Preprocess the epoch
    # Extract features
    features = extract_features([eeg_epoch], stim_freqs, fs)
    
    # Scale features
    scaler = model_package['scaler']
    classifier = model_package['classifier']
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    predicted_class = classifier.predict(features_scaled)[0]
    probabilities = classifier.predict_proba(features_scaled)[0]
    
    # Convert class (1-4) to frequency
    predicted_freq = stim_freqs[predicted_class-1]
    confidence = probabilities[predicted_class-1]
    
    return predicted_freq, confidence, probabilities

def process_dataset_with_model(file_path, dataset_name, model_package):
    """Process a dataset and classify with the trained model"""
    print(f"\n=== Processing {dataset_name} with combined model ===")
    
    # Load data
    print(f"Loading data from {file_path}...")
    eeg_data = load_data(file_path)
    
    if eeg_data is None:
        print(f"Failed to load data from {file_path}")
        return None
    
    # Find trigger intervals
    trigger_channel = eeg_data[9]
    trigger_intervals = find_trigger_intervals(trigger_channel)
    
    print(f"Found {len(trigger_intervals)} trigger intervals")
    
    # Expected pattern is 15Hz, 12Hz, 10Hz, 9Hz repeated 5 times
    expected_labels = np.array([1, 2, 3, 4] * 5)  # Class labels 1-4 repeated 5 times
    
    # Truncate to match actual number of intervals found
    if len(trigger_intervals) < len(expected_labels):
        print(f"Warning: Found fewer trigger intervals ({len(trigger_intervals)}) than expected ({len(expected_labels)})")
        expected_labels = expected_labels[:len(trigger_intervals)]
    elif len(trigger_intervals) > len(expected_labels):
        print(f"Warning: Found more trigger intervals ({len(trigger_intervals)}) than expected ({len(expected_labels)})")
        trigger_intervals = trigger_intervals[:len(expected_labels)]
    
    # Extract expected frequencies
    expected_freqs = [stim_freqs[label-1] for label in expected_labels]
    
    # Set up plots
    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.4)
    
    # Create subplots
    ax_eeg = plt.subplot(gs[0])
    ax_proba = plt.subplot(gs[1])
    ax_correct = plt.subplot(gs[2])
    
    # Initialize plots with more spacing
    ax_eeg.set_title(f"SSVEP Classification: {dataset_name}", pad=10)
    ax_eeg.set_ylabel("EEG Amplitude")
    ax_eeg.set_xlabel("Time (samples)")
    
    ax_proba.set_title("Classification Probabilities", pad=10)
    ax_proba.set_ylabel("Probability")
    ax_proba.set_ylim(0, 1)
    
    ax_correct.set_title("Classification Accuracy", pad=10)
    ax_correct.set_ylabel("Correct")
    ax_correct.set_ylim(-0.1, 1.1)
    ax_correct.set_yticks([0, 1])
    ax_correct.set_yticklabels(["Incorrect", "Correct"])
    
    # Process each interval
    all_results = []
    epoch_duration = 3  # seconds
    epoch_samples = int(epoch_duration * fs)
    window_step = fs // 4  # Update every 0.25 seconds
    
    for i, (start, end) in enumerate(trigger_intervals):
        # Skip if interval is too short
        if end - start < epoch_samples:
            continue
        
        # Get expected frequency
        expected_freq = expected_freqs[i]
        expected_class = expected_labels[i]
        
        # Extract EEG signals for this interval (CH2-CH9)
        eeg_signals = eeg_data[1:9, start:end]
        
        # Preprocess the signals
        preprocessed_signals = preprocess_eeg(eeg_signals, apply_notch=True, fs=fs)
        
        # Sliding window classification
        interval_results = []
        
        for j in range(0, preprocessed_signals.shape[1] - epoch_samples + 1, window_step):
            # Extract window
            window = preprocessed_signals[:, j:j+epoch_samples]
            
            # Classify with model
            pred_freq, confidence, probabilities = classify_epoch_with_model(window, model_package, fs)
            
            # Check if prediction is correct
            is_correct = pred_freq == expected_freq
            
            # Store results
            interval_results.append({
                'time': start + j,
                'expected_freq': expected_freq,
                'predicted_freq': pred_freq,
                'confidence': confidence,
                'probabilities': probabilities,
                'is_correct': is_correct
            })
        
        # Add results for this interval
        all_results.extend(interval_results)
    
    # Extract result arrays for plotting
    times = [r['time'] for r in all_results]
    expected_freqs = [r['expected_freq'] for r in all_results]
    predicted_freqs = [r['predicted_freq'] for r in all_results]
    confidences = [r['confidence'] for r in all_results]
    all_probas = np.array([r['probabilities'] for r in all_results])
    correct_predictions = [1 if r['is_correct'] else 0 for r in all_results]
    
    # Create plots
    ax_eeg.plot(times, predicted_freqs, 'r-', label='Predicted Frequency')
    ax_eeg.plot(times, expected_freqs, 'b--', label='Expected Frequency')
    ax_eeg.legend()
    
    # Plot probabilities
    for i in range(4):  # 4 classes
        ax_proba.plot(times, all_probas[:, i], label=f"Class {i+1} ({stim_freqs[i]} Hz)")
    ax_proba.legend()
    
    # Plot correct/incorrect
    ax_correct.plot(times, correct_predictions, 'g-')
    
    # Calculate accuracy
    accuracy = np.mean(correct_predictions)
    plt.figure(fig.number)  # Ensure we're working with the correct figure
    
    # Add model info to title
    classifier_name = model_package['best_classifier_name']
    training_acc = model_package['training_accuracy']
    
    plt.suptitle(
        f"SSVEP Classification: {dataset_name} using Combined Model ({classifier_name})\n"
        f"Accuracy: {accuracy:.2%} (Total Samples: {len(all_results)}, Window: 3s, Training Acc: {training_acc:.2%})", 
        fontsize=16, y=0.98
    )
    
    # Save figure
    output_file = f"images/combined_prediction_{dataset_name.replace(' ', '_').lower()}.png"
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for suptitle
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Total samples: {len(all_results)}")
    print(f"Visualization saved to {output_file}")
    
    # Create confusion matrix
    create_confusion_matrix(predicted_freqs, expected_freqs, dataset_name, model_package)
    
    return accuracy

def create_confusion_matrix(predicted_freqs, expected_freqs, dataset_name, model_package):
    """Create a confusion matrix for the predictions"""
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    
    # Convert frequencies to class indices (0-3)
    pred_indices = [stim_freqs.index(freq) for freq in predicted_freqs]
    expected_indices = [stim_freqs.index(freq) for freq in expected_freqs]
    
    # Calculate confusion matrix
    cm = confusion_matrix(expected_indices, pred_indices)
    
    # Calculate the total number of samples
    total_samples = len(predicted_freqs)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=freq_names, yticklabels=freq_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Calculate class distribution
    class_counts = []
    for i in range(len(stim_freqs)):
        count = np.sum(np.array(expected_indices) == i)
        class_counts.append(count)
    
    # Get model info
    classifier_name = model_package['best_classifier_name']
    
    # Create a more informative title
    plt.title(
        f'Combined Model Classification Confusion Matrix: {dataset_name}\n'
        f'Model: {classifier_name}, Total Samples: {total_samples} (Window Size: 3s, Step: 0.25s)\n'
        f'Class Distribution: {freq_names[0]}: {class_counts[0]}, {freq_names[1]}: {class_counts[1]}, '
        f'{freq_names[2]}: {class_counts[2]}, {freq_names[3]}: {class_counts[3]}',
        pad=20
    )
    
    # Save figure
    output_file = f"images/combined_confusion_matrix_{dataset_name.replace(' ', '_').lower()}.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {output_file}")

def visualize_accuracy_results(results):
    """Create a bar chart of accuracies"""
    # Filter out None values
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    # Calculate average
    average = np.mean(list(valid_results.values()))
    valid_results["Average"] = average
    
    # Create figure
    plt.figure(figsize=(10, 6))
    names = list(valid_results.keys())
    values = [v * 100 for v in valid_results.values()]
    
    # Get colors (highlight Subject 2 in a different color)
    colors = ['blue' if 'Subject 1' in name else 'green' if 'Subject 2' in name else 'red' for name in names]
    
    bars = plt.bar(names, values, color=colors)
    
    # Add accuracy values
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{values[i]:.1f}%', ha='center', va='bottom')
    
    plt.title("Combined Model Classification Accuracy")
    plt.xlabel("Dataset")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 105)
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    # Save figure
    plt.tight_layout()
    plt.savefig("images/combined_classification_accuracy.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n=== Overall Combined Model Classification Results ===")
    for name, accuracy in valid_results.items():
        print(f"{name}: {accuracy:.1%}")

def main():
    # Train combined model
    model_package = train_combined_classifier()
    
    # Dataset files and names
    datasets = [
        {"path": "data/subject_1_fvep_led_training_1.mat", "name": "Subject 1 Training 1"},
        {"path": "data/subject_1_fvep_led_training_2.mat", "name": "Subject 1 Training 2"},
        {"path": "data/subject_2_fvep_led_training_1.mat", "name": "Subject 2 Training 1"},
        {"path": "data/subject_2_fvep_led_training_2.mat", "name": "Subject 2 Training 2"}
    ]
    
    # Process each dataset
    results = {}
    for dataset in datasets:
        accuracy = process_dataset_with_model(dataset["path"], dataset["name"], model_package)
        results[dataset["name"]] = accuracy
    
    # Create summary visualization
    visualize_accuracy_results(results)

if __name__ == "__main__":
    main() 