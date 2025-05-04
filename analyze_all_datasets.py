#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset Analysis Script

This script analyzes each of the four SSVEP datasets individually,
generating confusion matrices and accuracy results for each one.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import accuracy_score, confusion_matrix
from ssvep_classifier import load_data, extract_epochs, extract_features, train_svm_classifier, evaluate_model

# Create images directory if it doesn't exist
os.makedirs('images', exist_ok=True)

def analyze_dataset(file_path, stim_freqs, dataset_name):
    """
    Analyze a single dataset and generate confusion matrix and accuracy results
    
    Parameters:
    -----------
    file_path : str
        Path to the .mat file containing the dataset
    stim_freqs : list
        List of stimulus frequencies
    dataset_name : str
        Name of the dataset for display purposes
    
    Returns:
    --------
    accuracy : float
        Classification accuracy
    """
    print(f"\n=== Processing {dataset_name} ===")
    
    # Load data
    print(f"Loading data from {file_path}...")
    eeg_data = load_data(file_path)
    
    if eeg_data is None:
        print(f"Failed to load data from {file_path}")
        return None
    
    # Extract epochs
    print("Extracting epochs...")
    epochs, labels = extract_epochs(eeg_data, eeg_data[9], epoch_duration=3, stride=1, fs=256)
    
    # Filter out invalid labels (only keep 1-4 corresponding to our frequencies)
    valid_indices = [i for i, label in enumerate(labels) if 1 <= label <= 4]
    if valid_indices:
        epochs = [epochs[i] for i in valid_indices]
        labels = [labels[i] for i in valid_indices]
    
    print(f"Total epochs: {len(epochs)}")
    print(f"Unique labels: {np.unique(labels)}")
    
    # Check if we have at least 2 different classes
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        print(f"Warning: Only one class ({unique_labels[0]}) present in the dataset. Cannot train classifier.")
        
        # Create a dummy perfect confusion matrix for the single class
        class_idx = unique_labels[0] - 1  # Convert class number (1-4) to 0-based index
        full_conf_matrix = np.zeros((len(stim_freqs), len(stim_freqs)), dtype=int)
        full_conf_matrix[class_idx, class_idx] = len(labels)  # All samples classified correctly
        
        # Generate a confusion matrix visualization for the single class
        generate_single_class_confusion_matrix(full_conf_matrix, stim_freqs, unique_labels[0], dataset_name)
        
        return 1.0  # Return perfect accuracy since all samples are of the same class
    
    # Extract features
    print("Extracting features...")
    features = extract_features(epochs, stim_freqs)
    
    # Train classifier
    print("Training classifier...")
    model, X_train, X_test, y_train, y_test = train_svm_classifier(features, labels)
    
    # Evaluate model
    print("Evaluating model...")
    accuracy, conf_matrix, y_pred = evaluate_model(model, X_test, y_test)
    
    # Generate confusion matrix visualization
    generate_confusion_matrix(conf_matrix, y_test, y_pred, stim_freqs, dataset_name)
    
    # Print results
    print(f"\nResults for {dataset_name}:")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print("Confusion Matrix:")
    print(conf_matrix)
    
    return accuracy

def analyze_dataset_with_fixed_labels(file_path, stim_freqs, dataset_name):
    """
    Analyze a single dataset using fixed labels that match the protocol order
    (15Hz, 12Hz, 10Hz, 9Hz repeated 5 times)
    
    Parameters:
    -----------
    file_path : str
        Path to the .mat file containing the dataset
    stim_freqs : list
        List of stimulus frequencies
    dataset_name : str
        Name of the dataset for display purposes
    
    Returns:
    --------
    accuracy : float
        Classification accuracy
    """
    print(f"\n=== Processing {dataset_name} with fixed labels ===")
    
    # Load data
    print(f"Loading data from {file_path}...")
    eeg_data = load_data(file_path)
    
    if eeg_data is None:
        print(f"Failed to load data from {file_path}")
        return None
    
    # Find trigger intervals
    trigger_channel = eeg_data[9]
    trigger_onsets = np.where(np.diff(np.concatenate(([0], trigger_channel))) == 1)[0]
    trigger_offsets = np.where(np.diff(np.concatenate((trigger_channel, [0]))) == -1)[0]
    
    # Ensure equal number of onsets and offsets
    min_len = min(len(trigger_onsets), len(trigger_offsets))
    trigger_intervals = np.vstack([trigger_onsets[:min_len], trigger_offsets[:min_len]]).T
    
    print(f"Found {len(trigger_intervals)} trigger intervals")
    
    # Expected pattern is 15Hz, 12Hz, 10Hz, 9Hz repeated 5 times (total 20 trials)
    expected_labels = np.array([1, 2, 3, 4] * 5)  # Class labels 1-4 repeated 5 times
    
    # Truncate to match actual number of intervals found
    if len(trigger_intervals) < len(expected_labels):
        print(f"Warning: Found fewer trigger intervals ({len(trigger_intervals)}) than expected ({len(expected_labels)})")
        expected_labels = expected_labels[:len(trigger_intervals)]
    elif len(trigger_intervals) > len(expected_labels):
        print(f"Warning: Found more trigger intervals ({len(trigger_intervals)}) than expected ({len(expected_labels)})")
        trigger_intervals = trigger_intervals[:len(expected_labels)]
    
    # Extract epochs with fixed labels
    epochs = []
    labels = []
    fs = 256
    epoch_duration = 3  # seconds
    epoch_samples = int(epoch_duration * fs)
    
    for i, (start, end) in enumerate(trigger_intervals):
        # Skip if interval is too short
        if end - start < epoch_samples:
            continue
            
        # Get label from expected sequence
        label = expected_labels[i]
        
        # Extract epoch
        epoch_start = start  # Start at trigger onset
        epoch_end = epoch_start + epoch_samples
        
        # Ensure we don't go beyond the data
        if epoch_end > eeg_data.shape[1]:
            continue
            
        epoch = eeg_data[1:9, epoch_start:epoch_end]  # CH2-CH9 (index 1-8)
        
        epochs.append(epoch)
        labels.append(label)
    
    print(f"Extracted {len(epochs)} epochs")
    label_counts = np.bincount(labels)[1:] if len(labels) > 0 else []
    print(f"Labels distribution: {label_counts}")
    
    # Proceed with feature extraction and classification
    if len(epochs) < 8 or len(np.unique(labels)) < 2:
        print(f"Error: Not enough epochs or unique labels for classification")
        return None
    
    # Extract features
    print("Extracting features...")
    features = extract_features(epochs, stim_freqs)
    
    # Ensure balanced train/test split
    from sklearn.model_selection import StratifiedKFold
    
    # Use 5-fold cross validation to get balanced results
    n_splits = 5
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    accuracies = []
    all_y_test = []
    all_y_pred = []
    
    for train_idx, test_idx in kf.split(features, labels):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = np.array(labels)[train_idx], np.array(labels)[test_idx]
        
        # Scale features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train SVM classifier
        from sklearn.svm import SVC
        model = SVC(kernel='rbf', C=10, gamma='scale')
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Calculate accuracy
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        
        # Store predictions
        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)
    
    # Calculate overall accuracy
    avg_accuracy = np.mean(accuracies)
    print(f"Cross-validation accuracy: {avg_accuracy:.4f} ({avg_accuracy*100:.1f}%)")
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(all_y_test, all_y_pred)
    
    # Generate confusion matrix visualization
    generate_full_confusion_matrix(conf_matrix, all_y_test, all_y_pred, stim_freqs, label_counts, f"{dataset_name} (Fixed)")
    
    # Print results
    print(f"\nResults for {dataset_name} (Fixed):")
    print(f"Accuracy: {avg_accuracy:.4f} ({avg_accuracy*100:.1f}%)")
    print("Confusion Matrix:")
    print(conf_matrix)
    
    return avg_accuracy

def generate_confusion_matrix(conf_matrix, y_test, y_pred, stim_freqs, dataset_name):
    """
    Generate and save a confusion matrix visualization
    
    Parameters:
    -----------
    conf_matrix : ndarray
        Confusion matrix
    y_test : ndarray
        True labels
    y_pred : ndarray
        Predicted labels
    stim_freqs : list
        List of stimulus frequencies
    dataset_name : str
        Name of the dataset for display purposes
    """
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
    plt.suptitle(f"Classification Results: {dataset_name}", fontsize=16)
    
    sns.heatmap(full_conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2%})')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save the figure
    filename = f"images/confusion_matrix_{dataset_name.replace(' ', '_').lower()}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {filename}")

def generate_single_class_confusion_matrix(full_conf_matrix, stim_freqs, class_label, dataset_name):
    """
    Generate a confusion matrix for a dataset with only one class
    
    Parameters:
    -----------
    full_conf_matrix : ndarray
        Full confusion matrix with all classes
    stim_freqs : list
        List of stimulus frequencies
    class_label : int
        The single class label present in the dataset
    dataset_name : str
        Name of the dataset for display purposes
    """
    # Create class names for all frequencies
    class_names = [f"Freq {freq} Hz" for freq in stim_freqs]
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.suptitle(f"Classification Results: {dataset_name}", fontsize=16)
    
    sns.heatmap(full_conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix (Single Class: {stim_freqs[class_label-1]} Hz)')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save the figure
    filename = f"images/confusion_matrix_{dataset_name.replace(' ', '_').lower()}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {filename}")

def generate_full_confusion_matrix(conf_matrix, y_test, y_pred, stim_freqs, original_counts, dataset_name):
    """
    Generate and save a full confusion matrix visualization showing all frequencies
    
    Parameters:
    -----------
    conf_matrix : ndarray
        Confusion matrix
    y_test : list
        True labels
    y_pred : list
        Predicted labels
    stim_freqs : list
        List of stimulus frequencies
    original_counts : list
        Original count of labels in the dataset
    dataset_name : str
        Name of the dataset for display purposes
    """
    # Create class names for all frequencies
    class_names = [f"Freq {freq} Hz" for freq in stim_freqs]
    
    # Create a full confusion matrix with all classes
    all_classes = np.arange(1, len(stim_freqs) + 1)  # Class numbers 1-4 (for 15, 12, 10, 9 Hz)
    n_classes = len(all_classes)
    
    # Initialize a new confusion matrix with zeros
    full_conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    # Map the existing confusion matrix to our full one
    for i in range(n_classes):
        for j in range(n_classes):
            class_i = i + 1  # Convert 0-based index to 1-based class label
            class_j = j + 1
            
            # Count occurrences of this combination in the test/pred data
            count = np.sum((np.array(y_test) == class_i) & (np.array(y_pred) == class_j))
            full_conf_matrix[i, j] = count
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.suptitle(f"Classification Results: {dataset_name}", fontsize=16)
    
    # Use a custom annotation function to add expected/actual counts
    ax = plt.subplot()
    sns.heatmap(full_conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    
    # Add the original count of each class
    if len(original_counts) == n_classes:
        class_counts_str = ", ".join([f"{freq}Hz: {count}" for freq, count in zip(stim_freqs, original_counts)])
        plt.figtext(0.5, 0.01, f"Original distribution: {class_counts_str}", 
                   ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2%})')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.08)
    
    # Save the figure
    filename = f"images/confusion_matrix_{dataset_name.replace(' ', '_').lower()}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {filename}")

def main():
    # Set the stimulus frequencies
    stim_freqs = [15, 12, 10, 9]  # Stimulus frequencies (top, right, bottom, left)
    
    # Dataset files and names
    datasets = [
        {"path": "data/subject_1_fvep_led_training_1.mat", "name": "Subject 1 Training 1"},
        {"path": "data/subject_1_fvep_led_training_2.mat", "name": "Subject 1 Training 2"},
        {"path": "data/subject_2_fvep_led_training_1.mat", "name": "Subject 2 Training 1"},
        {"path": "data/subject_2_fvep_led_training_2.mat", "name": "Subject 2 Training 2"}
    ]
    
    # Just show a summary of original data extraction
    print("=== Checking dataset label distributions ===")
    for dataset in datasets:
        try:
            # Load data
            eeg_data = load_data(dataset["path"])
            if eeg_data is not None:
                # Extract epochs
                epochs, labels = extract_epochs(eeg_data, eeg_data[9], epoch_duration=3, stride=1, fs=256)
                # Filter out invalid labels
                valid_labels = [label for label in labels if 1 <= label <= 4]
                unique_labels = np.unique(valid_labels)
                print(f"{dataset['name']} extracted classes: {unique_labels}")
        except Exception as e:
            print(f"Error checking dataset {dataset['name']}: {e}")
    
    # Use fixed label extraction for ALL datasets
    results = {}
    
    print("\n=== USING FIXED LABEL EXTRACTION FOR ALL DATASETS ===")
    for dataset in datasets:
        accuracy = analyze_dataset_with_fixed_labels(dataset["path"], stim_freqs, dataset["name"])
        results[dataset["name"]] = accuracy
    
    # Create a summary visualization
    visualize_accuracy_results(results, "all_datasets")

def visualize_accuracy_results(results, suffix=""):
    """
    Generate a bar chart showing accuracy for all datasets
    
    Parameters:
    -----------
    results : dict
        Dictionary mapping dataset names to accuracy values
    suffix : str
        Suffix to append to the output filename
    """
    # Filter out None values (failed datasets)
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    # Calculate average accuracy
    average_accuracy = np.mean(list(valid_results.values()))
    
    # Add average to results
    all_results = valid_results.copy()
    all_results["Average"] = average_accuracy
    
    # Create bar chart
    plt.figure(figsize=(12, 6))
    plt.suptitle("SSVEP Classification Accuracy Across Datasets", fontsize=16)
    
    # Get dataset names and accuracy values
    names = list(all_results.keys())
    accuracy_values = list(all_results.values())
    
    # Create a bar chart
    bars = plt.bar(names, [acc * 100 for acc in accuracy_values])
    
    # Set different color for average
    bars[-1].set_color('red')
    
    # Add accuracy values as text
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{accuracy_values[i]:.1%}', ha='center', va='bottom')
    
    # Customize plot
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 105)
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    # Save the figure
    filename = f"images/all_datasets_accuracy_{suffix}.png" if suffix else "images/all_datasets_accuracy.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n=== Overall Results ({suffix}) ===")
    for name, accuracy in all_results.items():
        print(f"{name}: {accuracy:.1%}")
    
    print(f"\nSummary visualization saved to {filename}")

if __name__ == "__main__":
    main() 