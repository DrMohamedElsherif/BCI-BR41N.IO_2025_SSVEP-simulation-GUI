#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train and save the enhanced SSVEP classifier model

This script trains the enhanced SSVEP classifier on all available datasets
and saves the model for use in the simulation.
"""

import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from ssvep_classifier import load_data, extract_epochs, extract_features
from sklearn.model_selection import train_test_split

def train_classifier_model():
    """Train classifier on all datasets and save the model"""
    print("Training enhanced SSVEP classifier model...")
    
    # Dataset files
    dataset_files = [
        "data/subject_1_fvep_led_training_1.mat",
        "data/subject_1_fvep_led_training_2.mat", 
        "data/subject_2_fvep_led_training_1.mat",
        "data/subject_2_fvep_led_training_2.mat"
    ]
    
    # Stimulus frequencies
    stim_freqs = [15, 12, 10, 9]  # Corrected LED frequencies (top, right, bottom, left)
    
    # Collect all features and labels
    all_features = []
    all_labels = []
    
    # Process each dataset
    for file_path in dataset_files:
        if os.path.exists(file_path):
            print(f"Processing {os.path.basename(file_path)}...")
            
            # Load data
            eeg_data = load_data(file_path)
            if eeg_data is None:
                print(f"Failed to load {file_path}")
                continue
            
            # Extract epochs
            epochs, labels = extract_epochs(eeg_data, eeg_data[9], epoch_duration=3, stride=1, fs=256)
            
            # Debug: print unique labels to understand the mapping
            unique_labels = np.unique(labels)
            print(f"Unique labels in dataset: {unique_labels}")
            
            # Display label count distribution
            for label in unique_labels:
                count = np.sum(labels == label)
                print(f"  Label {label}: {count} instances")
            
            print(f"Extracted {len(epochs)} epochs")
            
            # Extract features
            features = extract_features(epochs, stim_freqs, fs=256, include_spatial=True)
            print(f"Feature shape: {features.shape}")
            
            # Add to collection
            all_features.append(features)
            all_labels.append(labels)
    
    # Combine all datasets
    if all_features:
        features = np.vstack(all_features)
        labels = np.concatenate(all_labels)
        
        print(f"Combined feature shape: {features.shape}")
        print(f"Combined labels shape: {labels.shape}")
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Create ensemble classifier
        ensemble = VotingClassifier(
            estimators=[
                ('svm', SVC(kernel='rbf', C=10, gamma='scale', probability=True)),
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('mlp', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42))
            ],
            voting='soft'
        )
        
        # Train the ensemble
        print("Training ensemble classifier...")
        ensemble.fit(features_scaled, labels)
        
        # Save the trained model and scaler
        with open('trained_ssvep_classifier.pkl', 'wb') as f:
            pickle.dump(ensemble, f)
        
        with open('feature_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        print("Classifier model and scaler saved successfully.")
        
        # Test the model
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Evaluate
        accuracy = ensemble.score(X_test, y_test)
        print(f"Test accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        
        return ensemble, scaler
    else:
        print("No datasets were successfully processed.")
        return None, None

if __name__ == "__main__":
    train_classifier_model() 