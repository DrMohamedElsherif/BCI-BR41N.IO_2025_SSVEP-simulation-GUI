#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced SSVEP Experiment Simulation with Real-time Classification

This script extends simulation.py to display real-time classification results
using the enhanced SSVEP classifier.
"""

import pygame
import sys
import time
import numpy as np
import mat73
import os
from ssvep_classifier import preprocess_eeg, extract_features
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
from scipy import signal

# Create FBCCA functions for direct frequency detection
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

# Initialize Pygame
pygame.init()

# Set screen size (larger size, increased height for EEG preview)
screen_width = 1200
screen_height = 800
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Enhanced SSVEP Experiment Simulation with Classification")

# Set Pygame font (for frequency text display)
pygame.font.init()
font = pygame.font.SysFont('Arial', 16)
title_font = pygame.font.SysFont('Arial', 24, bold=True)
button_font = pygame.font.SysFont('Arial', 14)
result_font = pygame.font.SysFont('Arial', 18, bold=True)

# Define colors
LED_colors = {
    "off": (100, 100, 100),      # Gray (LED off)
    "on": (0, 255, 0),           # Green (LED on)
    "cue": (255, 255, 255),      # White (cue light)
    "button": (50, 50, 180),     # Blue (button)
    "highlight": (80, 80, 220),  # Light blue (button highlight)
    "dropdown": (40, 40, 40),    # Dark gray (dropdown background)
    "selected": (60, 100, 180),  # Selected item
    "correct": (0, 255, 0),      # Green for correct classification
    "incorrect": (255, 0, 0),    # Red for incorrect classification
    "processing": (255, 255, 0)  # Yellow for processing
}

# Import relevant functions and variables from simulation.py
from simulation import (
    eeg_channel_colors, background_color, led_positions, led_size, cue_radius,
    frequencies, fixed_led_order, channel_names, channel_colors, timing,
    get_available_datasets, DropdownMenu, DEFAULT_LED_SEQUENCE, analyze_channels
)

# Store classifier model and scaler
classifier_model = None
feature_scaler = None

# Classification results
classification_result = {
    "predicted_class": None,
    "confidence": 0.0,
    "is_correct": False,
    "processing": False,
    "target_freq": None,
    "predicted_freq": None
}

# Initialize model
def load_classifier_model():
    """Load or train classifier model"""
    global classifier_model, feature_scaler
    
    # Check if trained model exists
    model_path = "trained_ssvep_classifier.pkl"
    scaler_path = "feature_scaler.pkl"
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        # Load existing model
        try:
            with open(model_path, 'rb') as f:
                classifier_model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                feature_scaler = pickle.load(f)
            print("Loaded existing classifier model and scaler")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
    
    # Create a new model if loading fails
    print("Creating new classifier model")
    classifier_model = VotingClassifier(
        estimators=[
            ('svm', SVC(kernel='rbf', C=10, gamma='scale', probability=True)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('mlp', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42))
        ],
        voting='soft'
    )
    feature_scaler = StandardScaler()
    
    return False

# Update the load_data function to prepare classifier
def load_data(dataset_index):
    from simulation import load_data as sim_load_data
    
    # Call the original load_data function
    data, led_sequence, trigger_samples = sim_load_data(dataset_index)
    
    # Initialize classifier if needed
    if classifier_model is None:
        load_classifier_model()
    
    return data, led_sequence, trigger_samples

# Classify current epoch
def classify_current_epoch(data, current_sample, target_led_index, stim_freqs):
    """Classify the current epoch of EEG data"""
    global classification_result
    
    # Only classify if we have a valid model and data
    if classifier_model is None or data is None:
        return
    
    # Set classification status to processing
    classification_result["processing"] = True
    
    # We need a 3-second epoch for classification (assuming 256 Hz)
    fs = 256
    epoch_duration = 3  # seconds
    epoch_samples = int(epoch_duration * fs)
    
    # Make sure we have enough data for a complete epoch
    if current_sample < epoch_samples:
        classification_result["processing"] = False
        return
    
    # Extract the most recent 3-second epoch
    start_sample = current_sample - epoch_samples
    end_sample = current_sample
    
    try:
        # Extract EEG channels (1-8)
        epoch = data[1:9, start_sample:end_sample]
        
        # Preprocess epoch
        processed_epoch = preprocess_eeg(epoch, apply_notch=True, fs=fs)
        
        # Map LED positions to their corresponding frequencies
        led_index_to_position = ["top", "right", "bottom", "left"]
        target_position = led_index_to_position[target_led_index]
        
        # Corrected frequency mapping - positions have these frequencies:
        # top: 15Hz, right: 12Hz, bottom: 10Hz, left: 9Hz
        position_to_freq = {
            "top": 15,
            "right": 12, 
            "bottom": 10,
            "left": 9
        }
        target_freq = position_to_freq[target_position]
        
        # Use direct FBCCA to identify the frequency (more reliable than the trained model)
        # Duration of the epoch in seconds
        T = processed_epoch.shape[1] / fs
        
        # Create reference signals
        Y_ref = create_reference_signals(stim_freqs, fs, T)
        
        # Apply filter bank
        filter_bank_signals = apply_filter_bank(processed_epoch, fs, n_bands=8)
        
        # Calculate weights for each band (emphasize lower frequency bands)
        weights = np.power(np.arange(1, len(filter_bank_signals) + 1), -1.5)
        
        # Calculate CCA for each frequency
        r_values = []
        
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
            r_values.append(r_weighted)
        
        # Find frequency with highest correlation
        predicted_class = np.argmax(r_values)
        predicted_freq = stim_freqs[predicted_class]
        confidence = r_values[predicted_class] / sum(r_values)  # Normalize
        
        # Check if prediction matches target
        is_correct = (predicted_freq == target_freq)
        
        # Update classification result
        classification_result = {
            "predicted_class": int(predicted_class),
            "confidence": float(confidence),
            "is_correct": bool(is_correct),
            "processing": False,
            "target_freq": target_freq,
            "predicted_freq": predicted_freq
        }
        
        # Print debug info
        print(f"Classification: Target={target_freq}Hz (LED: {target_position}), Predicted={predicted_freq}Hz, Correct={is_correct}")
        print(f"CCA values: {r_values}")
        print(f"Confidence: {confidence*100:.1f}%")
        
    except Exception as e:
        print(f"Classification error: {e}")
        classification_result["processing"] = False

# Draw classification results
def draw_classification_results(current_trial, led_sequence):
    """Draw the classification results panel"""
    result_area = {
        'x': 400,
        'y': 120,
        'width': 180,
        'height': 160  # Slightly larger panel
    }
    
    # Draw background panel
    pygame.draw.rect(screen, (30, 30, 50), 
                    (result_area['x'], result_area['y'], 
                     result_area['width'], result_area['height']))
    
    # Draw border
    pygame.draw.rect(screen, (150, 150, 150), 
                    (result_area['x'], result_area['y'], 
                     result_area['width'], result_area['height']), 1)
    
    # Draw title
    screen.blit(title_font.render("Classification", True, (255, 255, 255)),
              (result_area['x'] + 10, result_area['y'] + 10))
    
    if classification_result["processing"]:
        # Show processing status
        color = LED_colors["processing"]
        status_text = "Processing..."
    else:
        # Show results
        if classification_result["predicted_class"] is not None:
            if classification_result["is_correct"]:
                color = LED_colors["correct"]
                status_text = "CORRECT"
            else:
                color = LED_colors["incorrect"]
                status_text = "INCORRECT"
            
            # Draw detailed results
            target_text = f"Target: {classification_result['target_freq']} Hz"
            pred_text = f"Predicted: {classification_result['predicted_freq']} Hz"
            conf_text = f"Confidence: {classification_result['confidence']*100:.1f}%"
            
            # Corrected LED position mapping
            position_to_freq = {
                "top": 15,
                "right": 12, 
                "bottom": 10,
                "left": 9
            }
            # Create reverse mapping
            freq_to_pos = {freq: pos.upper() for pos, freq in position_to_freq.items()}
            
            target_pos = freq_to_pos.get(classification_result['target_freq'], "Unknown")
            pred_pos = freq_to_pos.get(classification_result['predicted_freq'], "Unknown")
            
            pos_text = f"LED: {target_pos} → {pred_pos}"
            
            screen.blit(font.render(target_text, True, (200, 200, 200)),
                      (result_area['x'] + 10, result_area['y'] + 70))
            screen.blit(font.render(pred_text, True, (200, 200, 200)),
                      (result_area['x'] + 10, result_area['y'] + 90))
            screen.blit(font.render(conf_text, True, (200, 200, 200)),
                      (result_area['x'] + 10, result_area['y'] + 110))
            screen.blit(font.render(pos_text, True, (200, 200, 200)),
                      (result_area['x'] + 10, result_area['y'] + 130))
        else:
            color = (150, 150, 150)
            status_text = "Waiting..."
    
    # Draw status text with appropriate color
    screen.blit(result_font.render(status_text, True, color),
              (result_area['x'] + 10, result_area['y'] + 40))

# Main simulation function (extended from original)
def run_simulation():
    # Create our own utility functions instead of importing from simulation
    def draw_text(text, position, font_obj=font, color=(255, 255, 255)):
        text_surface = font_obj.render(text, True, color)
        screen.blit(text_surface, position)
    
    def draw_timeline(current_time, total_time):
        timeline_width = 300
        timeline_height = 20
        x_pos = 50
        y_pos = 680
        
        # Draw border
        pygame.draw.rect(screen, (150, 150, 150), (x_pos, y_pos, timeline_width, timeline_height), 1)
        
        # Draw progress
        progress = min(1.0, current_time / total_time)
        pygame.draw.rect(screen, (0, 200, 0), 
                        (x_pos, y_pos, int(timeline_width * progress), timeline_height))
        
        # Draw time text
        draw_text(f"Time: {current_time:.1f}s / {total_time:.1f}s", (x_pos, y_pos - 25))
        
        # Draw trial markers on the timeline
        for i, trial_time in enumerate(trial_start_times):
            if trial_time <= total_time:
                marker_x = x_pos + int(timeline_width * (trial_time / total_time))
                # Draw trial marker
                pygame.draw.line(screen, (255, 255, 255), 
                               (marker_x, y_pos), (marker_x, y_pos + timeline_height), 2)
    
    def draw_timing_info():
        timing_x = 50
        timing_y = 550
        
        draw_text("Experiment Timing:", (timing_x, timing_y), 
                 font_obj=font, color=(255, 255, 255))
        
        # Draw the initial delay
        draw_text(f"Initial Delay: {timing['initial_delay']} seconds", 
                 (timing_x, timing_y + 25), color=(200, 200, 200))
        
        # Draw the trial timing
        draw_text(f"Trial Structure: {timing['trial_pause']}s pause + {timing['stimulus_on']}s stimulus", 
                 (timing_x, timing_y + 50), color=(200, 200, 200))
        
        # Current time
        current_sec = data_time
        minutes = int(current_sec // 60)
        seconds = current_sec % 60
        draw_text(f"Current Time: {minutes:02d}:{seconds:05.2f}", 
                 (timing_x, timing_y + 75), color=(255, 255, 0))
    
    def draw_eeg_preview(data, current_sample, window_size=2000):
        # Set EEG preview area - display on right side and make it larger
        preview_area = {
            'x': 600,
            'y': 120,
            'width': 550,
            'height': 600,
            'channels': 10  # 8 EEG channels + Trigger + LDA output
        }
        
        # Draw background
        pygame.draw.rect(screen, (15, 15, 30), 
                        (preview_area['x'], preview_area['y'], 
                         preview_area['width'], preview_area['height']))
        
        # Draw border
        pygame.draw.rect(screen, (150, 150, 150), 
                        (preview_area['x'], preview_area['y'], 
                         preview_area['width'], preview_area['height']), 1)
        
        # Display EEG preview title
        draw_text("EEG Channels Preview", (preview_area['x'], preview_area['y'] - 30), 
                 font_obj=title_font, color=(255, 255, 255))
        
        # Set data range
        start = max(0, current_sample - window_size)
        end = min(len(data[1]) if len(data) > 1 else 0, current_sample)
        
        if end <= start:
            return
        
        # Calculate channel height
        channel_height = preview_area['height'] / preview_area['channels']
        
        # Define channels to display in order
        display_order = [
            'Trigger',  # Display trigger at the top
            'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2',  # EEG channels
            'LDA'  # LDA at the bottom
        ]
        
        # Simplified EEG display
        for idx, channel_name in enumerate(display_order):
            for ch_idx in range(1, min(10, len(data))):
                if ch_idx < len(data) and start < len(data[ch_idx]) and end <= len(data[ch_idx]):
                    # Get channel data
                    channel_slice = data[ch_idx][start:end]
                    if len(channel_slice) > 0:
                        # Calculate y position
                        y_pos = preview_area['y'] + idx * channel_height + channel_height / 2
                        
                        # Draw channel name
                        draw_text(f"Ch{ch_idx}", (preview_area['x'] - 40, y_pos), 
                                font_obj=font, color=(200, 200, 200))
                        
                        # Normalize data for display
                        ch_min, ch_max = np.min(channel_slice), np.max(channel_slice)
                        y_range = max(1, ch_max - ch_min)
                        
                        # Draw graph line
                        prev_x, prev_y = None, None
                        for i, val in enumerate(channel_slice[::10]):  # Sample every 10th point for performance
                            x = preview_area['x'] + int(i * 10 * preview_area['width'] / window_size)
                            y = y_pos - int((val - ch_min) * channel_height * 0.4 / y_range)
                            
                            if prev_x is not None:
                                pygame.draw.line(screen, eeg_channel_colors[ch_idx % len(eeg_channel_colors)], 
                                               (prev_x, prev_y), (x, y), 1)
                            prev_x, prev_y = x, y
    
    # Create dropdown menu
    dataset_dropdown = DropdownMenu(50, 20, 250, 30, get_available_datasets())
    
    # Load initial dataset
    current_dataset_index = 0
    data, led_sequence, trigger_samples = load_data(current_dataset_index)
    
    # Map LED index to position names
    led_index_to_position = ["top", "right", "bottom", "left"]
    
    # Define stimulus frequencies for classification
    # This must match the frequencies of the LEDs at top, right, bottom, left positions
    # Corrected order: the LED indices 0,1,2,3 correspond to frequencies 15,12,10,9 Hz
    stim_freqs = [15, 12, 10, 9]
    
    print(f"LED positions: {led_index_to_position}")
    print(f"Stimulus frequencies: {stim_freqs}")
    
    # Simulation parameters
    fs = 256  # Sampling rate (Hz)
    speed_multiplier = 1.0  # Play at real-time speed
    
    # Set up manual trial timing if no data loaded
    if data is None:
        data = np.zeros((11, fs * 180))  # 3 minutes of dummy data
        
    # Simulation time state
    sim_start_time = time.time()
    data_time = 0  # Current time in the data (seconds)
    paused = False
    
    # Determine experiment length from data
    experiment_duration = len(data[0]) / fs  # seconds
    
    # Track active trials and timing
    trial_start_times = []
    for idx in trigger_samples:
        trial_start_times.append(idx / fs)
    
    # Classification timer - how often to update classification
    last_classification_time = 0
    classification_interval = 0.5  # seconds
    
    # Enhanced LED drawing function with classification feedback
    def draw_leds(current_time, current_sample):
        # LED area
        led_area = {
            'x': 50,
            'y': 120,
            'width': 300,
            'height': 300
        }
        
        # Check if current_sample is valid
        if current_sample >= len(data[0]):
            return
        
        # Get trigger value
        trigger_value = data[9][current_sample] if 9 < len(data) and current_sample < len(data[9]) else 0
        trigger_is_active = round(trigger_value) == 1
        
        # Find current trial - which LED should be flickering
        current_trial = None
        
        # Find the most recent trigger start before current_sample
        for i in range(len(trigger_samples)-1, -1, -1):
            if current_sample >= trigger_samples[i]:
                current_trial = i
                break
        
        # Draw LED panel background
        pygame.draw.rect(screen, (20, 20, 40), 
                        (led_area['x'], led_area['y'], led_area['width'], led_area['height']))
        pygame.draw.rect(screen, (100, 100, 120), 
                        (led_area['x'], led_area['y'], led_area['width'], led_area['height']), 1)
        
        # Draw title
        draw_text("SSVEP LED Simulation", (led_area['x'] + 50, led_area['y'] - 30), 
                  font_obj=title_font, color=(255, 255, 255))
        
        # Add trigger debug information
        draw_text(f"Trigger: {trigger_value:.3f} (Active: {trigger_is_active})", 
                  (led_area['x'], led_area['y'] - 60), 
                  font_obj=font, color=(255, 255, 0))
        
        # Determine target LED
        target_led_index = None
        if current_trial is not None and current_trial < len(led_sequence):
            target_led_index = led_sequence[current_trial]
        
        # Draw all LEDs
        for position_idx, (position, (x, y)) in enumerate(led_positions.items()):
            freq = frequencies[position]
            
            # Determine if this LED should be active
            is_target = False
            if target_led_index is not None:
                is_target = (position == led_index_to_position[target_led_index])
            
            # LEDs only flicker if they are the target and trigger is active
            led_should_blink = is_target and trigger_is_active
            
            # Determine LED state (on/off) based on frequency
            is_on = False
            if led_should_blink:
                is_on = (current_time * freq) % 1 < 0.5
            
            # Calculate rectangle position
            rect_x = x - led_size[0] // 2
            rect_y = y - led_size[1] // 2
            
            # Set base color
            if led_should_blink and is_on:
                color = LED_colors["on"]
                border_color = (255, 255, 255)
                border_width = 2
            elif is_target:
                color = (40, 40, 40)
                border_color = (100, 100, 0)
                border_width = 2
            else:
                color = (20, 20, 20)
                border_color = (70, 70, 70)
                border_width = 1
            
            # Draw LED base
            pygame.draw.rect(screen, color, (rect_x, rect_y, led_size[0], led_size[1]))
            pygame.draw.rect(screen, border_color, (rect_x, rect_y, led_size[0], led_size[1]), border_width)
            
            # Add classification highlight if this is current prediction
            if (not classification_result["processing"] and 
                classification_result["predicted_freq"] is not None and
                classification_result["predicted_freq"] == freq and
                trigger_is_active):
                
                highlight_color = LED_colors["correct"] if classification_result["is_correct"] else LED_colors["incorrect"]
                pygame.draw.rect(screen, highlight_color, 
                               (rect_x-3, rect_y-3, led_size[0]+6, led_size[1]+6), 3)
                
                # Add prediction text
                pred_text = "PREDICTED" if not is_target else "CORRECT" if classification_result["is_correct"] else "INCORRECT"
                draw_text(pred_text, (x - 30, y - led_size[1]//2 - 15), color=highlight_color)
            
            # Draw frequency label
            label_color = (200, 200, 0) if is_target else (150, 150, 150)
            draw_text(f"{freq}Hz", (x - 15, y + led_size[1]//2 + 10), color=label_color)
            
            # Draw cue light if this is the target
            if is_target:
                cue_x = x + led_size[0]//2 - 10
                cue_y = y - led_size[1]//2 + 10
                cue_color = (255, 255, 255) if trigger_is_active else (150, 150, 150)
                pygame.draw.circle(screen, cue_color, (cue_x, cue_y), cue_radius)
        
        # Draw current trial and target info
        if current_trial is not None and current_trial < len(led_sequence):
            target_led_index = led_sequence[current_trial]
            target_position = led_index_to_position[target_led_index]
            target_freq = frequencies[target_position]
            
            draw_text(f"Trial {current_trial+1}: Target LED = {target_position.upper()} ({target_freq}Hz)", 
                     (led_area['x'], led_area['y'] + led_area['height'] + 20), color=(255, 255, 0))
        
        # Update classification result if needed
        nonlocal last_classification_time
        if (trigger_is_active and 
            current_time - last_classification_time >= classification_interval and
            target_led_index is not None):
            
            last_classification_time = current_time
            classify_current_epoch(data, current_sample, target_led_index, stim_freqs)
    
    # Debug
    print(f"LED Sequence: {led_sequence}")
    print(f"Trigger samples: {trigger_samples[:10]}...")
    
    # Main simulation loop
    clock = pygame.time.Clock()
    running = True
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_UP:
                    # Increase speed
                    current_data_time = data_time
                    speed_multiplier = min(10.0, speed_multiplier * 1.5)
                    sim_start_time = time.time() - (current_data_time / speed_multiplier)
                    print(f"Speed increased to {speed_multiplier:.1f}x")
                    
                elif event.key == pygame.K_DOWN:
                    # Decrease speed
                    current_data_time = data_time
                    speed_multiplier = max(0.1, speed_multiplier / 1.5)
                    sim_start_time = time.time() - (current_data_time / speed_multiplier)
                    print(f"Speed decreased to {speed_multiplier:.1f}x")
                
                elif event.key == pygame.K_r:
                    # Reset simulation
                    sim_start_time = time.time()
                    data_time = 0
                    last_classification_time = 0
                    classification_result["predicted_class"] = None
                    
                # Navigation between trials
                elif event.key == pygame.K_RIGHT:
                    # Move to next trial
                    current_sample = int(data_time * fs)
                    for i, trigger_sample in enumerate(trigger_samples):
                        if trigger_sample > current_sample:
                            data_time = trigger_samples[i] / fs
                            sim_start_time = time.time() - (data_time / speed_multiplier)
                            print(f"Moving to trial {i+1}")
                            break
                
                elif event.key == pygame.K_LEFT:
                    # Move to previous trial
                    current_sample = int(data_time * fs)
                    for i in range(len(trigger_samples)-1, -1, -1):
                        if trigger_samples[i] < current_sample - fs:
                            data_time = trigger_samples[i] / fs
                            sim_start_time = time.time() - (data_time / speed_multiplier)
                            print(f"Moving to trial {i+1}")
                            break
            
            # Handle dropdown events
            dataset_changed, new_index = dataset_dropdown.handle_event(event)
            if dataset_changed:
                # Load new dataset
                current_dataset_index = new_index
                data, led_sequence, trigger_samples = load_data(current_dataset_index)
                # Reset simulation
                sim_start_time = time.time()
                data_time = 0
                
                # Update trial start times
                trial_start_times = []
                for idx in trigger_samples:
                    trial_start_times.append(idx / fs)
        
        # Clear screen
        screen.fill(background_color)
        
        # Update simulation time if not paused
        if not paused:
            elapsed_real_time = time.time() - sim_start_time
            data_time = elapsed_real_time * speed_multiplier
        
        # Convert time to sample
        current_sample = int(data_time * fs)
        
        # Check if simulation is complete
        if current_sample >= len(data[0]):
            draw_text("Simulation complete. Press R to restart or select another dataset.", 
                     (50, 200), color=(255, 255, 0))
            
            # Draw dropdown menu at the end
            dataset_dropdown.draw(screen)
            
            pygame.display.flip()
            
            # Continue to handle events
            continue
        
        # Draw controls info
        draw_text(f"Speed: {speed_multiplier:.1f}x (Up/Down keys)", (350, 20))
        draw_text("SPACE: pause, R: restart, LEFT/RIGHT: change trial, ESC: exit", (50, 750))
        
        # Draw LEDs with classification feedback
        draw_leds(data_time, current_sample)
        
        # Draw classification results
        draw_classification_results(current_sample, led_sequence)
        
        # Draw timeline
        draw_timeline(data_time, experiment_duration)
        
        # Draw timing information
        draw_timing_info()
        
        # Draw EEG preview
        draw_eeg_preview(data, current_sample)
        
        # Draw dropdown menu
        dataset_dropdown.draw(screen)
        
        # Update display
        pygame.display.flip()
        
        # Control frame rate
        clock.tick(60)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    run_simulation() 