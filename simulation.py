import pygame
import sys
import time
import numpy as np
import mat73
import os

# Initialize Pygame
pygame.init()

# Set screen size (larger size, increased height for EEG preview)
screen_width = 1200
screen_height = 800  # Increased height
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("SSVEP Experiment Simulation")

# Set Pygame font (for frequency text display)
pygame.font.init()
font = pygame.font.SysFont('Arial', 16)
title_font = pygame.font.SysFont('Arial', 24, bold=True)
button_font = pygame.font.SysFont('Arial', 14)

# Define colors
LED_colors = {
    "off": (100, 100, 100),  # Gray (LED off)
    "on": (0, 255, 0),       # Green (LED on)
    "cue": (255, 255, 255),   # White (cue light)
    "button": (50, 50, 180),  # Blue (button)
    "highlight": (80, 80, 220),  # Light blue (button highlight)
    "dropdown": (40, 40, 40), # Dark gray (dropdown background)
    "selected": (60, 100, 180)  # Selected item
}

# Define EEG channel colors (8 channels)
eeg_channel_colors = [
    (255, 0, 0),      # Red - PO7
    (0, 255, 0),      # Green - PO3
    (0, 0, 255),      # Blue - POz
    (255, 255, 0),    # Yellow - PO4
    (255, 0, 255),    # Magenta - PO8
    (0, 255, 255),    # Cyan - O1
    (255, 165, 0),    # Orange - Oz
    (165, 42, 42)     # Brown - O2
]

background_color = (0, 0, 0)  # Black background

# Set LED positions (4 positions)
led_positions = {
    "top": (200, 190),
    "right": (330, 300),
    "bottom": (200, 410),
    "left": (70, 300)
}

# LED size (rectangle)
led_size = (80, 80)  # Rectangle size (width, height)
cue_radius = 5

# Frequency (Hz) settings
frequencies = {
    "top": 15,
    "right": 12,
    "bottom": 10,
    "left": 9
}

# Fixed order of LEDs for simulation (in case the file parsing fails)
fixed_led_order = [0, 1, 2, 3]  # top, right, bottom, left

# Channel names and mapping - update to match actual data order
# Format: [0: Sample Time, 1-8: EEG channels, 9: Trigger, 10: LDA]
channel_names = ['Sample Time', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Trigger', 'LDA']

# Define explicit colors for each channel for consistent display
channel_colors = {
    'PO7': (255, 0, 0),       # Red
    'PO3': (0, 255, 0),       # Green
    'POz': (0, 0, 255),       # Blue
    'PO4': (255, 255, 0),     # Yellow
    'PO8': (255, 0, 255),     # Magenta
    'O1': (0, 255, 255),      # Cyan
    'Oz': (255, 165, 0),      # Orange
    'O2': (165, 42, 42),      # Brown
    'Trigger': (255, 255, 255),   # White
    'LDA': (0, 255, 255)  # Cyan
}

# Timing settings (in seconds)
timing = {
    "initial_delay": 9.5,  # 9.5 seconds before first stimulus
    "trial_pause": 3.0,    # 3 seconds pause between trials
    "stimulus_on": 5.0     # 5 seconds of stimulus presentation
}

# Global variables for channel tracking
trigger_channel = 9  # Default trigger channel
manually_selected_trigger = None  # For manual override

# Get list of available dataset files
def get_available_datasets():
    datasets = []
    if os.path.exists('data'):
        for file in os.listdir('data'):
            if file.endswith('.mat') and ('subject' in file.lower() or 'training' in file.lower()):
                datasets.append({
                    "path": f'data/{file}',
                    "name": file.replace('.mat', '')
                })
    
    # If no datasets found, add some defaults
    if not datasets:
        datasets = [
            {"path": 'data/subject_1_fvep_led_training_1.mat', "name": "Subject 1, Training 1"},
            {"path": 'data/subject_1_fvep_led_training_2.mat', "name": "Subject 1, Training 2"},
            {"path": 'data/subject_2_fvep_led_training_1.mat', "name": "Subject 2, Training 1"},
            {"path": 'data/subject_2_fvep_led_training_2.mat', "name": "Subject 2, Training 2"}
        ]
    
    return datasets

# Dataset information
datasets = get_available_datasets()

# Dropdown menu class
class DropdownMenu:
    def __init__(self, x, y, width, height, options):
        self.rect = pygame.Rect(x, y, width, height)
        self.options = options
        self.selected_index = 0
        self.is_open = False
        self.option_height = 30
        self.max_visible_options = 4
        
    def draw(self, surface):
        # Draw the main button
        pygame.draw.rect(surface, LED_colors["button"], self.rect)
        pygame.draw.rect(surface, (200, 200, 200), self.rect, 1)  # Border
        
        # Draw the selected option text
        text = button_font.render(self.options[self.selected_index]["name"], True, (255, 255, 255))
        text_rect = text.get_rect(midleft=(self.rect.x + 10, self.rect.y + self.rect.height // 2))
        surface.blit(text, text_rect)
        
        # Draw dropdown arrow
        pygame.draw.polygon(surface, (255, 255, 255), [
            (self.rect.right - 15, self.rect.centery - 3),
            (self.rect.right - 5, self.rect.centery - 3),
            (self.rect.right - 10, self.rect.centery + 5)
        ])
        
        # Draw dropdown menu if open
        if self.is_open:
            visible_options = min(len(self.options), self.max_visible_options)
            dropdown_height = visible_options * self.option_height
            dropdown_rect = pygame.Rect(self.rect.x, self.rect.bottom, self.rect.width, dropdown_height)
            pygame.draw.rect(surface, LED_colors["dropdown"], dropdown_rect)
            pygame.draw.rect(surface, (200, 200, 200), dropdown_rect, 1)  # Border
            
            # Draw options
            for i, option in enumerate(self.options[:visible_options]):
                option_rect = pygame.Rect(self.rect.x, self.rect.bottom + i * self.option_height, 
                                         self.rect.width, self.option_height)
                
                # Highlight option under mouse
                mouse_pos = pygame.mouse.get_pos()
                if option_rect.collidepoint(mouse_pos):
                    pygame.draw.rect(surface, LED_colors["highlight"], option_rect)
                
                # Draw option text
                text = button_font.render(option["name"], True, (255, 255, 255))
                text_rect = text.get_rect(midleft=(option_rect.x + 10, option_rect.y + option_rect.height // 2))
                surface.blit(text, text_rect)
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Check if dropdown button was clicked
            if self.rect.collidepoint(event.pos):
                self.is_open = not self.is_open
                return False, self.selected_index
            
            # Check if an option was clicked
            elif self.is_open:
                visible_options = min(len(self.options), self.max_visible_options)
                for i in range(visible_options):
                    option_rect = pygame.Rect(self.rect.x, self.rect.bottom + i * self.option_height, 
                                             self.rect.width, self.option_height)
                    if option_rect.collidepoint(event.pos):
                        self.selected_index = i
                        self.is_open = False
                        return True, self.selected_index
        
        # Close dropdown when clicking outside
        if event.type == pygame.MOUSEBUTTONDOWN and self.is_open:
            self.is_open = False
            
        return False, self.selected_index

# Hard-coded trial sequence (if classInfo file fails)
DEFAULT_LED_SEQUENCE = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]  # Cycle through all LEDs

# Add a function to analyze all channels for potential trigger signals
def analyze_channels(data):
    # Check each available channel for characteristics of a trigger signal
    results = []
    for ch in range(len(data)):
        if ch < len(data) and len(data[ch]) > 100:  # Ensure channel has enough data
            # Analyze a sample section
            sample_size = min(5000, len(data[ch]))
            sample = data[ch][:sample_size]
            
            # Check characteristics of potential trigger channel
            unique_vals = np.unique(sample)
            unique_count = len(unique_vals)
            has_zero = 0 in unique_vals
            has_one = 1 in unique_vals or any(abs(v - 1.0) < 0.1 for v in unique_vals)
            max_val = np.max(sample)
            min_val = np.min(sample)
            range_small = max_val - min_val < 10
            
            # Score based on trigger-like properties (0-100)
            score = 0
            if 1 <= unique_count <= 5:  # Few unique values
                score += 40
            elif 5 < unique_count <= 10:
                score += 20
            
            if has_zero and has_one:  # Has both 0 and 1
                score += 40
            
            if range_small:  # Small data range
                score += 20
            
            # Add to results
            results.append({
                'channel': ch,
                'name': channel_names[ch] if ch < len(channel_names) else f"Channel {ch}",
                'unique_values': unique_count,
                'has_zero': has_zero,
                'has_one': has_one,
                'value_range': (min_val, max_val),
                'score': score
            })
    
    # Sort by score (highest first)
    results.sort(key=lambda x: x['score'], reverse=True)
    return results

# Update the load_data function to better detect the trigger channel
def load_data(dataset_index):
    global trigger_channel, manually_selected_trigger
    
    print(f"Loading dataset: {datasets[dataset_index]['name']}...")
    
    # Load selected dataset
    try:
        data_dict = mat73.loadmat(datasets[dataset_index]['path'])
        data = data_dict['y']
        
        # Use manually selected trigger if set
        if manually_selected_trigger is not None:
            trigger_channel = manually_selected_trigger
            print(f"Using manually selected trigger channel: {trigger_channel}")
        else:
            # Run channel analysis
            channel_analysis = analyze_channels(data)
            
            # Print analysis results
            print("\nChannel Analysis Results:")
            for result in channel_analysis[:5]:  # Show top 5
                print(f"Ch{result['channel']} ({result['name']}): Score {result['score']}, " +
                      f"Unique Values: {result['unique_values']}, " +
                      f"Has 0/1: {result['has_zero']}/{result['has_one']}, " +
                      f"Range: {result['value_range'][0]:.2f} to {result['value_range'][1]:.2f}")
            
            # Set trigger channel based on analysis
            if len(channel_analysis) > 0 and channel_analysis[0]['score'] >= 60:
                trigger_channel = channel_analysis[0]['channel']
                print(f"Auto-detected trigger channel at index {trigger_channel} with score {channel_analysis[0]['score']}")
            else:
                # Use traditional detection if analysis doesn't yield good results
                if len(data) == 11:  # Standard format with 11 channels
                    trigger_channel = 9  # Default if 11 channels
                    print(f"Using default trigger channel 9 for 11-channel data")
                else:
                    # Try to find a channel with binary values
                    for ch in range(len(data)):
                        if ch < len(data) and len(data[ch]) > 0:
                            sample = data[ch][:min(2000, len(data[ch]))]
                            if len(sample) > 0:
                                unique_vals = np.unique(sample)
                                # Trigger typically has few unique values (0, 1 or similar)
                                if len(unique_vals) <= 3 and 0 in unique_vals:
                                    # This could be a trigger channel
                                    trigger_channel = ch
                                    print(f"Detected likely trigger channel at index {ch}")
                                    break
        
        # Print the detected trigger channel info
        print(f"\nUSING TRIGGER CHANNEL: {trigger_channel}")
        if trigger_channel < len(channel_names):
            print(f"Trigger channel name: {channel_names[trigger_channel]}")
        
        # Validate if the trigger channel has expected values (0 and 1)
        if trigger_channel < len(data) and len(data[trigger_channel]) > 0:
            sample = data[trigger_channel][:min(5000, len(data[trigger_channel]))]
            unique_vals = sorted(np.unique(sample))
            print(f"Trigger channel unique values: {unique_vals}")
            if not (0 in unique_vals or any(v < 0.1 for v in unique_vals)):
                print("WARNING: Trigger channel may not have expected 0 values!")
            if not (1 in unique_vals or any(abs(v - 1.0) < 0.1 for v in unique_vals)):
                print("WARNING: Trigger channel may not have expected 1 values!")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # Return dummy data
        trigger_channel = 9  # Default for dummy data
        return None, DEFAULT_LED_SEQUENCE, []
    
    # Load class info (which LED is active in each trial)
    led_sequence = []
    try:
        with open('data/classInfo_4_5.m', 'r') as f:
            class_info_lines = f.readlines()
        
        # Parse the trial sequence
        for line in class_info_lines:
            line = line.strip()
            if line and not line.startswith('%'):  # Skip comments and empty lines
                values = line.split()
                if len(values) == 4:  # Check if it's a valid line with 4 values
                    try:
                        values = [int(v) for v in values]
                        # Find the index of '1' to determine which LED was active
                        if 1 in values:
                            led_idx = values.index(1)
                            led_sequence.append(led_idx)
                    except ValueError:
                        pass  # Skip lines that can't be converted to integers
    except Exception as e:
        print(f"Error loading class info: {e}")
        # Use default sequence if file parsing fails
        led_sequence = DEFAULT_LED_SEQUENCE
    
    # If no valid LED sequence was found, use the default
    if not led_sequence:
        led_sequence = DEFAULT_LED_SEQUENCE
        
    print(f"LED sequence: {led_sequence}")
    
    # Get trigger information from data
    fs = 256  # Sampling rate (Hz)
    trigger_samples = []
    
    try:
        triggers = data[9]  # Trigger channel
        # Find rising edges in the trigger signal
        trigger_samples = np.where(np.diff(np.round(triggers)) > 0)[0]
        
        # If no triggers detected or too few triggers, create artificial ones
        if len(trigger_samples) < len(led_sequence):
            print("Too few triggers detected, creating additional ones")
            existing_count = len(trigger_samples)
            
            # Keep existing triggers
            artificial_triggers = list(trigger_samples)
            
            # Calculate start time for artificial triggers
            if existing_count > 0:
                last_trigger = trigger_samples[-1]
                start_sample = last_trigger + int((timing["trial_pause"] + timing["stimulus_on"]) * fs)
            else:
                # Start with initial delay if no triggers
                start_sample = int(timing["initial_delay"] * fs)
            
            # Add artificial triggers for remaining trials
            for i in range(existing_count, len(led_sequence)):
                artificial_triggers.append(start_sample)
                start_sample += int((timing["trial_pause"] + timing["stimulus_on"]) * fs)
            
            trigger_samples = np.array(artificial_triggers)
    except Exception as e:
        print(f"Error processing triggers: {e}")
        # Create artificial trigger samples based on expected trial timing
        trigger_samples = []
        start_sample = int(timing["initial_delay"] * fs)  # Start after initial delay
        
        for i in range(len(led_sequence)):
            trigger_samples.append(start_sample)
            start_sample += int((timing["trial_pause"] + timing["stimulus_on"]) * fs)
        
        trigger_samples = np.array(trigger_samples)
    
    # Create a clean trigger signal if needed
    if 9 in data and (len(data[9]) == 0 or np.max(data[9]) == 0):
        print("Creating clean trigger signal")
        # Create a fresh trigger channel with correct timing
        data[9] = np.zeros(data[0].shape)
        
        for trigger_start in trigger_samples:
            trigger_end = trigger_start + int(timing["stimulus_on"] * fs)
            if trigger_end < len(data[9]):
                data[9][trigger_start:trigger_end] = 1.0
    
    print(f"Found {len(trigger_samples)} trigger points")
    return data, led_sequence, trigger_samples

# Main simulation function
def run_simulation():
    # Create dropdown menu
    dataset_dropdown = DropdownMenu(50, 20, 250, 30, datasets)
    
    # Load initial dataset
    current_dataset_index = 0
    data, led_sequence, trigger_samples = load_data(current_dataset_index)
    
    # Map LED index to position names
    led_index_to_position = ["top", "right", "bottom", "left"]
    
    # Simulation parameters
    fs = 256  # Sampling rate (Hz)
    speed_multiplier = 1.0  # Play at real-time speed
    
    # Set up manual trial timing if no data loaded
    if data is None:
        data = np.zeros((11, fs * 180))  # 3 minutes of dummy data
        for i in range(len(led_sequence)):
            if i < len(trigger_samples):
                start_idx = trigger_samples[i]
                end_idx = min(start_idx + int(timing["stimulus_on"] * fs), len(data[0]))  # 5 seconds of trigger ON
                if start_idx < len(data[9]) and end_idx <= len(data[9]):
                    data[9][start_idx:end_idx] = 1.0  # Set trigger ON
    
    # Simulation time state
    sim_start_time = time.time()
    data_time = 0  # Current time in the data (seconds)
    reset_simulation = True
    
    # Determine experiment length from data
    experiment_duration = len(data[0]) / fs  # seconds
    
    # Track active trials and timing
    trial_start_times = []
    for idx in trigger_samples:
        trial_start_times.append(idx / fs)
    
    # Display text
    def draw_text(text, position, font_obj=font, color=(255, 255, 255)):
        text_surface = font_obj.render(text, True, color)
        screen.blit(text_surface, position)
    
    # Draw trial timeline
    def draw_timeline(current_time, total_time):
        timeline_width = 300
        timeline_height = 20
        x_pos = 50
        y_pos = 680  # Adjusted position - moved above EEG preview
        
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
                
                # Draw trial number
                if i < len(led_sequence):
                    led_idx = led_sequence[i]
                    color = (255, 255, 0)
                    pygame.draw.circle(screen, color, (marker_x, y_pos - 5), 3)
    
    # Add a function to draw experiment timing information
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
            
    # Draw all LEDs
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
        
        # Get trigger value directly from the data at current sample
        trigger_value = 0
        if trigger_channel < len(data) and current_sample < len(data[trigger_channel]):
            trigger_value = data[trigger_channel][current_sample]
        
        # Check several nearby samples to see if trigger is changing
        trigger_values_nearby = []
        for offset in range(-10, 11, 2):  # Check 10 samples before and after
            check_sample = max(0, min(len(data[0])-1, current_sample + offset))
            if trigger_channel < len(data) and check_sample < len(data[trigger_channel]):
                trigger_values_nearby.append(data[trigger_channel][check_sample])
        
        # Determine if trigger is active (EXACTLY 1, not just > 0)
        # Some datasets might have noise, so round to the nearest integer
        rounded_trigger = round(trigger_value)
        trigger_is_active = (rounded_trigger == 1)
        
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
        draw_text(f"Trigger Ch{trigger_channel}: {trigger_value:.3f} (Active: {trigger_is_active})", 
                  (led_area['x'], led_area['y'] - 60), 
                  font_obj=font, color=(255, 255, 0))
                
        # Draw all LEDs
        for position, (x, y) in led_positions.items():
            freq = frequencies[position]
            
            # Determine if this LED should be active in the current trial
            is_target = False
            if current_trial is not None and current_trial < len(led_sequence):
                target_led_index = led_sequence[current_trial]
                is_target = (position == led_index_to_position[target_led_index])
            
            # LEDs ONLY flicker if:
            # 1. They are the target for the current trial
            # 2. The trigger value is EXACTLY 1 (or very close to 1)
            led_should_blink = is_target and trigger_is_active
            
            # Determine LED state (on/off) based on frequency
            is_on = False
            if led_should_blink:
                # Make LED flicker at the specified frequency
                is_on = (current_time * freq) % 1 < 0.5
            
            # Calculate rectangle position (centered on the position coordinates)
            rect_x = x - led_size[0] // 2
            rect_y = y - led_size[1] // 2
            
            # Set colors based on state
            if led_should_blink and is_on:
                # LED is ON - draw with bright color
                color = LED_colors["on"]
                # Draw LED as rectangle
                pygame.draw.rect(screen, color, (rect_x, rect_y, led_size[0], led_size[1]))
                
                # Add a bright border
                pygame.draw.rect(screen, (255, 255, 255), (rect_x, rect_y, led_size[0], led_size[1]), 2)
            else:
                if is_target:
                    # Target LED but not actively blinking - draw dim outline only
                    pygame.draw.rect(screen, (40, 40, 40), (rect_x, rect_y, led_size[0], led_size[1]))
                    pygame.draw.rect(screen, (100, 100, 0), (rect_x, rect_y, led_size[0], led_size[1]), 2)
                else:
                    # Not the target LED - draw very dim outline
                    pygame.draw.rect(screen, (20, 20, 20), (rect_x, rect_y, led_size[0], led_size[1]))
                    pygame.draw.rect(screen, (70, 70, 70), (rect_x, rect_y, led_size[0], led_size[1]), 1)
            
            # Draw frequency label
            label_color = (200, 200, 0) if is_target else (150, 150, 150)
            draw_text(f"{freq}Hz", (x - 15, y + led_size[1]//2 + 10), color=label_color)
            
            # Draw cue light if this is the target (even if LED is not currently blinking)
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
                     (led_area['x'], led_area['y'] + led_area['height'] + 100), color=(255, 255, 0))
            
            # Show LED sequence for debugging
            sequence_text = "LED Sequence: " + " → ".join([led_index_to_position[idx].upper() for idx in led_sequence[:4]])
            if len(led_sequence) > 6:
                sequence_text += " ..."
            draw_text(sequence_text, (led_area['x'], led_area['y'] + led_area['height'] + 80), color=(180, 180, 180))
    
    # Draw EEG signal preview (all channels)
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
        
        # Add a debug display of available data channels
        if len(data) > 0:
            draw_text(f"Available Data Channels: {len(data)} (Trigger at Ch{trigger_channel})", 
                     (preview_area['x'], preview_area['y'] - 50), 
                     font_obj=font, color=(200, 200, 200))
        
        # Draw time scale (in seconds)
        seconds_visible = window_size / fs
        for i in range(int(seconds_visible) + 1):
            sec_x = preview_area['x'] + (i * preview_area['width'] / seconds_visible)
            if sec_x <= preview_area['x'] + preview_area['width']:
                # Draw vertical line for each second
                pygame.draw.line(screen, (50, 50, 70), 
                              (sec_x, preview_area['y']), 
                              (sec_x, preview_area['y'] + preview_area['height']), 
                              1)
                # Draw time label
                current_sec = int((current_sample / fs) - (window_size / fs) + i)
                if current_sec >= 0:
                    draw_text(f"{current_sec}s", (sec_x - 10, preview_area['y'] + preview_area['height'] + 5), 
                             font_obj=button_font, color=(180, 180, 180))
        
        # Set data range
        start = max(0, current_sample - window_size)
        end = min(len(data[1]) if len(data) > 1 else 0, current_sample)
        
        if end <= start:
            return
        
        # Channel mapping: Maps data indices to channel names
        # We need to handle datasets with different structures
        channel_mapping = {}
        
        # For standard 11-channel data: [Sample Time, 8 EEG, Trigger, LDA]
        if len(data) == 11:
            for i in range(min(len(data), len(channel_names))):
                channel_mapping[channel_names[i]] = i
        else:
            # For other formats, make our best guess
            # Assume first channel is sample time if available
            if len(data) > 0:
                channel_mapping['Sample Time'] = 0
            
            # Map EEG channels to the first 8 channels after sample time
            eeg_indices = []
            for i in range(1, min(9, len(data))):
                if i != trigger_channel:  # Skip the trigger channel
                    eeg_indices.append(i)
            
            # Assign EEG channel names to available indices
            eeg_names = ['PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']
            for i, name in enumerate(eeg_names):
                if i < len(eeg_indices):
                    channel_mapping[name] = eeg_indices[i]
            
            # Map trigger and LDA if available
            channel_mapping['Trigger'] = trigger_channel
            if len(data) > trigger_channel + 1:
                channel_mapping['LDA'] = trigger_channel + 1
        
        # Print channel mapping for debugging
        print("Channel mapping:")
        for name, idx in channel_mapping.items():
            print(f"  {name}: Channel {idx}")
        
        
        # Calculate channel height
        channel_height = preview_area['height'] / preview_area['channels']
        
        # Define channels to display in order
        display_order = [
            'Trigger',  # Display trigger at the top
            'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2',  # EEG channels
            'LDA'  # LDA at the bottom
        ]
        
        # Draw channel labels and data
        for idx, channel_name in enumerate(display_order):
            if channel_name in channel_mapping:
                ch_idx = channel_mapping[channel_name]
                if ch_idx < len(data):
                    # Calculate y position for this channel
                    legend_y = preview_area['y'] + idx * channel_height + channel_height / 2
                    
                    # Get channel color
                    ch_color = channel_colors.get(channel_name, (255, 255, 255))
                    
                    # Special styling for trigger channel - just use text color, no background box
                    text_color = (255, 0, 0) if channel_name == 'Trigger' else (255, 255, 255)
                    
                    # Display color block
                    pygame.draw.rect(screen, ch_color, 
                                   (preview_area['x'] - 20, legend_y - 8, 15, 15))
                    
                    # Display channel name
                    draw_text(f"{channel_name} (Ch{ch_idx})", 
                             (preview_area['x'] - 120, legend_y - 8), 
                             font_obj=font, color=text_color)
                    
                    # Draw the actual data
                    if start < len(data[ch_idx]) and end <= len(data[ch_idx]):
                        channel_slice = data[ch_idx][start:end]
                        
                        # Normalize data
                        if len(channel_slice) > 0:
                            ch_min, ch_max = np.min(channel_slice), np.max(channel_slice)
                            y_range = max(1, ch_max - ch_min)
                            
                            # Draw graph line
                            prev_x, prev_y = None, None
                            for i, val in enumerate(channel_slice):
                                x = preview_area['x'] + int(i * preview_area['width'] / window_size)
                                # Calculate height
                                y = legend_y - int((val - ch_min) * channel_height * 0.4 / y_range)
                                
                                if prev_x is not None:
                                    # Use thicker line for trigger and LDA
                                    line_width = 2
                                    if channel_name == 'Trigger':
                                        line_width = 3
                                        # For trigger, emphasize ON values with brighter color
                                        line_color = (255, 0, 0) if val > 0 else (100, 0, 0)
                                    else:
                                        line_color = ch_color
                                    
                                    pygame.draw.line(screen, line_color, 
                                                   (prev_x, prev_y), (x, y), line_width)
                                prev_x, prev_y = x, y
        
        # Draw horizontal lines to separate channels
        for ch in range(1, preview_area['channels']):
            line_y = preview_area['y'] + ch * channel_height
            pygame.draw.line(screen, (50, 50, 70), 
                           (preview_area['x'], line_y), 
                           (preview_area['x'] + preview_area['width'], line_y), 
                           1)
        
        # Show current point (vertical line) - moved to right edge rather than middle
        # This replaces the large white bar in the middle that was causing confusion
        right_edge_x = preview_area['x'] + preview_area['width']
        pygame.draw.line(screen, (100, 100, 255),  # Changed to blue color for less visual interference
                       (right_edge_x - 2, preview_area['y']), 
                       (right_edge_x - 2, preview_area['y'] + preview_area['height']), 
                       1)  # Thinner line
        
        # Add small text indicator at the bottom
        draw_text("Now", (right_edge_x - 20, preview_area['y'] + preview_area['height'] + 20), 
                 font_obj=button_font, color=(100, 100, 255))
    
    # Debug
    print(f"LED Sequence: {led_sequence}")
    print(f"Trigger samples: {trigger_samples[:10]}...")
    
    # Main simulation loop
    clock = pygame.time.Clock()
    running = True
    paused = False
    
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
                    # Save current data_time before changing speed
                    current_data_time = data_time
                    # Increase speed
                    speed_multiplier = min(10.0, speed_multiplier * 1.5)
                    # Adjust sim_start_time to maintain position at new speed
                    sim_start_time = time.time() - (current_data_time / speed_multiplier)
                    print(f"Speed increased to {speed_multiplier:.1f}x")
                    
                elif event.key == pygame.K_DOWN:
                    # Save current data_time before changing speed
                    current_data_time = data_time
                    # Decrease speed
                    speed_multiplier = max(0.1, speed_multiplier / 1.5)
                    # Adjust sim_start_time to maintain position at new speed
                    sim_start_time = time.time() - (current_data_time / speed_multiplier)
                    print(f"Speed decreased to {speed_multiplier:.1f}x")
                
                # Add ability to manually set trigger channel with number keys
                elif event.key >= pygame.K_0 and event.key <= pygame.K_9:
                    # Set trigger channel directly (0-9)
                    global manually_selected_trigger
                    channel = event.key - pygame.K_0
                    manually_selected_trigger = channel
                    print(f"MANUALLY SET TRIGGER CHANNEL TO {channel}")
                    # Reload data to apply the change
                    data, led_sequence, trigger_samples = load_data(current_dataset_index)
                    
                elif event.key == pygame.K_r:
                    # Reset simulation
                    sim_start_time = time.time()
                    data_time = 0
                # Add navigation between trials using left/right keys
                elif event.key == pygame.K_RIGHT:
                    # Move to next trial
                    current_sample = int(data_time * fs)
                    next_trial = None
                    
                    # Find the next trial from current position
                    for i, trigger_sample in enumerate(trigger_samples):
                        if trigger_sample > current_sample:
                            next_trial = i
                            break
                    
                    if next_trial is not None:
                        # Set time to the start of the next trial
                        data_time = trigger_samples[next_trial] / fs
                        # Reset simulation start time to maintain proper playback
                        sim_start_time = time.time() - (data_time / speed_multiplier)
                        print(f"Moving to trial {next_trial+1}")
                
                elif event.key == pygame.K_LEFT:
                    # Move to previous trial
                    current_sample = int(data_time * fs)
                    prev_trial = None
                    
                    # Find the previous trial from current position
                    for i in range(len(trigger_samples)-1, -1, -1):
                        if trigger_samples[i] < current_sample - fs:  # Small offset to ensure we're in previous trial
                            prev_trial = i
                            break
                    
                    if prev_trial is not None:
                        # Set time to the start of the previous trial
                        data_time = trigger_samples[prev_trial] / fs
                        # Reset simulation start time to maintain proper playback
                        sim_start_time = time.time() - (data_time / speed_multiplier)
                        print(f"Moving to trial {prev_trial+1}")
                    else:
                        # If no previous trial found, go to beginning
                        data_time = 0
                        sim_start_time = time.time()
                        print("Moving to beginning")
            
            # Handle dropdown events
            dataset_changed, new_index = dataset_dropdown.handle_event(event)
            if dataset_changed:
                # Load new dataset
                current_dataset_index = new_index
                data, led_sequence, trigger_samples = load_data(current_dataset_index)
                # Reset simulation
                sim_start_time = time.time()
                data_time = 0
                experiment_duration = len(data[0]) / fs
                reset_simulation = True
                
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
        draw_text(f"Speed: {speed_multiplier:.1f}x (Up/Down to change)", (350, 20))
        draw_text("Press SPACE to pause/resume, R to restart, ESC to exit", (50, 750))
        
        # Draw LEDs
        draw_leds(data_time, current_sample)
        
        # Draw timeline
        draw_timeline(data_time, experiment_duration)
        
        # Draw timing information
        draw_timing_info()
        
        # Draw EEG preview
        draw_eeg_preview(data, current_sample)
        
        # Draw dropdown menu last, on top of everything else
        dataset_dropdown.draw(screen)
        
        # Update display
        pygame.display.flip()
        
        # Control frame rate
        clock.tick(60)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    run_simulation()