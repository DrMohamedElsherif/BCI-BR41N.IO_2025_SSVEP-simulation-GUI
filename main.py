import mat73
from scipy.signal import resample, find_peaks
import matplotlib.pyplot as plt
import numpy as np

# Load data
sub1_train1_data = f'data/subject_1_fvep_led_training_1.mat'
sub1_train2_data = f'data/subject_1_fvep_led_training_2.mat'
sub2_train1_data = f'data/subject_2_fvep_led_training_1.mat'
sub2_train2_data = f'data/subject_2_fvep_led_training_2.mat'

data_dict_sub1_train1 = mat73.loadmat(sub1_train1_data)
data_dict_sub1_train2 = mat73.loadmat(sub1_train2_data)
data_dict_sub2_train1 = mat73.loadmat(sub2_train1_data)
data_dict_sub2_train2 = mat73.loadmat(sub2_train2_data)

data_sub1_train1 = data_dict_sub1_train1['y']
data_sub1_train2 = data_dict_sub1_train2['y']
data_sub2_train1 = data_dict_sub2_train1['y']
data_sub2_train2 = data_dict_sub2_train2['y']

# Load trial sequence information from classInfo_4_5.m
with open('data/classInfo_4_5.m', 'r') as f:
    class_info_lines = f.readlines()

# Parse the trial sequence - each row indicates which LED was active
led_sequence = []
for line in class_info_lines:
    values = line.strip().split()
    if len(values) == 4:  # Check if it's a valid line with 4 values
        try:
            values = [int(v) for v in values]
            # Find the index of '1' to determine which LED was active
            if 1 in values:
                led_idx = values.index(1)
                led_sequence.append(led_idx)
        except ValueError:
            pass  # Skip lines that can't be converted to integers

# Define LED information
led_frequencies = [9, 10, 12, 15]  # Hz
led_positions = ["Top", "Right", "Bottom", "Left"]
led_colors = ['red', 'green', 'blue', 'purple']

# Sampling rate
fs = 256  # Hz

# Function to convert sample number to time
def sample_to_time(sample):
    return sample / fs

# Function to mark trials on plots
def mark_trials_on_plot(ax, data, time_vector=None):
    """Mark trials on a plot with LED information"""
    if time_vector is None:
        # Create time vector (in seconds)
        time_vector = np.arange(len(data[9])) / fs
    
    # Find trial onsets from trigger channel
    trials = np.where(np.diff(np.round(data[9])) > 0)[0]
    
    # Initial delay is 10s
    initial_delay = 10  # seconds
    
    # Mark each trial with LED info
    for i, trial in enumerate(trials):
        if i < len(led_sequence):
            led_idx = led_sequence[i]
            trial_time = time_vector[trial]
            
            # Add vertical line at trial start
            ax.axvline(x=trial_time, color=led_colors[led_idx], linestyle='-', alpha=0.5)
            
            # Add text label with LED info
            label = f"{led_positions[led_idx]}\n{led_frequencies[led_idx]}Hz"
            y_pos = ax.get_ylim()[1] * 0.9
            ax.text(trial_time + 0.1, y_pos, label, 
                   color=led_colors[led_idx], fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.7))

# Function to plot data with trial information
def plot_data_with_trials(data, title, sample_range=None):
    """Plot EEG channels, LDA output, and trigger with trial information"""
    # Define channel names
    channel_names = ['Sample Time', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Trigger', 'LDA-Output']
    
    # Convert sample range to time range
    if sample_range is None:
        sample_range = (0, len(data[0]))
    time_range = (sample_to_time(sample_range[0]), sample_to_time(sample_range[1]))
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True,
                           gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Create time vector
    time_vector = np.arange(sample_range[0], sample_range[1]) / fs
    
    # Plot EEG channels
    for i in range(1, 9):
        axes[0].plot(time_vector, 
                    data[i][sample_range[0]:sample_range[1]], 
                    label=channel_names[i])
    
    axes[0].set_title(f'EEG Channels - {title}')
    axes[0].set_ylabel('Amplitude (μV)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Mark trials on EEG plot
    mark_trials_on_plot(axes[0], data, time_vector)
    
    # Plot LDA output
    axes[1].plot(time_vector, 
                data[10][sample_range[0]:sample_range[1]], 
                color='purple', linewidth=2)
    axes[1].set_title('LDA Output')
    axes[1].set_ylabel('Classification Output')
    axes[1].grid(True, alpha=0.3)
    
    # Plot trigger
    axes[2].plot(time_vector, 
                data[9][sample_range[0]:sample_range[1]], 
                color='orange', linewidth=1.5)
    axes[2].set_title('Trigger Channel')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_ylabel('Trigger Value')
    axes[2].grid(True, alpha=0.3)
    
    # Add overall title
    plt.suptitle(f'{title} - LED Stimulus Trials', fontsize=16)
    
    # Add timeline indicators for 10s initial delay and 3s pauses
    # Calculate expected trial times based on protocol
    expected_times = [10]  # Start with initial 10s delay
    for i in range(1, len(led_sequence)):
        expected_times.append(expected_times[-1] + 3)  # 3s pause
    
    # Add timeline indicator at bottom
    for t in expected_times:
        if t >= time_range[0] and t <= time_range[1]:
            axes[2].axvline(x=t, color='black', linestyle='--', alpha=0.5)
            axes[2].text(t, axes[2].get_ylim()[1]*0.5, f"{t}s", 
                        fontsize=8, ha='center')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

# Function to plot regions around LDA peaks with trial information
def plot_peaks_with_trials(data, dataset_name, n_peaks=3, window_size=500):
    """Plot regions where LDA output peaks, showing channels and trial info"""
    lda_output = data[10]
    
    # Find peaks in LDA output
    peaks, _ = find_peaks(lda_output, height=np.percentile(lda_output, 99), distance=1000)
    
    # If no peaks found, try with a lower threshold
    if len(peaks) == 0:
        peaks, _ = find_peaks(lda_output, height=np.percentile(lda_output, 95), distance=1000)
    
    # Use top n_peaks
    if len(peaks) > n_peaks:
        # Get peaks sorted by height
        peak_heights = lda_output[peaks]
        sorted_indices = np.argsort(peak_heights)[::-1]  # Sort in descending order
        peaks = peaks[sorted_indices[:n_peaks]]
    
    # Make sure peaks are sorted by time
    peaks = np.sort(peaks)
    
    # Define channel names
    channel_names = ['Sample Time', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Trigger', 'LDA-Output']
    
    print(f"Found {len(peaks)} peaks. Plotting top {min(n_peaks, len(peaks))} peaks.")
    
    # Plot each peak region
    for i, peak in enumerate(peaks):
        # Define window start and end
        start = max(0, peak - window_size)
        end = min(len(lda_output), peak + window_size)
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True,
                               gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Create time vector
        time_vector = np.arange(start, end) / fs
        
        # Plot EEG channels
        for ch in range(1, 9):
            axes[0].plot(time_vector, data[ch][start:end], label=channel_names[ch])
        
        axes[0].set_title(f'EEG Channels Around Peak {i+1}')
        axes[0].set_ylabel('Amplitude (μV)')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        
        # Mark trials on EEG plot
        mark_trials_on_plot(axes[0], data, time_vector)
        
        # Plot LDA output
        axes[1].plot(time_vector, data[10][start:end], color='purple', linewidth=2)
        axes[1].axvline(x=sample_to_time(peak), color='red', linestyle='--', 
                       label='Peak Location')
        axes[1].set_title('LDA Output')
        axes[1].set_ylabel('Classification Output')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        # Plot trigger
        axes[2].plot(time_vector, data[9][start:end], color='orange', linewidth=1.5)
        axes[2].set_title('Trigger Channel')
        axes[2].set_xlabel('Time (seconds)')
        axes[2].set_ylabel('Trigger Value')
        axes[2].grid(True, alpha=0.3)
        
        # Add overall title
        peak_time = sample_to_time(peak)
        plt.suptitle(f'{dataset_name} - Peak at {peak_time:.2f}s (Sample {peak})', fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()

# Plot full dataset with trials
print("Subject 1, Training 1 - Full Dataset with Trial Information:")
plot_data_with_trials(data_sub1_train1, "Subject 1, Training 1")

# Plot segments with interesting features
print("Subject 1, Training 1 - First 60 seconds:")
plot_data_with_trials(data_sub1_train1, "Subject 1, Training 1 - First 60s", 
                     sample_range=(0, 60*fs))

print("Subject 1, Training 1 - LED Trial Sequence:")
for i in range(min(5, len(led_sequence))):
    start_sample = 10*fs + i*(3*fs)  # 10s initial delay + i*(3s pause)
    end_sample = start_sample + 10*fs  # Show 10s of data
    led_idx = led_sequence[i]
    plot_data_with_trials(data_sub1_train1, 
                         f"Subject 1 - Trial {i+1} - {led_positions[led_idx]} LED ({led_frequencies[led_idx]}Hz)",
                         sample_range=(start_sample, end_sample))

# Plot regions around LDA peaks
print("Subject 1, Training 1 - Regions Around LDA Peaks:")
plot_peaks_with_trials(data_sub1_train1, "Subject 1, Training 1", n_peaks=3)

print("Subject 1, Training 2 - Regions Around LDA Peaks:")
plot_peaks_with_trials(data_sub1_train2, "Subject 1, Training 2", n_peaks=3)

print("Subject 2, Training 1 - Regions Around LDA Peaks:")
plot_peaks_with_trials(data_sub2_train1, "Subject 2, Training 1", n_peaks=3)

# Bonus: Compare LDA peaks across datasets
def plot_lda_comparison():
    """Compare LDA output across datasets with trial information"""
    plt.figure(figsize=(15, 10))
    
    datasets = [
        (data_sub1_train1, "Subject 1, Training 1"),
        (data_sub1_train2, "Subject 1, Training 2"), 
        (data_sub2_train1, "Subject 2, Training 1"),
        (data_sub2_train2, "Subject 2, Training 2")
    ]
    
    for i, (data, title) in enumerate(datasets):
        plt.subplot(2, 2, i+1)
        
        # Convert to time
        time = np.arange(len(data[10])) / fs
        
        # Plot LDA output
        plt.plot(time, data[10], label='LDA Output')
        
        # Find peaks
        peaks, _ = find_peaks(data[10], height=np.percentile(data[10], 98), distance=1000)
        peak_heights = data[10][peaks]
        peak_times = peaks / fs
        
        # Plot peaks
        plt.plot(peak_times, peak_heights, 'rx', markersize=8, label='Peaks')
        
        # Mark trials (first 10)
        trials = np.where(np.diff(np.round(data[9])) > 0)[0]
        for j, trial in enumerate(trials[:10]):
            if j < len(led_sequence):
                led_idx = led_sequence[j]
                trial_time = trial / fs
                plt.axvline(x=trial_time, color=led_colors[led_idx], 
                           linestyle='-', alpha=0.3)
        
        plt.title(title)
        plt.xlabel('Time (seconds)')
        plt.ylabel('LDA Output')
        plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Show comparison of LDA output across datasets
plot_lda_comparison()