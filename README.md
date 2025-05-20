#### © Copyright [Moahmed Elsherif/Uni-Tübingen]. All rights reserved. Unauthorized copying, cloning, or distribution is prohibited.  

#  --- Brain-Computer Interface (BCI) Hackathon 2025 ---
# SSVEP Classification and Simulation

A comprehensive toolset for SSVEP (Steady-State Visual Evoked Potential) EEG data analysis, classification, and visualization. This application allows you to visualize EEG data, simulate flickering LED stimuli, and apply advanced classification methods to detect brain responses to visual stimuli.

## Features

- **Real-time Simulation**: Visualize flickering LEDs alongside EEG data
- **Multiple Classification Methods**:
  - Direct FBCCA frequency detection (traditional method)
  - Advanced machine learning model (trained on combined datasets)
- **Continuous Visualization**: See classification results over time
- **Support for Multiple Subjects**: Handles inter-subject variability

## Classification Performance

| Method            | Subject 1 | Subject 2 | Average |
| ----------------- | --------- | --------- | ------- |
| FBCCA (Direct)    | 92-95%    | 60-66%    | ~78%    |
| Combined ML Model | 100%      | 100%      | 100%    |

## Requirements

- Python 3.12 or higher
- Pygame
- NumPy
- mat73
- scikit-learn
- scipy
- matplotlib
- seaborn

## Installation

1. Clone this repository:

```bash
git clone https://github.com/yacaeh/br41n.io-ssvep-simulation-GUI.git
cd SSVEP_hackathon
```

2. Set up a virtual environment:

```bash
# Using uv (recommended)
uv init --no-workspace --python 3.12
source .venv/bin/activate

# Or using venv
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
# Using uv
uv pip install -r requirements.txt

# Or using pip
pip install -r requirements.txt
```

## Classification Scripts

The project includes several scripts for SSVEP classification:

### 1. Basic Simulation

```bash
python simulation.py
```

Visualize EEG data with simulated LED flickering.

### 2. Enhanced Simulation with Classification

```bash
python enhanced_simulation.py
```

Real-time simulation with FBCCA-based frequency classification, showing predictions while the simulation runs.

### 3. Analyze Datasets

```bash
python analyze_all_datasets.py
```

Perform static analysis on all datasets using cross-validation.

### 4. Continuous Prediction Visualization

```bash
python visualize_predictions.py
```

Generate visualizations showing continuous prediction over time using direct FBCCA.

### 5. Combined Model Training and Visualization

```bash
python train_combined_visualize.py
```

Train a machine learning model on all datasets combined and visualize prediction performance.

## Key Insights

- **Inter-subject Variability**: Subject 2's SSVEP responses differ significantly from Subject 1's, requiring adaptive methods
- **Machine Learning Advantage**: ML approaches outperform direct signal processing when handling multiple subjects
- **Combined Training**: Training on data from multiple subjects dramatically improves classification for all subjects
- **Real-time vs. Offline**: Real-time FBCCA shows lower accuracy than offline ML approaches

## Controls for Simulation

- **Space**: Pause/resume the simulation
- **R**: Restart the simulation from the beginning
- **Left/Right Arrows**: Navigate between trials
- **Up/Down Arrows**: Increase/decrease playback speed
- **Number Keys (0-9)**: Manually select trigger channel
- **ESC**: Exit the simulation

## Dataset Structure

The simulation expects MATLAB .mat files in the `data/` directory containing:

- Variable `y` with EEG data in the format: [Sample Time, 8 EEG channels, Trigger, LDA]
- The trigger channel (typically channel 9) indicates when stimuli are presented
- Stimuli frequencies: 15Hz (top), 12Hz (right), 10Hz (bottom), 9Hz (left)

## Classification Methods

### FBCCA (Filter Bank Canonical Correlation Analysis)

- Direct frequency detection using reference sine/cosine signals
- No training required, works on each dataset independently
- Lower accuracy for subjects with atypical SSVEP responses
- Used in `enhanced_simulation.py` and `visualize_predictions.py`

### Machine Learning Model

- Combines features from all datasets
- Learns subject-specific patterns
- Significantly higher accuracy for all subjects
- Used in `train_combined_visualize.py`

## Creating Executables

You can create standalone executables for both macOS and Windows using the provided build scripts:

### For macOS (Recommended):

```bash
python mac_build.py
```

This will create a proper macOS application bundle (.app) with all data files included.
The application will be in `dist/SSVEP_Simulation.app` and a ZIP archive will be created for easy distribution.

### For Windows:

```bash
python build_executable.py
```

## Troubleshooting

If the trigger channel is not correctly detected:

- Use number keys (0-9) to manually select the correct trigger channel
- Check that your data files contain properly formatted EEG data with trigger markers

### Classification Issues

If classification results are poor:

1. **Subject variability**: Different subjects may require personalized training
2. **Window size**: Try adjusting the epoch duration (default: 3 seconds)
3. **Frequency detection**: Ensure the reference frequencies match the actual stimuli
4. **Training data quality**: Check for artifacts or noisy channels in the training data

### Build Issues

If you encounter problems with executable creation:

1. **"Failed to exec pkg" error**:

   - On macOS, use the `mac_build.py` script which creates a proper app bundle
   - Ensure your data folder exists and contains the necessary files

2. **Missing data in the executable**:
   - Both build scripts are configured to include the data folder
   - Check that your data is in a folder named `data` at the project root
