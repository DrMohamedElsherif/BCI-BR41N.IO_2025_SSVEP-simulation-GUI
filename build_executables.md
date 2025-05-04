# Building Executables for SSVEP Simulation

This guide explains how to create standalone executable files for both macOS and Windows.

## Prerequisites

Install PyInstaller:

```bash
pip install pyinstaller
```

## Building on macOS

1. Activate your virtual environment:

```bash
source .venv/bin/activate
```

2. Run PyInstaller:

```bash
# Basic version
pyinstaller --onefile --windowed simulation.py

# Enhanced version with icon and name
pyinstaller --onefile --windowed --name "SSVEP_Simulation" --icon=path/to/icon.icns simulation.py
```

3. Find your executable in the `dist` folder.

4. To make a proper macOS application bundle:

```bash
# Create app with data folder included
pyinstaller --onefile --windowed --name "SSVEP_Simulation" \
  --add-data "data:data" \
  --icon=path/to/icon.icns simulation.py
```

## Building on Windows

1. Activate your virtual environment:

```cmd
.venv\Scripts\activate
```

2. Run PyInstaller:

```cmd
# Basic version
pyinstaller --onefile --windowed simulation.py

# Enhanced version with icon and name
pyinstaller --onefile --windowed --name "SSVEP_Simulation" --icon=path\to\icon.ico simulation.py
```

3. Find your executable in the `dist` folder.

4. To include the data folder:

```cmd
pyinstaller --onefile --windowed --name "SSVEP_Simulation" ^
  --add-data "data;data" ^
  --icon=path\to\icon.ico simulation.py
```

## Creating a Spec File (Advanced)

For more control, create and modify a spec file:

```bash
# Generate spec file
pyi-makespec --onefile --windowed simulation.py

# Edit the simulation.spec file to customize the build

# Build using the spec file
pyinstaller simulation.spec
```

## Cross-Platform Builds

For true cross-platform builds, you need to run PyInstaller on each target operating system.

## Common Issues

1. **Missing modules**: If the executable fails with import errors, add the specific modules:

   ```
   pyinstaller --onefile --windowed --hidden-import=pygame.mixer simulation.py
   ```

2. **Missing data files**: Ensure all required files are included with `--add-data`.

3. **Code signing**: For distribution on macOS, you may need to sign your application with a Developer ID.
