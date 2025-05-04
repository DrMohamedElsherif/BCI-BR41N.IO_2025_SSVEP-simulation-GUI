#!/usr/bin/env python
"""
Build script for creating SSVEP Simulation executables
This script automatically builds an executable for the current platform
with all data files properly included
"""

import os
import sys
import platform
import subprocess
import shutil
from pathlib import Path

# Check if PyInstaller is installed
try:
    import PyInstaller
except ImportError:
    print("PyInstaller is not installed. Installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

# Determine the operating system
os_name = platform.system()
print(f"Building for {os_name}")

# Base command line arguments - using directory mode instead of onefile for better data access
app_name = "SSVEP_Simulation"
base_args = ["--windowed", "--name", app_name]

# Additional modules that might need explicit inclusion
hidden_imports = [
    "--hidden-import=pygame.mixer", 
    "--hidden-import=numpy.core",
    "--hidden-import=matplotlib"
]

# Platform-specific settings
if os_name == "Darwin":  # macOS
    icon_file = "icon.icns"  # Create or provide a macOS icon file
    data_separator = ":"
elif os_name == "Windows":
    icon_file = "icon.ico"  # Create or provide a Windows icon file
    data_separator = ";"
else:
    print(f"Unsupported OS: {os_name}")
    sys.exit(1)

# Check for icon file and use it if available
icon_arg = []
if os.path.exists(icon_file):
    icon_arg = ["--icon", icon_file]

# Clean up previous build artifacts
if os.path.exists("build"):
    shutil.rmtree("build")
if os.path.exists("dist"):
    shutil.rmtree("dist")
if os.path.exists(f"{app_name}.spec"):
    os.remove(f"{app_name}.spec")

# First create the spec file
spec_cmd = [
    sys.executable, 
    "-m", 
    "PyInstaller", 
    "--name", 
    app_name
] + hidden_imports + ["simulation.py"]

print("Creating spec file...")
subprocess.check_call(spec_cmd)

# Now modify the spec file to include all data files
spec_file = f"{app_name}.spec"
if os.path.exists(spec_file):
    with open(spec_file, 'r') as f:
        spec_content = f.read()
    
    # Check if data directory exists
    data_path = Path("data")
    if data_path.exists() and data_path.is_dir():
        print("Adding data directory to spec file...")
        
        # Find all files in the data directory
        data_files = []
        for root, dirs, files in os.walk("data"):
            for file in files:
                source_path = os.path.join(root, file)
                target_path = os.path.join(os.path.relpath(root, "."), file)
                data_files.append((source_path, os.path.dirname(target_path)))
        
        # Create data files section
        data_section = "datas=[\n"
        for source, target in data_files:
            data_section += f"    ('{source}', '{target}'),\n"
        data_section += "],\n"
        
        # Insert the data section into the spec
        spec_content = spec_content.replace("datas=[],", data_section)
        
        with open(spec_file, 'w') as f:
            f.write(spec_content)
            
        print(f"Added {len(data_files)} data files to spec")
    else:
        print("Warning: Data directory not found!")

# Build using the modified spec file
print("Building application with data...")
build_cmd = [sys.executable, "-m", "PyInstaller", "--clean", spec_file]
try:
    subprocess.check_call(build_cmd)
    print(f"\nBuild completed successfully!")
    print(f"Your application is located in the 'dist/{app_name}' folder")
    
    # Create a zip archive for easier distribution
    if os_name == "Darwin":
        print("Creating macOS application package...")
        # For macOS - create a .app package
        app_dir = f"dist/{app_name}.app"
        if os.path.exists(app_dir):
            print(f"Application bundle created at {app_dir}")
        else:
            print(f"Warning: Expected .app bundle not found at {app_dir}")
    else:
        # For Windows - create a ZIP file
        print("Creating ZIP package for distribution...")
        shutil.make_archive(f"dist/{app_name}", 'zip', f"dist/{app_name}")
        print(f"ZIP package created at dist/{app_name}.zip")
        
except subprocess.CalledProcessError as e:
    print(f"Build failed with error code {e.returncode}")
    sys.exit(e.returncode) 