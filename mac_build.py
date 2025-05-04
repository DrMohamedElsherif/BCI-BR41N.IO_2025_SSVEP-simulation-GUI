#!/usr/bin/env python
"""
macOS-specific build script for SSVEP Simulation
Creates a proper macOS .app bundle with specific data files included
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# Check if running on macOS
import platform
if platform.system() != "Darwin":
    print("This script is for macOS only. Please use build_executable.py on Windows.")
    sys.exit(1)

# Check if PyInstaller is installed
try:
    import PyInstaller
except ImportError:
    print("PyInstaller is not installed. Installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

print("Building macOS application bundle...")

# App name
app_name = "SSVEP_Simulation"

# Clean up previous build artifacts
if os.path.exists("build"):
    shutil.rmtree("build")
if os.path.exists("dist"):
    shutil.rmtree("dist")
if os.path.exists(f"{app_name}.spec"):
    os.remove(f"{app_name}.spec")

# Icon file - convert from png to icns if needed
icon_file = "icon.icns"
if not os.path.exists(icon_file):
    print("Icon file not found. Using default icon.")
    icon_arg = []
else:
    icon_arg = ["--icon", icon_file]

# Specify only the requested data files
specific_mat_files = [
    "data/subject_1_fvep_led_training_1.mat",
    "data/subject_1_fvep_led_training_2.mat",
    "data/subject_2_fvep_led_training_1.mat",
    "data/subject_2_fvep_led_training_2.mat"
]

# Also include the classInfo file needed for LED sequence
classinfo_file = "data/classInfo_4_5.m"

# Verify that the specified files exist
print("Checking for required data files:")
data_dir = os.path.join(os.getcwd(), "data")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"Created data directory: {data_dir}")

for file_path in specific_mat_files:
    if os.path.exists(file_path):
        print(f"✓ Found {file_path}")
    else:
        print(f"✗ Missing {file_path} - creating empty placeholder")
        # Create directory if needed
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # Create empty file as placeholder to prevent build errors
        with open(file_path, 'w') as f:
            f.write("")

# Create classInfo file if it doesn't exist
if not os.path.exists(classinfo_file):
    print(f"Creating placeholder {classinfo_file}")
    with open(classinfo_file, 'w') as f:
        f.write("% SSVEP Class Info for 4 LEDs, 5-second trials\n")
        f.write("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")

# Create a debugging script to help diagnose crashes
debug_script = """
import sys
import os
import traceback

def log_exception(exc_type, exc_value, exc_traceback):
    with open(os.path.expanduser("~/ssvep_error.log"), "a") as f:
        f.write("Exception occurred:\\n")
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)
    return sys.__excepthook__(exc_type, exc_value, exc_traceback)

sys.excepthook = log_exception
"""

with open("error_logger.py", "w") as f:
    f.write(debug_script)

# Create the spec file with bundle configuration
spec_content = f"""# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# Specify only the required data files
data_files = [
    ('{specific_mat_files[0]}', 'data'),
    ('{specific_mat_files[1]}', 'data'),
    ('{specific_mat_files[2]}', 'data'),
    ('{specific_mat_files[3]}', 'data'),
    ('{classinfo_file}', 'data')
]

a = Analysis(
    ['simulation.py'],
    pathex=[],
    binaries=[],
    datas=data_files,
    hiddenimports=[
        'pygame.mixer', 
        'numpy.core', 
        'matplotlib', 
        'scipy', 
        'mat73', 
        'h5py'
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=['error_logger.py'],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='{app_name}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Changed to True for debugging
    disable_windowed_traceback=False,
    argv_emulation=True,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='{app_name}',
)

app = BUNDLE(
    coll,
    name='{app_name}.app',
    icon='{icon_file}' if os.path.exists('{icon_file}') else None,
    bundle_identifier='com.ssvep.simulation',
    info_plist={{
        'CFBundleShortVersionString': '1.0.0',
        'NSHighResolutionCapable': 'True',
        'NSPrincipalClass': 'NSApplication',
        'CFBundleName': '{app_name}',
        'CFBundleDisplayName': 'SSVEP Simulation',
        'CFBundleGetInfoString': 'SSVEP Experiment Simulation',
    }},
)
"""

# Write the spec file
with open(f"{app_name}.spec", 'w') as f:
    f.write(spec_content)

print(f"Created custom spec file: {app_name}.spec")

# Build using the custom spec file with debugging enabled
build_cmd = [sys.executable, "-m", "PyInstaller", "--clean", f"{app_name}.spec"]

try:
    subprocess.check_call(build_cmd)
    print(f"\nBuild completed successfully!")
    
    # Check if app bundle was created
    app_path = f"dist/{app_name}.app"
    if os.path.exists(app_path):
        print(f"macOS application bundle created at: {app_path}")
        
        # Copy the specific mat files to the app bundle to ensure they're included
        app_data_dir = os.path.join(app_path, "Contents/Resources/data")
        os.makedirs(app_data_dir, exist_ok=True)
        
        for mat_file in specific_mat_files:
            if os.path.exists(mat_file):
                basename = os.path.basename(mat_file)
                dest_path = os.path.join(app_data_dir, basename)
                print(f"Copying {basename} to {dest_path}")
                shutil.copy2(mat_file, dest_path)
        
        # Copy classInfo file
        if os.path.exists(classinfo_file):
            basename = os.path.basename(classinfo_file)
            dest_path = os.path.join(app_data_dir, basename)
            print(f"Copying {basename} to {dest_path}")
            shutil.copy2(classinfo_file, dest_path)
        
        # Create a ZIP archive for distribution
        print("Creating ZIP archive for easy distribution...")
        shutil.make_archive(f"dist/{app_name}", 'zip', "dist", f"{app_name}.app")
        print(f"ZIP archive created at: dist/{app_name}.zip")
        
        print("\nTROUBLESHOOTING INFO:")
        print("- If the app crashes, check ~/ssvep_error.log for error details")
        print("- Try opening the app from Terminal with:")
        print(f"  open {app_path}")
        print("- Or run the app directly from Terminal to see console output:")
        print(f"  {app_path}/Contents/MacOS/{app_name}")
    else:
        print(f"Error: Application bundle was not created at {app_path}")
        sys.exit(1)
        
except subprocess.CalledProcessError as e:
    print(f"Build failed with error code {e.returncode}")
    sys.exit(e.returncode) 