# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# Specify only the required data files
data_files = [
    ('data/subject_1_fvep_led_training_1.mat', 'data'),
    ('data/subject_1_fvep_led_training_2.mat', 'data'),
    ('data/subject_2_fvep_led_training_1.mat', 'data'),
    ('data/subject_2_fvep_led_training_2.mat', 'data'),
    ('data/classInfo_4_5.m', 'data')
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
    hooksconfig={},
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
    name='SSVEP_Simulation',
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
    name='SSVEP_Simulation',
)

app = BUNDLE(
    coll,
    name='SSVEP_Simulation.app',
    icon='icon.icns' if os.path.exists('icon.icns') else None,
    bundle_identifier='com.ssvep.simulation',
    info_plist={
        'CFBundleShortVersionString': '1.0.0',
        'NSHighResolutionCapable': 'True',
        'NSPrincipalClass': 'NSApplication',
        'CFBundleName': 'SSVEP_Simulation',
        'CFBundleDisplayName': 'SSVEP Simulation',
        'CFBundleGetInfoString': 'SSVEP Experiment Simulation',
    },
)
