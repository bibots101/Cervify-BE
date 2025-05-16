# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['main.py'],
    pathex=[
        '.',
        './external/dinov2/'
    ],
    binaries=[],
    datas=[
        ('models/yolo/yolo_model.pt', 'models/yolo'),
        ('models/dinov2_vitl14_full.pt', 'models'),
        ('models/SVM/scaler_pca.joblib', 'models/SVM'),
        ('models/SVM/pca_model.joblib', 'models/SVM'),
        ('models/SVM/svm_scaler.joblib', 'models/SVM'),
        ('models/SVM/svm_model.joblib', 'models/SVM'),
        ('models/mlp/mlp_model.keras', 'models/mlp'),
        ('models/mlp/pca_model.joblib', 'models/mlp'),
        ('models/mlp/scaler_pca.joblib', 'models/mlp'),
        ('models/mlp/scaler_classifer.joblib', 'models/mlp'),
    ],
    hiddenimports=[
        'pipeline',
        'models',
        'utils',
        'dinov2',
        'dinov2.configs',
        'dinov2.data',
        'dinov2.data.adapters',
        'dinov2.models',
        'dinov2.hub',
        'dinov2.utils',
        'click', 
        'six', 
        'charset_normalizer'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['pytest', 'pyinstaller', 'PyQt5'],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Cervify_backend',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Cervify_backend',
)
