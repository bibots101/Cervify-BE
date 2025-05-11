# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['main.py'],
    pathex=[
        'C:/Users/wassi/Desktop/cervify_build/backend',
        'C:/Users/wassi/Desktop/cervify_build/backend/external/dinov2/'
    ],
    binaries=[],
    datas=[
        ('models/yolo/yolo_model.pt', 'models/yolo'),
        ('models/dinov2_vitl14_full.pt', 'models'),
        ('models/SVM/scaler_pca.joblib', 'models/SVM'),
        ('models/SVM/pca_model.joblib', 'models/SVM'),
        ('models/SVM/svm_scaler.joblib', 'models/SVM'),
        ('models/SVM/svm_model.joblib', 'models/SVM'),
        ('models/mlp/mlp_model.h5', 'models/mlp'),
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
        'dinov2.evaluation',
        'dinov2.evaluation.evaluator',
        ],
    hookspath=['./hooks'],
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
    a.binaries,
    a.datas,
    [],
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
