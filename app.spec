# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH)
MODEL_FILE = ROOT / "model" / "model.onnx"
TAGS_FILE = ROOT / "model" / "selected_tags.csv"
RULES_FILE = ROOT / "config" / "tag_rules.json"

for required in (MODEL_FILE, TAGS_FILE, RULES_FILE):
    if not required.is_file():
        raise FileNotFoundError(
            f"Required build file is missing: {required}\n"
            "Run 'python download_model.py' before building."
        )

ort_datas, ort_bins, ort_hidden = collect_all("onnxruntime")
numpy_datas, numpy_bins, numpy_hidden = collect_all("numpy")
cv2_datas, cv2_bins, cv2_hidden = collect_all("cv2")
pil_datas, pil_bins, pil_hidden = collect_all("PIL")

datas = ort_datas + numpy_datas + cv2_datas + pil_datas
binaries = ort_bins + numpy_bins + cv2_bins + pil_bins
hiddenimports = ort_hidden + numpy_hidden + cv2_hidden + pil_hidden

datas += [
    (str(MODEL_FILE), "model"),
    (str(TAGS_FILE), "model"),
    (str(RULES_FILE), "config"),
]

a = Analysis(
    [str(ROOT / "app.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tensorflow", "keras", "deepdanbooru"],
    cipher=None,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="UltraStrictSorter",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    contents_directory="_internal",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="UltraStrictSorter",
)
