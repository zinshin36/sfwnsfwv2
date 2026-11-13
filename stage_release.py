"""Stage and verify the PyInstaller build before GitHub uploads it as an artifact.

GitHub Actions automatically delivers uploaded artifacts as ZIP downloads.
Uploading this staged folder directly avoids an unnecessary nested ZIP.
"""
from __future__ import annotations

import argparse
import sys
import shutil
from pathlib import Path

APP_EXE = "UltraStrictSorter.exe"
MODEL_RELATIVE = Path("_internal") / "model" / "model.onnx"
TAGS_RELATIVE = Path("_internal") / "model" / "selected_tags.csv"
RULES_RELATIVE = Path("_internal") / "config" / "tag_rules.json"
STAGED_DIR_NAME = "UltraStrictSorter"
MIN_MODEL_BYTES = 1_000_000_000

RUN_README = """Ultra Strict Media Sorter - WD EVA02 Large v3
================================================

HOW TO RUN
1. Extract every file and folder from this ZIP to a normal folder.
2. Keep UltraStrictSorter.exe beside the _internal folder.
3. Double-click UltraStrictSorter.exe.
4. Click 'Select Folder & Scan' and choose the folder you want to scan.

IMPORTANT
- Do not move UltraStrictSorter.exe away from the _internal folder.
- Do not delete or rename files inside _internal. It contains the application
  runtime, WD EVA02 Large v3 model, tag list, and configuration.
- The first model load can take a little while because this is a large model.
- The application runs its model with ONNX Runtime CPU support by default.
- Scan results are moved into folders inside the folder you select.

MODEL
WD EVA02 Large Tagger v3 by SmilingWolf
https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3
Model license: Apache-2.0 (see the upstream repository for license terms).

This is a portable Windows ONEDIR application. Python does not need to be
installed on the computer that runs the extracted application.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dist-dir",
        type=Path,
        default=Path("dist") / "UltraStrictSorter",
        help="PyInstaller ONEDIR output directory (default: dist/UltraStrictSorter)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("release"),
        help="Directory where the staged application folder will be written (default: release)",
    )
    return parser.parse_args()


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    if path.stat().st_size <= 0:
        raise RuntimeError(f"{description} is empty: {path}")


def main() -> int:
    args = parse_args()
    dist_dir = args.dist_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not dist_dir.is_dir():
        raise NotADirectoryError(
            f"PyInstaller output folder was not found: {dist_dir}. Build app.spec first."
        )

    exe_path = dist_dir / APP_EXE
    model_path = dist_dir / MODEL_RELATIVE
    tags_path = dist_dir / TAGS_RELATIVE
    rules_path = dist_dir / RULES_RELATIVE

    require_file(exe_path, "application executable")
    require_file(model_path, "bundled WD EVA02 model")
    require_file(tags_path, "bundled model tag CSV")
    require_file(rules_path, "sorting rules")

    model_size = model_path.stat().st_size
    if model_size < MIN_MODEL_BYTES:
        raise RuntimeError(
            "The bundled model is unexpectedly small "
            f"({model_size:,} bytes). Refusing to package a possible Git LFS pointer "
            "or incomplete model download."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    staged_dir = output_dir / STAGED_DIR_NAME

    if staged_dir.exists():
        shutil.rmtree(staged_dir)
    shutil.copytree(dist_dir, staged_dir)
    (staged_dir / "README-RUN.txt").write_text(RUN_README, encoding="utf-8")

    # Confirm the staged folder has the files required to run on another PC.
    staged_required = [
        staged_dir / APP_EXE,
        staged_dir / MODEL_RELATIVE,
        staged_dir / TAGS_RELATIVE,
        staged_dir / RULES_RELATIVE,
        staged_dir / "README-RUN.txt",
    ]
    for path in staged_required:
        require_file(path, "staged release file")

    staged_model_size = (staged_dir / MODEL_RELATIVE).stat().st_size
    if staged_model_size != model_size:
        shutil.rmtree(staged_dir, ignore_errors=True)
        raise RuntimeError("The model size changed while staging the release folder.")

    total_bytes = sum(path.stat().st_size for path in staged_dir.rglob("*") if path.is_file())
    file_count = sum(1 for path in staged_dir.rglob("*") if path.is_file())

    print("Windows application folder staged successfully.")
    print(f"Staged folder: {staged_dir}")
    print(f"Files: {file_count:,}")
    print(f"Total size: {total_bytes:,} bytes")
    print(f"Model size: {model_size:,} bytes")
    print("Executable: UltraStrictSorter.exe")
    print("GitHub Actions will upload this folder as a ZIP artifact.")
    print("Keep the _internal folder beside the executable after extraction.")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"STAGING FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
