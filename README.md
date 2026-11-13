# Ultra Strict Media Sorter — WD EVA02 Large v3

This project builds a portable Windows application using the official **SmilingWolf WD EVA02 Large Tagger v3** ONNX model.

## What GitHub Actions produces

The GitHub Actions workflow builds the Windows application, stages all runtime files, and uploads one GitHub Actions artifact named:

`UltraStrictSorter-WD-EVA02-Large-v3-Windows`

**When you download that artifact from the Actions run, GitHub gives you a ZIP file.** The ZIP contains:

- `UltraStrictSorter.exe` at the ZIP root
- `_internal/` with the bundled application runtime, ONNX Runtime, WD EVA02 model, tag CSV, and sorting configuration
- `README-RUN.txt` with extraction and startup instructions

The `_internal` folder must remain beside `UltraStrictSorter.exe`. This is a PyInstaller **ONEDIR** application packaged into a single ZIP artifact; do not extract only the EXE or move it away from `_internal`. Python does not need to be installed on the target Windows computer.

## Build it entirely through GitHub

1. Create an empty GitHub repository.
2. Extract this source ZIP and put its contents directly in the repository's top-level folder. Commit and push the files and folders, including `.github/workflows/build.yml`. Do not leave the project inside an extra nested subfolder; GitHub must see `.github/workflows/build.yml` at the repository root.
3. Push the project to the `main` or `master` branch. The workflow also supports manual runs.
4. In GitHub, open the repository's **Actions** tab.
5. Select **Build Windows ZIP - WD EVA02 Large v3**. For a manual build, choose **Run workflow**.
6. Open the completed workflow run and download the artifact named **UltraStrictSorter-WD-EVA02-Large-v3-Windows**.
7. Extract the downloaded ZIP and double-click `UltraStrictSorter.exe`.

GitHub's Windows runner downloads the pinned model, verifies its SHA-256 hash and tag-list count, builds the app, stages the executable and all runtime files, and uploads them as a ZIP artifact. No local Python installation or local model download is needed to create the application.

## Model information

- Model: `SmilingWolf/wd-eva02-large-tagger-v3`
- Pinned Hugging Face revision: `fa2b83fd2d68b3d994e22fed5859d7cc99e1cdb0`
- Input resolution: 448 × 448
- Output vocabulary: 10,861 tags
- Runtime: ONNX Runtime (CPU package by default)
- Model file: approximately 1.26 GB
- License stated by upstream: Apache-2.0

The workflow downloads the model from the pinned revision; it is not committed to the Git repository.

## Build locally (optional)

On 64-bit Windows with Python 3.11:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python download_model.py
python verify_model.py
pyinstaller --clean --noconfirm app.spec
python stage_release.py --dist-dir dist/UltraStrictSorter --output-dir release
```

The staged application folder is written to `release\UltraStrictSorter`. GitHub Actions uploads that folder as a ZIP artifact automatically.

## Sorting behavior retained

The original output structure and `config/tag_rules.json` classification rules are retained:

- `sfw`
- `sfw/furry`
- `sfw/animated`
- `nsfw/boys/yaoi`
- `nsfw/boys/sex`
- `nsfw/girls/lesbian`
- `nsfw/girls/sex`
- `nsfw/furry/sex`
- `nsfw/animated`
- `duplicates`

The original thresholds and lists are unchanged. A warning is logged for configured rule tags that are not in WD EVA02's tag vocabulary. Those rules cannot match until their tag names are adjusted.

## Video handling

The sorter samples up to one frame per second from the first 10 seconds of a video. GIF files are sampled using Pillow; MP4/WebM and other supported formats are read through OpenCV/FFmpeg. Codec support depends on the FFmpeg build bundled with OpenCV.

## Important performance note

WD EVA02 Large v3 is a large model. The default build uses ONNX Runtime CPU execution for broad compatibility. It may take time to load and can be slower than smaller taggers, especially on CPU.

## Upstream model

https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3

Review the upstream model license and terms before redistributing the built application and model.
