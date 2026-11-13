import csv
import hashlib
import json
import logging
import os
import shutil
import sys
import threading
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import tkinter as tk
from PIL import Image, ImageOps, ImageSequence
from tkinter import filedialog, messagebox, ttk


# ============================================================
# APPLICATION SETTINGS
# ============================================================

APP_TITLE = "Ultra Strict Media Sorter - WD EVA02 Large v3"
MODEL_FILENAME = "model.onnx"
TAGS_FILENAME = "selected_tags.csv"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".gif", ".mp4", ".webm", ".mkv", ".avi", ".mov"}
VIDEO_SAMPLE_SECONDS = 10
MODEL_TAG_COUNT = 10861


# ============================================================
# RESOURCE PATHS (SOURCE + PYINSTALLER ONEDIR SAFE)
# ============================================================

def resource_path(relative_path: str) -> Path:
    """Resolve bundled data under PyInstaller or beside this source file."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / relative_path


MODEL_PATH = resource_path(str(Path("model") / MODEL_FILENAME))
TAGS_PATH = resource_path(str(Path("model") / TAGS_FILENAME))
RULES_PATH = resource_path(str(Path("config") / "tag_rules.json"))


# ============================================================
# APPLICATION STATE
# ============================================================

root: Optional[tk.Tk] = None
scan_button: Optional[ttk.Button] = None
progress_bar: Optional[ttk.Progressbar] = None
status_var: Optional[tk.StringVar] = None

RULES = {}
TH = {}
model_session: Optional[ort.InferenceSession] = None
model_lock = threading.Lock()
model_loaded = False
model_input_name = ""
model_input_layout = "NHWC"
model_target_size = 448
tags: List[str] = []
tag_categories: List[int] = []


# ============================================================
# LOGGING + UI THREAD HELPERS
# ============================================================

def setup_logger(scan_folder: Path) -> None:
    log_dir = scan_folder / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "scan.log"

    logging.basicConfig(
        filename=str(log_file),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
        encoding="utf-8",
    )
    logging.info("===== NEW WD EVA02 LARGE V3 SCAN SESSION =====")
    logging.info("Model: %s", MODEL_PATH)
    logging.info("Tags: %s", TAGS_PATH)


def ui_call(callback, *args) -> None:
    """Schedule UI work on Tk's main thread."""
    if root is not None:
        root.after(0, lambda: callback(*args))


def set_status(message: str) -> None:
    if status_var is not None:
        status_var.set(message)


def show_error(title: str, message: str) -> None:
    if root is not None:
        messagebox.showerror(title, message, parent=root)


# ============================================================
# LOAD CLASSIFICATION RULES
# ============================================================

def load_rules() -> None:
    global RULES, TH

    if not RULES_PATH.is_file():
        raise FileNotFoundError(
            f"Could not find the classification rules file:\n{RULES_PATH}"
        )

    with RULES_PATH.open("r", encoding="utf-8") as file:
        RULES = json.load(file)

    required_sections = {
        "thresholds", "rating", "force_nsfw", "force_furry",
        "force_yaoi", "force_lesbian", "girls_explicit", "boys_explicit",
    }
    missing = required_sections.difference(RULES)
    if missing:
        raise ValueError("tag_rules.json is missing: " + ", ".join(sorted(missing)))

    TH = RULES["thresholds"]
    for key in ("high", "medium"):
        if key not in TH:
            raise ValueError(f"tag_rules.json thresholds is missing '{key}'")


# ============================================================
# MODEL + CSV TAG LOADING
# ============================================================

def load_model_once() -> bool:
    """Load the WD EVA02 ONNX model and its exact, ordered tag CSV once."""
    global model_session, model_loaded, model_input_name
    global model_input_layout, model_target_size, tags, tag_categories

    with model_lock:
        if model_loaded:
            return True

        missing_files = [str(path) for path in (MODEL_PATH, TAGS_PATH) if not path.is_file()]
        if missing_files:
            raise FileNotFoundError(
                "The WD EVA02 model files are missing. Run this from the project folder:\n\n"
                "python download_model.py\n\nMissing files:\n" + "\n".join(missing_files)
            )

        loaded_tags: List[str] = []
        loaded_categories: List[int] = []

        with TAGS_PATH.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames or "name" not in reader.fieldnames or "category" not in reader.fieldnames:
                raise ValueError(
                    "selected_tags.csv must contain 'name' and 'category' columns. "
                    f"Found columns: {reader.fieldnames}"
                )

            for row in reader:
                name = (row.get("name") or "").strip()
                if not name:
                    continue
                try:
                    category = int(row.get("category", "-1"))
                except (TypeError, ValueError):
                    category = -1
                # Keep the raw Danbooru tag spelling (underscores included) so
                # tag_rules.json can match tags exactly.
                loaded_tags.append(name)
                loaded_categories.append(category)

        if len(loaded_tags) != MODEL_TAG_COUNT:
            raise ValueError(
                f"Tag CSV contains {len(loaded_tags):,} tags, but this model expects "
                f"{MODEL_TAG_COUNT:,}. Do not mix model.onnx and selected_tags.csv "
                "from different model revisions. Run download_model.py again."
            )

        available_providers = ort.get_available_providers()
        requested_providers = []
        if "CUDAExecutionProvider" in available_providers:
            requested_providers.append("CUDAExecutionProvider")
        requested_providers.append("CPUExecutionProvider")

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = max(1, (os.cpu_count() or 2) - 1)
        session_options.inter_op_num_threads = 1

        session = ort.InferenceSession(
            str(MODEL_PATH),
            sess_options=session_options,
            providers=requested_providers,
        )

        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if not inputs or not outputs:
            raise RuntimeError("The ONNX model does not expose an input and output tensor.")

        input_shape = inputs[0].shape
        if len(input_shape) != 4:
            raise RuntimeError(f"Unexpected model input shape: {input_shape}")

        # The official EVA02 ONNX model uses NHWC [batch, 448, 448, 3].
        # The layout check also gives a clear error if the downloaded model differs.
        if input_shape[-1] == 3:
            layout = "NHWC"
            height = input_shape[1]
            width = input_shape[2]
        elif input_shape[1] == 3:
            layout = "NCHW"
            height = input_shape[2]
            width = input_shape[3]
        else:
            raise RuntimeError(f"Cannot identify image channel layout for input shape {input_shape}")

        if isinstance(height, int) and isinstance(width, int) and height != width:
            raise RuntimeError(f"Expected a square model input, received {height}x{width}")
        size_value = height if isinstance(height, int) else width
        if not isinstance(size_value, int):
            size_value = 448

        output_shape = outputs[0].shape
        output_count = output_shape[-1] if output_shape and isinstance(output_shape[-1], int) else None
        if output_count is not None and output_count != len(loaded_tags):
            raise RuntimeError(
                f"Model returns {output_count:,} outputs but the CSV has {len(loaded_tags):,} tags."
            )

        model_session = session
        model_input_name = inputs[0].name
        model_input_layout = layout
        model_target_size = size_value
        tags = loaded_tags
        tag_categories = loaded_categories
        model_loaded = True

        logging.info("WD EVA02 Large v3 loaded successfully.")
        logging.info("Input shape: %s; layout: %s; target size: %s", input_shape, layout, size_value)
        logging.info("Providers: %s", session.get_providers())
        logging.info("Loaded tag count: %s", len(tags))

        configured_rule_tags = set()
        configured_rule_tags.add(RULES["rating"]["safe"])
        configured_rule_tags.add(RULES["rating"]["explicit"])
        for key in (
            "force_nsfw", "force_furry", "force_yaoi", "force_lesbian",
            "girls_explicit", "boys_explicit",
        ):
            configured_rule_tags.update(RULES.get(key, []))
        configured_rule_tags.update(("1boy", "1girl"))
        missing_rule_tags = sorted(configured_rule_tags.difference(set(tags)))
        if missing_rule_tags:
            logging.warning(
                "Some configured rule tags are not in this model's vocabulary: %s",
                ", ".join(missing_rule_tags),
            )

        return True


# ============================================================
# IMAGE PREPROCESSING (OFFICIAL WD TAGGER FORMAT)
# ============================================================

def prepare_pil_image(image: Image.Image) -> np.ndarray:
    """White-composite, square-pad, bicubic-resize, then RGB -> BGR float32."""
    image = ImageOps.exif_transpose(image)

    # Composite transparency onto white, matching the official WD tagger.
    rgba = image.convert("RGBA")
    white_background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    rgb_image = Image.alpha_composite(white_background, rgba).convert("RGB")

    width, height = rgb_image.size
    if width <= 0 or height <= 0:
        raise ValueError("Image has invalid dimensions.")

    max_dimension = max(width, height)
    left = (max_dimension - width) // 2
    top = (max_dimension - height) // 2

    square = Image.new("RGB", (max_dimension, max_dimension), (255, 255, 255))
    square.paste(rgb_image, (left, top))

    if max_dimension != model_target_size:
        resampling = getattr(Image, "Resampling", Image).BICUBIC
        square = square.resize((model_target_size, model_target_size), resampling)

    # The ONNX model expects float32 BGR pixels in the original 0..255 range.
    image_array = np.asarray(square, dtype=np.float32)
    image_array = image_array[:, :, ::-1].copy()

    if model_input_layout == "NCHW":
        image_array = np.transpose(image_array, (2, 0, 1))

    return image_array


def predict_frames(frames: Sequence[Image.Image]) -> Dict[str, float]:
    if model_session is None:
        raise RuntimeError("The model has not been loaded.")

    combined: Dict[str, float] = {}

    # Run one frame per inference call to keep peak RAM use reasonable with
    # this 1.26 GB large model. Per-tag max across sampled frames is retained.
    for frame in frames:
        prepared = prepare_pil_image(frame)
        batch = np.expand_dims(prepared, axis=0).astype(np.float32, copy=False)
        raw_output = model_session.run(None, {model_input_name: batch})[0]
        prediction = np.asarray(raw_output, dtype=np.float32).reshape(-1)

        if prediction.size != len(tags):
            raise RuntimeError(
                f"Model returned {prediction.size} scores, but {len(tags)} tags are loaded."
            )

        for index, score in enumerate(prediction):
            tag = tags[index]
            value = float(score)
            if value > combined.get(tag, 0.0):
                combined[tag] = value

    return combined


# ============================================================
# FILE HASHING + SAFE MOVE
# ============================================================

def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_destination(destination: Path) -> Path:
    """Avoid silently overwriting an existing file with the same name."""
    if not destination.exists():
        return destination

    stem = destination.stem
    suffix = destination.suffix
    parent = destination.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def move_file(source: Path, destination_folder: Path) -> Path:
    destination_folder.mkdir(parents=True, exist_ok=True)
    destination = unique_destination(destination_folder / source.name)
    return Path(shutil.move(str(source), str(destination)))


# ============================================================
# IMAGE + VIDEO LOADING
# ============================================================

def load_still_image(path: Path) -> Image.Image:
    with Image.open(path) as source:
        source.load()
        return source.copy()


def extract_gif_frames(path: Path) -> List[Image.Image]:
    frames: List[Image.Image] = []
    next_sample_ms = 0
    elapsed_ms = 0

    with Image.open(path) as gif:
        for frame in ImageSequence.Iterator(gif):
            if elapsed_ms >= next_sample_ms and next_sample_ms < VIDEO_SAMPLE_SECONDS * 1000:
                frames.append(frame.copy())
                next_sample_ms += 1000
            duration = frame.info.get("duration", gif.info.get("duration", 100))
            try:
                duration = max(1, int(duration))
            except (TypeError, ValueError):
                duration = 100
            elapsed_ms += duration
            if next_sample_ms >= VIDEO_SAMPLE_SECONDS * 1000:
                break

    if not frames:
        with Image.open(path) as gif:
            frames.append(gif.copy())
    return frames


def extract_video_frames(path: Path) -> List[Image.Image]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            "OpenCV could not open this video. The installed OpenCV/FFmpeg build may "
            "not support its codec or container."
        )

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or not np.isfinite(fps):
        fps = 24.0

    frames: List[Image.Image] = []
    for second in range(VIDEO_SAMPLE_SECONDS):
        capture.set(cv2.CAP_PROP_POS_MSEC, second * 1000.0)
        success, frame = capture.read()
        if not success or frame is None:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))

    capture.release()
    if not frames:
        raise RuntimeError("No readable frames were found in the video.")
    return frames


def load_media_frames(path: Path) -> List[Image.Image]:
    extension = path.suffix.lower()
    if extension == ".gif":
        return extract_gif_frames(path)
    if extension in VIDEO_EXTENSIONS:
        return extract_video_frames(path)
    return [load_still_image(path)]


# ============================================================
# STRICT CLASSIFIER (EXISTING RULES PRESERVED)
# ============================================================

def tag_high(scores: Dict[str, float], tag: str) -> bool:
    return scores.get(tag, 0.0) >= float(TH["high"])


def tag_medium(scores: Dict[str, float], tag: str) -> bool:
    return scores.get(tag, 0.0) >= float(TH["medium"])


def classify(scores: Dict[str, float], extension: str) -> str:
    logging.info(
        "TOP TAGS: %s",
        sorted(scores.items(), key=lambda item: item[1], reverse=True)[:30],
    )

    nsfw = False

    # RATING CHECK
    if tag_medium(scores, RULES["rating"]["explicit"]):
        nsfw = True
    elif tag_high(scores, RULES["rating"]["safe"]):
        nsfw = False

    # FORCE NSFW TAGS
    for tag in RULES["force_nsfw"]:
        if tag_medium(scores, tag):
            nsfw = True

    # GIRLS EXPLICIT
    for tag in RULES["girls_explicit"]:
        if tag_medium(scores, tag):
            return "nsfw/girls/sex"

    # BOYS EXPLICIT
    for tag in RULES["boys_explicit"]:
        if tag_medium(scores, tag):
            return "nsfw/boys/sex"

    # FURRY REQUIRES 2 TAGS
    furry_hits = sum(1 for tag in RULES["force_furry"] if tag_medium(scores, tag))
    if furry_hits >= 2:
        return "nsfw/furry/sex" if nsfw else "sfw/furry"

    # YAOI
    for tag in RULES["force_yaoi"]:
        if tag_medium(scores, tag):
            return "nsfw/boys/yaoi"

    # LESBIAN (existing rule retained exactly)
    for tag in RULES["force_lesbian"]:
        if tag_medium(scores, tag):
            return "nsfw/girls/lesbian"

    # GENDER DOMINANCE
    boy = scores.get("1boy", 0.0)
    girl = scores.get("1girl", 0.0)

    if boy >= 0.65 and boy > girl + 0.25:
        return "nsfw/boys/sex"

    if girl >= 0.65 and girl > boy + 0.25:
        return "nsfw/girls/sex"

    # VIDEO
    if extension in VIDEO_EXTENSIONS:
        return "nsfw/animated" if nsfw else "sfw/animated"

    # FINAL
    if nsfw:
        return "nsfw/girls/sex"

    return "sfw"


# ============================================================
# OUTPUT FOLDER STRUCTURE + FILE DISCOVERY
# ============================================================

OUTPUT_DIRECTORIES = {"sfw", "nsfw", "duplicates"}


def create_structure(base: Path) -> None:
    paths = [
        "sfw",
        "sfw/furry",
        "sfw/animated",
        "nsfw/boys/yaoi",
        "nsfw/boys/sex",
        "nsfw/girls/lesbian",
        "nsfw/girls/sex",
        "nsfw/furry/sex",
        "nsfw/animated",
        "duplicates",
    ]
    for relative_path in paths:
        (base / relative_path).mkdir(parents=True, exist_ok=True)


def discover_media(folder: Path) -> List[Path]:
    found: List[Path] = []
    for current_root, dirs, filenames in os.walk(folder):
        current_path = Path(current_root)
        if current_path.resolve() == folder.resolve():
            dirs[:] = [directory for directory in dirs if directory not in OUTPUT_DIRECTORIES]

        for filename in filenames:
            path = current_path / filename
            extension = path.suffix.lower()
            if extension in IMAGE_EXTENSIONS or extension in VIDEO_EXTENSIONS:
                found.append(path)

    found.sort(key=lambda item: str(item).casefold())
    return found


# ============================================================
# SCAN WORKER
# ============================================================

def scan_worker(folder_string: str) -> None:
    folder = Path(folder_string)
    try:
        setup_logger(folder)
        logging.info("Selected folder: %s", folder)

        ui_call(set_status, "Loading WD EVA02 Large v3 model...")
        load_model_once()
        create_structure(folder)

        files = discover_media(folder)
        total = len(files)
        ui_call(set_progress_max, total)

        if total == 0:
            logging.info("No supported media files found.")
            ui_call(finish_scan, "No supported image or video files were found.", False)
            return

        seen_hashes = set()
        processed = 0
        duplicates = 0
        errors = 0

        for index, path in enumerate(files, start=1):
            ui_call(set_status, f"Processing {index}/{total}: {path.name}")
            try:
                digest = file_hash(path)
                if digest in seen_hashes:
                    moved_to = move_file(path, folder / "duplicates")
                    duplicates += 1
                    logging.info("DUPLICATE: %s -> %s", path, moved_to)
                else:
                    seen_hashes.add(digest)
                    extension = path.suffix.lower()
                    frames = load_media_frames(path)
                    scores = predict_frames(frames)
                    if not scores:
                        raise RuntimeError("The model did not return any tag scores.")

                    destination = classify(scores, extension)
                    moved_to = move_file(path, folder / destination)
                    processed += 1
                    logging.info("MOVED: %s -> %s", path, moved_to)

            except Exception:
                errors += 1
                logging.error("FAILED: %s\n%s", path, traceback.format_exc())

            ui_call(set_progress_value, index)

        logging.info(
            "===== SCAN COMPLETE: classified=%s duplicates=%s errors=%s total=%s =====",
            processed, duplicates, errors, total,
        )
        summary = (
            f"Scan complete.\n\nClassified: {processed}\n"
            f"Duplicates: {duplicates}\nErrors: {errors}\nTotal files: {total}\n\n"
            f"Log: {folder / 'logs' / 'scan.log'}"
        )
        ui_call(finish_scan, summary, errors > 0)

    except Exception as error:
        logging.exception("SCAN ABORTED")
        ui_call(finish_scan, str(error), True)


def set_progress_max(total: int) -> None:
    if progress_bar is not None:
        progress_bar["maximum"] = max(1, total)
        progress_bar["value"] = 0


def set_progress_value(value: int) -> None:
    if progress_bar is not None:
        progress_bar["value"] = value


def finish_scan(message: str, is_error: bool) -> None:
    if scan_button is not None:
        scan_button.configure(state="normal")
    if status_var is not None:
        status_var.set("Ready")
    if is_error:
        messagebox.showwarning("Scan finished with issues", message, parent=root)
    else:
        messagebox.showinfo("Scan complete", message, parent=root)


# ============================================================
# GUI
# ============================================================

def start_scan() -> None:
    folder = filedialog.askdirectory(title="Select the folder to scan", parent=root)
    if not folder:
        return

    if scan_button is not None:
        scan_button.configure(state="disabled")
    if progress_bar is not None:
        progress_bar["value"] = 0

    worker = threading.Thread(target=scan_worker, args=(folder,), daemon=True)
    worker.start()


def main() -> None:
    global root, scan_button, progress_bar, status_var

    root = tk.Tk()
    root.title(APP_TITLE)
    root.geometry("620x235")
    root.resizable(False, False)

    try:
        load_rules()
    except Exception as error:
        messagebox.showerror("Configuration error", str(error), parent=root)
        root.destroy()
        return

    title = ttk.Label(root, text="Ultra Strict Media Sorter", font=("Segoe UI", 15, "bold"))
    title.pack(pady=(18, 4))

    subtitle = ttk.Label(root, text="WD EVA02 Large v3 · 10,861 tags · ONNX Runtime")
    subtitle.pack(pady=(0, 14))

    scan_button = ttk.Button(root, text="Select Folder & Scan", command=start_scan)
    scan_button.pack(pady=6, ipadx=14, ipady=5)

    progress_bar = ttk.Progressbar(root, length=550, mode="determinate")
    progress_bar.pack(pady=(14, 8))

    status_var = tk.StringVar(value="Ready")
    status_label = ttk.Label(root, textvariable=status_var, width=82, anchor="center")
    status_label.pack(pady=4)

    note = ttk.Label(root, text="The model is large. First-time model loading may take a little while.")
    note.pack(pady=(5, 0))

    root.mainloop()


if __name__ == "__main__":
    main()
