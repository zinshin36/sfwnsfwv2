"""Validate the local WD EVA02 Large v3 ONNX model and CSV tag file."""
import csv
from pathlib import Path
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "model" / "model.onnx"
TAGS = ROOT / "model" / "selected_tags.csv"
EXPECTED_TAGS = 10861


def main() -> int:
    for path in (MODEL, TAGS):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required model file: {path}. Run python download_model.py")

    with TAGS.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "name" not in reader.fieldnames or "category" not in reader.fieldnames:
            raise RuntimeError(f"Unexpected selected_tags.csv header: {reader.fieldnames}")
        tag_count = sum(1 for row in reader if (row.get("name") or "").strip())

    if tag_count != EXPECTED_TAGS:
        raise RuntimeError(f"Expected {EXPECTED_TAGS} tags; found {tag_count}")

    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if not inputs or not outputs:
        raise RuntimeError("The ONNX model has no input or output tensors")

    print("WD EVA02 Large v3 files verified.")
    print(f"Model size: {MODEL.stat().st_size / (1024**3):.2f} GiB")
    print(f"Tags: {tag_count:,}")
    print(f"Input: {inputs[0].shape}")
    print(f"Output: {outputs[0].shape}")
    print(f"Providers: {session.get_providers()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
