"""Download the official, pinned WD EVA02 Large v3 ONNX model and tag CSV."""
import csv
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

REPOSITORY = "SmilingWolf/wd-eva02-large-tagger-v3"
REVISION = "fa2b83fd2d68b3d994e22fed5859d7cc99e1cdb0"
BASE_URL = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}"
MODEL_URL = f"{BASE_URL}/model.onnx?download=true"
TAGS_URL = f"{BASE_URL}/selected_tags.csv?download=true"
MODEL_PATH = MODEL_DIR / "model.onnx"
TAGS_PATH = MODEL_DIR / "selected_tags.csv"
EXPECTED_MODEL_SHA256 = "9e768793060c7939b277ccb382783e8670e8a042d29d77aa736be0c8cc898bfc"
EXPECTED_TAGS = 10861
CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, expected_sha256: str | None = None) -> None:
    if destination.exists():
        if expected_sha256 is None or sha256_file(destination) == expected_sha256:
            print(f"Already present: {destination}")
            return
        print(f"Existing file failed SHA-256 check; downloading again: {destination}")
        destination.unlink()

    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "UltraStrictSorter-WD-EVA02/1.0"},
    )
    print(f"Downloading {destination.name}...")

    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            total = int(response.headers.get("Content-Length", "0") or 0)
            downloaded = 0
            last_reported = -1
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    percent = int(downloaded * 100 / total)
                    if percent >= last_reported + 5:
                        print(f"  {percent}% ({downloaded / (1024**3):.2f} GiB)")
                        last_reported = percent
                elif downloaded % (100 * CHUNK_SIZE) < CHUNK_SIZE:
                    print(f"  {downloaded / (1024**3):.2f} GiB downloaded")

        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    print(f"Downloaded: {destination} ({destination.stat().st_size:,} bytes)")

    if expected_sha256:
        actual = sha256_file(destination)
        if actual.lower() != expected_sha256.lower():
            destination.unlink(missing_ok=True)
            raise RuntimeError(
                f"SHA-256 mismatch for {destination.name}.\n"
                f"Expected: {expected_sha256}\nActual:   {actual}"
            )
        print(f"SHA-256 verified: {actual}")


def validate_tags(path: Path) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"name", "category"}.issubset(reader.fieldnames):
            raise RuntimeError(f"Unexpected tag CSV header: {reader.fieldnames}")
        count = sum(1 for row in reader if (row.get("name") or "").strip())

    if count != EXPECTED_TAGS:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"The selected_tags.csv file contains {count:,} tags; expected {EXPECTED_TAGS:,}. "
            "The downloaded files may not match the pinned model revision."
        )
    print(f"Tag CSV verified: {count:,} tags")


def main() -> int:
    try:
        download(MODEL_URL, MODEL_PATH, EXPECTED_MODEL_SHA256)
        download(TAGS_URL, TAGS_PATH)
        validate_tags(TAGS_PATH)
        print("\nWD EVA02 Large v3 model files are ready.")
        return 0
    except Exception as error:
        print(f"\nMODEL DOWNLOAD FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
