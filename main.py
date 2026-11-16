"""Load the test form pages, read tables (OMRChecker) and text (TrOCR), write output.json."""

import json
from pathlib import Path

import cv2
from PIL import Image

from aligner import align_page
from omr_reader import read_tables
from page_identifier import identify_page
from text_reader import TextReader

PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = PROJECT_DIR / "test_images"
IMAGE_NAMES = ["page1.jpg", "page2.jpg", "page3.jpg"]
ALIGNED_DIR = PROJECT_DIR / "aligned"  # aligned pages, kept for inspection
OUTPUT_PATH = PROJECT_DIR / "output.json"


def clear_aligned_dir() -> None:
    """Delete the previous run's aligned images before doing any work."""
    ALIGNED_DIR.mkdir(exist_ok=True)
    for path in ALIGNED_DIR.iterdir():
        try:
            path.unlink()
        except PermissionError as e:
            raise RuntimeError(f"Cannot delete '{path}'. Is it open in another program?") from e


def main() -> None:
    image_paths = [IMAGE_DIR / name for name in IMAGE_NAMES]
    for path in image_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    clear_aligned_dir()

    # Identify each page, then warp it onto that page's reference image.
    pages = []
    for path in image_paths:
        page_type = identify_page(path)
        alignment = align_page(path, page_type)
        aligned_path = ALIGNED_DIR / f"{path.stem}.png"
        # imwrite returns False on failure instead of raising.
        if not cv2.imwrite(str(aligned_path), alignment.image):
            raise RuntimeError(f"Could not write '{aligned_path}'")
        print(
            f"{path.name}: {page_type}, {alignment.matches} matches, {alignment.inliers} inliers, "
            f"{alignment.control_points} TPS control points"
        )
        pages.append({"source": path, "page_type": page_type, "aligned_path": aligned_path, "alignment": alignment})

    # Both readers work on the aligned images.
    omr_results = read_tables([(p["aligned_path"], p["page_type"]) for p in pages])

    text_reader = TextReader()
    output_pages = []
    for p in pages:
        with Image.open(p["aligned_path"]) as image:
            text = text_reader.read(image)
        output_pages.append(
            {
                "page": p["source"].name,
                "page_type": p["page_type"],
                "alignment": {"matches": p["alignment"].matches, "inliers": p["alignment"].inliers},
                "tables": omr_results.get(p["page_type"], {}).get(p["aligned_path"].name, {}),
                "text": text,
            }
        )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"pages": output_pages}, f, indent=2)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
