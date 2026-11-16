"""One-time setup: render each page of the original PDF to a blank reference image.

templates/page<N>/reference.png is used later to identify and align scanned pages.
Pixel coordinates in a reference image = PDF points * DPI / 72.
"""

from pathlib import Path

import pymupdf

PROJECT_DIR = Path(__file__).resolve().parent
PDF_PATH = PROJECT_DIR / "test_images" / "orig_doc.pdf"
TEMPLATES_DIR = PROJECT_DIR / "templates"
DPI = 200


def main() -> None:
    with pymupdf.open(PDF_PATH) as doc:
        for page in doc:
            out_dir = TEMPLATES_DIR / f"page{page.number + 1}"
            out_dir.mkdir(parents=True, exist_ok=True)
            pixmap = page.get_pixmap(dpi=DPI, colorspace=pymupdf.csGRAY, alpha=False)
            out_path = out_dir / "reference.png"
            pixmap.save(out_path)
            print(f"{out_path.relative_to(PROJECT_DIR)}: {pixmap.width}x{pixmap.height}")


if __name__ == "__main__":
    main()
