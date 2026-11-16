"""Decide which form page (template folder under templates/) an image is.

Test version: the image file name must match a template folder, e.g. page1.jpg -> page1.
To be replaced by matching the image against each templates/<page>/reference.png.
"""

from pathlib import Path

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def identify_page(image_path: Path) -> str:
    page_type = image_path.stem
    if not (TEMPLATES_DIR / page_type / "template.json").is_file():
        raise ValueError(f"No template for '{image_path.name}' (expected templates/{page_type}/template.json)")
    return page_type
