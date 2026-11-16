"""Wrapper around the vendored OMRChecker (https://github.com/Udayraj123/OMRChecker).

OMRChecker is not a pip package; it is imported from vendor/OMRChecker as the
top-level `src` package via its programmatic entry point `src.entry.entry_point`.
"""

import csv
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
OMRCHECKER_DIR = PROJECT_DIR / "vendor" / "OMRChecker"
TEMPLATES_DIR = PROJECT_DIR / "templates"

if str(OMRCHECKER_DIR) not in sys.path:
    sys.path.insert(0, str(OMRCHECKER_DIR))

from src.entry import entry_point  # noqa: E402  (OMRChecker)


def read_tables(pages: list[tuple[Path, str]]) -> dict[str, dict[str, dict[str, str]]]:
    """Run OMRChecker over (image path, page type) pairs.

    Returns {page type: {image file name: {column: value}}}.
    """
    with tempfile.TemporaryDirectory(prefix="omr_") as tmp:
        tmp_dir = Path(tmp)
        input_dir = tmp_dir / "input"
        output_dir = tmp_dir / "output"

        # OMRChecker uses the template.json found in each folder it processes, so each
        # page type gets its own subfolder: input/<page type>/{template.json, images}.
        # Only template.json is copied; reference.png would otherwise be read as a form.
        for image_path, page_type in pages:
            page_dir = input_dir / page_type
            page_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(TEMPLATES_DIR / page_type / "template.json", page_dir / "template.json")
            shutil.copy(TEMPLATES_DIR / page_type / "config.json", page_dir / "config.json")
            shutil.copy(image_path, page_dir / image_path.name)

        entry_point(input_dir, {"output_dir": str(output_dir), "setLayout": False})

        # Results are written per subfolder: output/<page type>/Results/Results_<hour>.csv
        results: dict[str, dict[str, dict[str, str]]] = {}
        for page_type in {page_type for _, page_type in pages}:
            page_results = results.setdefault(page_type, {})
            for csv_path in (output_dir / page_type / "Results").glob("Results_*.csv"):
                with open(csv_path, newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        file_id = row.pop("file_id")
                        for key in ("input_path", "output_path", "score"):
                            row.pop(key, None)
                        page_results[file_id] = row
        return results
