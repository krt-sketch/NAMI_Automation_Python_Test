"""One-time setup: dump the text lines and vector drawings of each PDF page.

Writes templates/page<N>/layout.json. All coordinates are reference-image pixels
([x0, y0, x1, y1]) at the same DPI used for reference.png: pixel = PDF point * DPI / 72.
"""

import json
from pathlib import Path

import pymupdf

PROJECT_DIR = Path(__file__).resolve().parent
PDF_PATH = PROJECT_DIR / "test_images" / "orig_doc.pdf"
TEMPLATES_DIR = PROJECT_DIR / "templates"
DPI = 200
SCALE = DPI / 72


def to_pixels(rect) -> list[int]:
    return [round(v * SCALE) for v in (rect[0], rect[1], rect[2], rect[3])]


def extract_lines(page) -> list[dict]:
    # get_text("words") -> (x0, y0, x1, y1, word, block_no, line_no, word_no)
    lines: dict[tuple[int, int], dict] = {}
    for x0, y0, x1, y1, word, block_no, line_no, _ in page.get_text("words", sort=True):
        line = lines.setdefault((block_no, line_no), {"words": [], "rect": pymupdf.Rect(x0, y0, x1, y1)})
        line["words"].append(word)
        line["rect"] |= pymupdf.Rect(x0, y0, x1, y1)
    return [{"text": " ".join(l["words"]), "bbox": to_pixels(l["rect"])} for l in lines.values()]


def extract_drawings(page) -> list[dict]:
    return [
        {
            "bbox": to_pixels(d["rect"]),
            "type": d["type"],  # "f" fill, "s" stroke, "fs" both
            "items": "".join(item[0] for item in d["items"]),  # e.g. "re", "llll"
        }
        for d in page.get_drawings()
    ]


def main() -> None:
    with pymupdf.open(PDF_PATH) as doc:
        for page in doc:
            out_path = TEMPLATES_DIR / f"page{page.number + 1}" / "layout.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            layout = {"dpi": DPI, "lines": extract_lines(page), "drawings": extract_drawings(page)}
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(layout, f, indent=1, ensure_ascii=False)
            print(f"{out_path.relative_to(PROJECT_DIR)}: {len(layout['lines'])} lines, {len(layout['drawings'])} drawings")


if __name__ == "__main__":
    main()
