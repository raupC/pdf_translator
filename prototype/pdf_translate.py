#!/usr/bin/env python3
import argparse
import textwrap
from typing import List

import pdfplumber
from reportlab.lib.pagesizes import A4, letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

import argostranslate.translate as argos_translate


def extract_text(pdf_path: str) -> List[str]:
    """
    Extract text page-by-page. Returns list of page texts.
    """
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            pages_text.append(t.strip())
    return pages_text


def translate_pages(pages_text: List[str], src: str, dst: str) -> List[str]:
    """
    Translate each page. Argos can be slower for huge pages; we chunk lines.
    """
    translated_pages = []
    for page_text in pages_text:
        if not page_text:
            translated_pages.append("")
            continue

        # Chunk by lines to avoid very large calls
        lines = page_text.splitlines()
        out_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                out_lines.append("")
                continue
            out_lines.append(argos_translate.translate(line, src, dst))
        translated_pages.append("\n".join(out_lines))
    return translated_pages


def write_translated_pdf(
    out_path: str,
    translated_pages: List[str],
    page_size_name: str,
    font_path: str | None,
    font_size: int,
    margin: int,
    wrap_width: int,
):
    page_size = A4 if page_size_name.lower() == "a4" else letter

    c = canvas.Canvas(out_path, pagesize=page_size)
    width, height = page_size

    font_name = "Helvetica"
    if font_path:
        # Register a TTF to support non-latin scripts better
        font_name = "CustomFont"
        pdfmetrics.registerFont(TTFont(font_name, font_path))

    c.setFont(font_name, font_size)

    max_y = height - margin
    min_y = margin
    line_height = int(font_size * 1.35)

    for page_text in translated_pages:
        y = max_y

        # Wrap long lines to fit better.
        # wrap_width is character-based, not perfect, but practical.
        wrapped_lines: List[str] = []
        for raw_line in page_text.splitlines():
            if not raw_line.strip():
                wrapped_lines.append("")
                continue
            wrapped_lines.extend(textwrap.wrap(raw_line, width=wrap_width))

        for line in wrapped_lines:
            if y <= min_y:
                c.showPage()
                c.setFont(font_name, font_size)
                y = max_y

            c.drawString(margin, y, line)
            y -= line_height

        c.showPage()

    c.save()


def main():
    ap = argparse.ArgumentParser(
        description="Translate a text-based PDF to another language and output a new PDF (offline via Argos Translate)."
    )
    ap.add_argument("input_pdf", help="Path to input PDF")
    ap.add_argument("output_pdf", help="Path to output translated PDF")
    ap.add_argument("--src", required=True, help="Source language code (e.g., en, fi, es)")
    ap.add_argument("--dst", required=True, help="Target language code (e.g., en, fi, es)")
    ap.add_argument("--page-size", choices=["A4", "LETTER"], default="A4", help="Output page size")
    ap.add_argument("--font", default=None, help="Optional path to a .ttf font for better unicode support")
    ap.add_argument("--font-size", type=int, default=11, help="Font size")
    ap.add_argument("--margin", type=int, default=48, help="Page margin in points")
    ap.add_argument("--wrap-width", type=int, default=95, help="Approx characters per line (tweak if lines overflow)")
    args = ap.parse_args()

    pages = extract_text(args.input_pdf)
    if all(not p for p in pages):
        raise SystemExit(
            "No extractable text found. If this is a scanned PDF, you need OCR mode (ask me and I’ll add it)."
        )

    translated = translate_pages(pages, args.src, args.dst)
    write_translated_pdf(
        args.output_pdf,
        translated,
        page_size_name=args.page_size,
        font_path=args.font,
        font_size=args.font_size,
        margin=args.margin,
        wrap_width=args.wrap_width,
    )
    print(f"Done: {args.output_pdf}")


if __name__ == "__main__":
    main()
