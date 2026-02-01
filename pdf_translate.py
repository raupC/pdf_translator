#!/usr/bin/env python3
import argparse
import io
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import fitz  # PyMuPDF
import numpy as np
import cv2
from PIL import Image
import pytesseract

import argostranslate.translate as argos_translate


# ----------------------------
# Data structures
# ----------------------------
@dataclass
class OCRWord:
    text: str
    conf: int
    x: int
    y: int
    w: int
    h: int
    block_num: int
    par_num: int
    line_num: int


@dataclass
class OCRLine:
    key: Tuple[int, int, int]  # (block, par, line)
    words: List[OCRWord]
    rect: Tuple[int, int, int, int]  # x0,y0,x1,y1 in pixels

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words).strip()


@dataclass
class OCRPara:
    key: Tuple[int, int]  # (block, par)
    lines: List[OCRLine]
    rect: Tuple[int, int, int, int]  # x0,y0,x1,y1 in pixels

    @property
    def text(self) -> str:
        return "\n".join(ln.text for ln in self.lines).strip()


# ----------------------------
# Image conversions
# ----------------------------
def pil_to_cv(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv_to_pil(arr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ----------------------------
# OCR preprocessing
# ----------------------------
def preprocess_for_ocr(pil_img: Image.Image) -> Image.Image:
    cv_img = pil_to_cv(pil_img)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    gray = cv2.fastNlMeansDenoising(gray, None, 20, 7, 21)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)
    gray = cv2.filter2D(gray, -1, kernel)

    return Image.fromarray(gray)


# ----------------------------
# OCR extraction
# ----------------------------
def ocr_words(img_for_ocr: Image.Image, lang: str, psm: int) -> List[OCRWord]:
    config = f"--oem 1 --psm {psm}"
    d = pytesseract.image_to_data(
        img_for_ocr, lang=lang, config=config, output_type=pytesseract.Output.DICT
    )

    out: List[OCRWord] = []
    n = len(d["text"])
    for i in range(n):
        t = (d["text"][i] or "").strip()
        if not t:
            continue
        try:
            conf = int(float(d["conf"][i]))
        except Exception:
            conf = -1

        out.append(
            OCRWord(
                text=t,
                conf=conf,
                x=int(d["left"][i]),
                y=int(d["top"][i]),
                w=int(d["width"][i]),
                h=int(d["height"][i]),
                block_num=int(d["block_num"][i]),
                par_num=int(d["par_num"][i]),
                line_num=int(d["line_num"][i]),
            )
        )
    return out


# ----------------------------
# Text-likeness filter (reject symbol junk)
# ----------------------------
_word_ok_charset = re.compile(r"[A-Za-zÄÖÅäöå]")

def word_looks_like_text(w: OCRWord) -> bool:
    t = w.text.strip()
    if len(t) < 2:
        return False
    if not _word_ok_charset.search(t):
        return False
    letters = sum(ch.isalpha() for ch in t)
    if letters / max(1, len(t)) < 0.5:
        return False
    if w.h < 8 or w.h > 80:
        return False
    if w.w / max(1, w.h) > 18:
        return False
    return True


# ----------------------------
# Grouping: words -> lines -> paragraphs
# ----------------------------
def group_words_to_lines(words: List[OCRWord]) -> List[OCRLine]:
    if not words:
        return []
    lines_by_key: Dict[Tuple[int, int, int], List[OCRWord]] = {}
    for w in words:
        key = (w.block_num, w.par_num, w.line_num)
        lines_by_key.setdefault(key, []).append(w)

    lines: List[OCRLine] = []
    for key, ws in lines_by_key.items():
        ws.sort(key=lambda w: (w.x, w.y))
        x0 = min(w.x for w in ws)
        y0 = min(w.y for w in ws)
        x1 = max(w.x + w.w for w in ws)
        y1 = max(w.y + w.h for w in ws)
        lines.append(OCRLine(key=key, words=ws, rect=(x0, y0, x1, y1)))

    lines.sort(key=lambda ln: (ln.rect[1], ln.rect[0]))
    return lines


def rect_union(rects: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    x0 = min(r[0] for r in rects)
    y0 = min(r[1] for r in rects)
    x1 = max(r[2] for r in rects)
    y1 = max(r[3] for r in rects)
    return (x0, y0, x1, y1)


def group_lines_to_paragraphs(lines: List[OCRLine]) -> List[OCRPara]:
    if not lines:
        return []
    paras_by_key: Dict[Tuple[int, int], List[OCRLine]] = {}
    for ln in lines:
        b, p, _ = ln.key
        paras_by_key.setdefault((b, p), []).append(ln)

    paras: List[OCRPara] = []
    for key, lns in paras_by_key.items():
        lns.sort(key=lambda ln: (ln.rect[1], ln.rect[0]))
        r = rect_union([ln.rect for ln in lns])
        paras.append(OCRPara(key=key, lines=lns, rect=r))

    paras.sort(key=lambda p: (p.rect[1], p.rect[0]))
    return paras


# ----------------------------
# Column-aware ordering (k-means)
# ----------------------------
def order_paragraphs_column_aware(paras: List[OCRPara]) -> List[OCRPara]:
    if len(paras) < 10:
        return sorted(paras, key=lambda p: (p.rect[1], p.rect[0]))

    xs = np.array([[(p.rect[0] + p.rect[2]) / 2.0] for p in paras], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.1)
    _compactness, labels, centers = cv2.kmeans(xs, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    col0 = [p for p, lab in zip(paras, labels.flatten()) if lab == 0]
    col1 = [p for p, lab in zip(paras, labels.flatten()) if lab == 1]

    if centers[0][0] <= centers[1][0]:
        left, right = col0, col1
    else:
        left, right = col1, col0

    if len(left) < 3 or len(right) < 3:
        return sorted(paras, key=lambda p: (p.rect[1], p.rect[0]))

    left.sort(key=lambda p: (p.rect[1], p.rect[0]))
    right.sort(key=lambda p: (p.rect[1], p.rect[0]))
    return left + right


# ----------------------------
# Image-region detection from PDF (embedded images only)
# ----------------------------
def get_image_rects(page: fitz.Page) -> List[fitz.Rect]:
    """
    Get rectangles of embedded image XObjects.
    NOTE: vector diagrams are NOT images; this only catches embedded bitmap images.
    """
    rects: List[fitz.Rect] = []
    for img in page.get_images(full=True):
        xref = img[0]
        try:
            rs = page.get_image_rects(xref)
            rects.extend(rs)
        except Exception:
            pass
    return rects


def pdf_rects_to_pixel_mask(page_rect: fitz.Rect, img_w: int, img_h: int, rects: List[fitz.Rect]) -> np.ndarray:
    """
    Convert PDF-coordinate image rects into a pixel mask (255=image region).
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if not rects:
        return mask

    sx = img_w / page_rect.width
    sy = img_h / page_rect.height

    for r in rects:
        # clamp to page
        rr = r & page_rect
        x0 = int(max(0, rr.x0 * sx))
        y0 = int(max(0, rr.y0 * sy))
        x1 = int(min(img_w - 1, rr.x1 * sx))
        y1 = int(min(img_h - 1, rr.y1 * sy))
        if x1 > x0 and y1 > y0:
            cv2.rectangle(mask, (x0, y0), (x1, y1), 255, thickness=-1)
    return mask


def paint_out_mask_on_pil(img: Image.Image, mask: np.ndarray, fill=(255, 255, 255)) -> Image.Image:
    """
    Paints masked regions on a PIL image.
    Used to "remove images" for OCR only.
    """
    cv_img = pil_to_cv(img)
    cv_img[mask == 255] = (fill[2], fill[1], fill[0])  # BGR
    return cv_to_pil(cv_img)


# ----------------------------
# Masking + inpainting (text only, excluding image regions)
# ----------------------------
def build_text_mask_from_words(img_w: int, img_h: int, words: List[OCRWord], pad: int) -> np.ndarray:
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for w in words:
        x0 = max(0, w.x - pad)
        y0 = max(0, w.y - pad)
        x1 = min(img_w - 1, w.x + w.w + pad)
        y1 = min(img_h - 1, w.y + w.h + pad)
        cv2.rectangle(mask, (x0, y0), (x1, y1), 255, thickness=-1)
    return mask


def inpaint(cv_img: np.ndarray, mask: np.ndarray, radius: int, dilate: int) -> np.ndarray:
    if dilate > 0:
        kernel = np.ones((dilate, dilate), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return cv2.inpaint(cv_img, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)


# ----------------------------
# Placement helpers
# ----------------------------
def px_to_pdf_rect(px_rect: Tuple[int, int, int, int], page_rect: fitz.Rect, img_w: int, img_h: int) -> fitz.Rect:
    x0, y0, x1, y1 = px_rect
    sx = page_rect.width / img_w
    sy = page_rect.height / img_h
    return fitz.Rect(x0 * sx, y0 * sy, x1 * sx, y1 * sy)


def insert_text_wrapped_fit(page: fitz.Page, rect: fitz.Rect, text: str, start_font: float, min_font: float) -> bool:
    cleaned = "\n".join([ln.strip() for ln in text.splitlines()]).strip()
    if not cleaned:
        return True

    for fs in np.linspace(start_font, min_font, num=12):
        rc = page.insert_textbox(
            rect,
            cleaned,
            fontsize=float(fs),
            fontname="helv",
            color=(0, 0, 0),
            align=fitz.TEXT_ALIGN_LEFT,
        )
        if rc >= 0:
            return True

    page.insert_textbox(rect, cleaned, fontsize=float(min_font), fontname="helv", color=(0, 0, 0))
    return False


def expand_rect_down(px_rect: Tuple[int, int, int, int], img_h: int, extra: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = px_rect
    return (x0, y0, x1, min(img_h - 1, y1 + extra))


# ----------------------------
# Main pipeline: "remove images for OCR, then keep them in final output"
# ----------------------------
def translate_pdf_remove_images_then_restore(
    input_pdf: str,
    output_pdf: str,
    src_lang: str,
    dst_lang: str,
    ocr_lang: str,
    dpi: int,
    psm: int,
    min_conf: int,
    # image removal for OCR
    image_pad_px: int,
    # inpaint
    text_mask_pad: int,
    text_mask_dilate: int,
    inpaint_radius: int,
    # placement
    rect_expand_px: int,
    column_order: bool,
):
    src_doc = fitz.open(input_pdf)
    out_doc = fitz.open()

    for i in range(len(src_doc)):
        src_page = src_doc[i]
        page_rect = src_page.rect

        # Render original page (this will be our final background)
        pix = src_page.get_pixmap(dpi=dpi, alpha=False)
        base_img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        img_w, img_h = base_img.size

        # Get embedded image rectangles and build pixel mask
        img_rects = get_image_rects(src_page)
        img_mask = pdf_rects_to_pixel_mask(page_rect, img_w, img_h, img_rects)

        # Expand image mask a bit (protect edges)
        if image_pad_px > 0 and np.any(img_mask):
            k = np.ones((image_pad_px, image_pad_px), np.uint8)
            img_mask = cv2.dilate(img_mask, k, iterations=1)

        # --- "Remove images" only for OCR ---
        ocr_view = paint_out_mask_on_pil(base_img, img_mask, fill=(255, 255, 255))
        ocr_view = preprocess_for_ocr(ocr_view)

        # OCR words (won't see the images)
        words = ocr_words(ocr_view, lang=ocr_lang, psm=psm)

        # Filter: confidence + text-likeness
        words = [w for w in words if w.conf >= min_conf and word_looks_like_text(w)]

        # Extra safety: drop any word that overlaps image mask (in case OCR still hits edges)
        if np.any(img_mask):
            kept = []
            for w in words:
                x0 = max(0, w.x)
                y0 = max(0, w.y)
                x1 = min(img_w - 1, w.x + w.w)
                y1 = min(img_h - 1, w.y + w.h)
                roi = img_mask[y0:y1, x0:x1]
                if roi.size == 0:
                    kept.append(w)
                    continue
                if (np.count_nonzero(roi) / float(roi.size)) < 0.10:
                    kept.append(w)
            words = kept

        # If nothing, just output original render
        if not words:
            out_page = out_doc.new_page(width=page_rect.width, height=page_rect.height)
            out_page.insert_image(page_rect, stream=pil_to_png_bytes(base_img))
            continue

        # Group -> paragraphs
        lines = group_words_to_lines(words)
        paras = group_lines_to_paragraphs(lines)
        if column_order:
            paras = order_paragraphs_column_aware(paras)

        # Translate per paragraph
        translated_paras: List[Tuple[OCRPara, str]] = []
        for p in paras:
            txt = p.text
            if not txt or len(txt.strip()) < 2:
                continue
            tr = argos_translate.translate(txt, src_lang, dst_lang)
            translated_paras.append((p, tr))

        # Build text mask from words BUT EXCLUDE image regions so images never get inpainted
        text_mask = build_text_mask_from_words(img_w, img_h, words, pad=text_mask_pad)
        if np.any(img_mask):
            text_mask = cv2.bitwise_and(text_mask, cv2.bitwise_not(img_mask))

        # Inpaint text from the ORIGINAL base image (which contains images)
        cv_base = pil_to_cv(base_img)
        cleaned_cv = inpaint(cv_base, text_mask, radius=inpaint_radius, dilate=text_mask_dilate)
        cleaned_img = cv_to_pil(cleaned_cv)

        # Output page: cleaned background (images preserved)
        out_page = out_doc.new_page(width=page_rect.width, height=page_rect.height)
        out_page.insert_image(page_rect, stream=pil_to_png_bytes(cleaned_img))

        # Overlay English text in paragraph boxes
        for p, tr in translated_paras:
            pxr = expand_rect_down(p.rect, img_h, extra=rect_expand_px)
            pdf_rect = px_to_pdf_rect(pxr, page_rect, img_w, img_h)

            pad_x = page_rect.width * (2 / img_w)
            pad_y = page_rect.height * (2 / img_h)
            pdf_rect = fitz.Rect(
                pdf_rect.x0 + pad_x,
                pdf_rect.y0 + pad_y,
                pdf_rect.x1 - pad_x,
                pdf_rect.y1 - pad_y,
            )

            start_font = max(7.0, min(14.0, pdf_rect.height * 0.35))
            insert_text_wrapped_fit(out_page, pdf_rect, tr, start_font=start_font, min_font=6.0)

    out_doc.save(output_pdf)
    out_doc.close()
    src_doc.close()


def main():
    ap = argparse.ArgumentParser(
        description="PDF translate (English-only, close to original): hide embedded images for OCR, keep images in final output."
    )
    ap.add_argument("input_pdf")
    ap.add_argument("output_pdf")
    ap.add_argument("--src", required=True, help="translator source code (fi)")
    ap.add_argument("--dst", required=True, help="translator target code (en)")
    ap.add_argument("--ocr-lang", default="fin", help="tesseract OCR language (fin)")

    ap.add_argument("--dpi", type=int, default=350, help="render DPI (300-400 recommended)")
    ap.add_argument("--psm", type=int, default=4, help="tesseract psm (4=columns, 6=block)")
    ap.add_argument("--min-conf", type=int, default=60, help="min OCR confidence (50-75)")

    # NEW: image-removal for OCR
    ap.add_argument("--image-pad", type=int, default=9, help="dilate image mask (pixels) to protect image borders")

    ap.add_argument("--text-mask-pad", type=int, default=2, help="padding around word boxes")
    ap.add_argument("--text-mask-dilate", type=int, default=4, help="dilate text mask (ghost removal)")
    ap.add_argument("--inpaint-radius", type=int, default=3, help="inpaint radius (2-5)")

    ap.add_argument("--rect-expand", type=int, default=18, help="expand paragraph box downward (pixels)")
    ap.add_argument("--no-column-order", action="store_true", help="disable column-aware ordering")

    args = ap.parse_args()

    translate_pdf_remove_images_then_restore(
        input_pdf=args.input_pdf,
        output_pdf=args.output_pdf,
        src_lang=args.src,
        dst_lang=args.dst,
        ocr_lang=args.ocr_lang,
        dpi=args.dpi,
        psm=args.psm,
        min_conf=args.min_conf,
        image_pad_px=args.image_pad,
        text_mask_pad=args.text_mask_pad,
        text_mask_dilate=args.text_mask_dilate,
        inpaint_radius=args.inpaint_radius,
        rect_expand_px=args.rect_expand,
        column_order=not args.no_column_order,
    )

    print(f"Done: {args.output_pdf}")


if __name__ == "__main__":
    main()
