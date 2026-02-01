# 📄 PDF Finnish → English Translator (Layout-Preserving, Image-Safe)

Command-line tool to translate Finnish PDFs to English while keeping the layout as close as possible to the original.

This script:

- ✅ Translates Finnish → English
- ✅ Preserves page layout visually
- ✅ Protects embedded images from OCR damage
- ✅ Removes original Finnish text via inpainting
- ✅ Overlays translated English text in the same regions
- ✅ Works fully offline (OCR + Argos Translate)

Designed for slide decks, lecture notes, and technical PDFs with diagrams.

---

# ✨ Features

- High-DPI page rendering
- OCR with Tesseract (Finnish language model)
- Image-region detection — images are hidden during OCR and preserved in output
- Text box detection → paragraph grouping → translation
- Background text removal using OpenCV inpainting
- Column-aware reading order
- CLI interface with tunable parameters

---

# 🧰 Requirements

## System

- Python 3.10+
- Tesseract OCR installed

### Install Tesseract

Windows installer:
https://github.com/tesseract-ocr/tesseract

Verify installation:

```bash
tesseract --version
```

Check Finnish language support:

```bash
tesseract --list-langs
```

You must see:

```
fin
```

---

## Python Packages

Install dependencies:

```bash
pip install pymupdf pytesseract pillow opencv-python numpy argostranslate
```

Or use requirements.txt:

```bash
pip install -r requirements.txt
```

---

# 🌍 Install Argos Translate Language Package

Example: Finnish → English

```bash
python -c "import argostranslate.package as p; p.update_package_index(); pkg=[x for x in p.get_available_packages() if x.from_code=='fi' and x.to_code=='en'][0]; p.install_from_path(pkg.download())"
```

---

# 🚀 Usage

Basic:

```bash
python pdf_translate.py input.pdf output.pdf --src fi --dst en
```

Recommended quality settings:

```bash
python pdf_translate.py input.pdf output.pdf \
  --src fi --dst en \
  --dpi 350 \
  --psm 4 \
  --min-conf 60
```

---

# ⚙️ Important Options

## OCR Quality

Higher DPI improves OCR accuracy but is slower:

```bash
--dpi 350
```

Better for multi-column layouts:

```bash
--psm 4
```

Minimum OCR confidence filter:

```bash
--min-conf 60
```

---

## Image Protection

Images are automatically excluded from OCR and inpainting.

Increase protection border if diagram edges get damaged:

```bash
--image-pad 15
```

---

## Stronger Finnish Text Removal

If original Finnish text faintly remains:

```bash
--text-mask-dilate 6
```

If background becomes too blurry, reduce to 2–3.

---

## Paragraph Space Expansion

English text is often longer than Finnish:

```bash
--rect-expand 25
```

---

# 🧪 Example

```bash
python pdf_translate.py databases.pdf databases_en.pdf \
  --src fi --dst en \
  --dpi 350 \
  --psm 4 \
  --min-conf 55 \
  --image-pad 12 \
  --text-mask-dilate 5
```

---

# 📌 Limitations

- Vector diagrams are not embedded images — OCR may still detect their labels
- Complex tables may translate imperfectly
- Fonts are approximated (Helvetica)
- Very dense layouts may cause text overflow

Goal: visual similarity, not perfect semantic reconstruction.

---

# 🛠 Pipeline Overview

1. Render PDF page to high-resolution image
2. Detect embedded image regions
3. Hide images for OCR pass
4. OCR Finnish text with Tesseract
5. Filter OCR noise
6. Group words → lines → paragraphs
7. Translate paragraphs with Argos Translate
8. Build mask from text boxes (excluding images)
9. Inpaint original Finnish text
10. Overlay English text
11. Save rebuilt PDF

---

# 📂 Project Structure

```
pdf-translate-layout/
 ├─ pdf_translate.py
 ├─ README.md
 └─ requirements.txt
```

---

# 📄 requirements.txt

```
pymupdf
pytesseract
pillow
opencv-python
numpy
argostranslate
```

---

# 🤝 Contributing

Contributions welcome for:

- Better layout detection
- Table handling
- More languages
- PaddleOCR backend
- Improved font matching

---

# ⚖️ License

MIT License — free to use and modify.
