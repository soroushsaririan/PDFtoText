"""Label-anchored field reading: find each printed label on a rendered page,
crop where its handwritten answer should be, and read that crop with the
handwriting model.

This is the cropping the ground-truth benchmark was scored on
(benchmark/run_prompt_styles.py used to own it), moved here so production
reads the exact crops the accuracy numbers describe.
"""

import re

import numpy as np
import pytesseract
from PIL import Image
from pytesseract import Output
from scipy import ndimage

CROP_PADDING = 6
NOTES_LINE_SPAN = 6
PAGE_MARGIN = 32

# A printed label token below this tesseract confidence may have been fused
# with the handwriting right after it -- see label_hits().
LOW_CONF_LABEL_TOKEN = 50

# The last word of a printed label: a real word (or "#") ending in ':' or '?',
# e.g. "Date:", "Tester:", "difficult?", "(yrs):", "#:". Handwriting that
# tesseract mangles into something ending in ':' or '?' -- a circled "2" read
# as "(2?", a scrawled "VPT" read as "Vf?", "3:" -- has fewer letters than
# that, and must not be taken for the next field's label.
LABEL_END_RE = re.compile(r"^(?:\(?[A-Za-z]{3,}\)?|#)[:?]$")

# On these forms the answer underline sits level with the printed label's
# baseline, and people write ON TOP of the underline -- so most of the
# handwriting is above the label's line box, not inside it. The label-height
# crop then only catches the bottom of it (a "7" shows as its "/" stem, the
# top of "22" is sliced off). The "raised" variant covers this field's own
# band -- from the bottom of the printed line above (on tightly spaced rows
# that's where the previous answer starts) down to just under the
# underline -- and then grows upward to take in any handwriting stroke that
# starts inside that band, so an answer that rises into the row above is
# kept whole while the row above's own answer (which sits entirely above
# the band) stays out.
RAISE_REACH_UP = 1.1
RAISE_LIMIT_UP = 2.2
CEILING_WORD_CONF = 60
STROKE_MIN_PIXELS = 12

# Dark-pixel threshold for the ink-trimmed variant, and how long a
# horizontal dark run has to be (in label-line-heights) to count as a
# printed underline rather than a handwriting stroke.
INK_DARK = 150
UNDERLINE_MIN_RUN = 4.0
INK_TRIM_MARGIN = 14

CROP_VARIANTS = ("label", "raised", "ink")


def field_kind_for(key, kind, label):
    text = label.lower()
    if "pain" in text:
        return "pain"
    if key in ("notes", "comments"):
        return "notes"
    if kind == "yesno":
        return "yesno"
    if kind == "number":
        return "number"
    return "text"


def page_words(page_image):
    return pytesseract.image_to_data(page_image, config="--oem 3 --psm 6", output_type=Output.DICT)


def line_index(data):
    lines = {}
    for i in range(len(data["text"])):
        if float(data["conf"][i]) < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(i)
    return lines


def _norm(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def label_hits(data, lines, label):
    """(box, line_key) for the printed label words, dropping the dotted
    guideline tesseract merges in. line_key lets the caller check what else
    is printed further along the same physical line (see next_label_start_x).

    A printed label sitting right up against the handwritten answer (no gap)
    sometimes gets OCR'd as ONE token spanning both, e.g. "ID:_/P" at 38%
    confidence for "Participant ID:" immediately followed by a handwritten
    "VPT020" -- the real printed text is just "ID:", but the token's measured
    box also covers the first stroke or two of the handwriting. Trusting that
    token's right edge pushed the label box (and therefore the answer crop
    that starts just after it) past the "V" and "P", clipping them from the
    saved crop entirely. So: for any token below LOW_CONF_LABEL_TOKEN, use its
    LEFT edge rather than its right edge when computing the box's rightmost
    extent -- a real printed word is normally high-confidence, so a low score
    is itself the signal that the token's width can't be trusted."""
    want = _norm(label)
    if len(want) < 6:
        return []
    hits = []
    for line_key, idxs in lines.items():
        idxs = sorted(idxs, key=lambda i: data["left"][i])
        running = ""
        covered = []
        for i in idxs:
            tok = data["text"][i].strip()
            if not tok:
                continue
            running += _norm(tok)
            covered.append(i)
            if want in running:
                xs = [data["left"][j] for j in covered]
                ys = [data["top"][j] for j in covered]
                xe = [data["left"][j] + data["width"][j]
                      if float(data["conf"][j]) >= LOW_CONF_LABEL_TOKEN else data["left"][j]
                      for j in covered]
                ye = [data["top"][j] + data["height"][j] for j in covered]
                hits.append(((min(xs), min(ys), max(xe), max(ye)), line_key))
                break
    return hits


def next_label_start_x(data, line_idxs, after_x):
    """Leftmost x, after after_x on this tesseract line, where another
    printed field label starts -- an answer crop should stop there instead
    of bleeding into that field's own answer. Without this, a fixed-width
    crop for e.g. "Participant ID:" can capture "Date: ... Tester: ..." from
    later in the same printed row.

    Every label on these forms ends in ':' or '?' (see field_extraction.py's
    *_LABELS), so a word ending in one of those -- regardless of whether
    it's part of the field currently being cropped -- marks a label start
    (see LABEL_END_RE for what counts as a word).
    This can land one word into a multi-word label (e.g. stop at "ID:"
    rather than "Participant ID:"), which just trims a little extra label
    text into the crop rather than the next field's answer; harmless.

    Requires a minimum OCR confidence: messy handwriting Tesseract can't
    read sometimes gets misread as a token that happens to end in ':' or
    '?' purely as noise (seen: "left inner" read as "Ce:" at confidence 1),
    which would wrongly truncate a real multi-word answer. Real printed
    labels in this form set score at least ~30 confidence even when faint
    ("Date:" ~33); a low floor here is enough to reject pure noise without
    rejecting a real label."""
    MIN_LABEL_CONF = 10
    candidates = [i for i in sorted(line_idxs, key=lambda i: data["left"][i])
                  if data["text"][i].strip() and data["left"][i] > after_x]
    for i in candidates:
        tok = data["text"][i].strip()
        if float(data["conf"][i]) >= MIN_LABEL_CONF and LABEL_END_RE.search(tok):
            return data["left"][i]
    return None


def answer_box(page_size, label_box, field_kind, max_x1=None):
    """Page-pixel (x0, y0, x1, y1) where the label's answer is expected, or
    None if that region is degenerate."""
    x0, y0, x1, y1 = label_box
    line_h = max(y1 - y0, 12)
    w, h = page_size
    if field_kind == "notes":
        cx0, cx1 = PAGE_MARGIN, w - PAGE_MARGIN
        cy0 = min(h, y1 + CROP_PADDING)
        cy1 = min(h, y1 + line_h * NOTES_LINE_SPAN)
    else:
        cx0 = x1 + CROP_PADDING
        # Runs to the page margin unless another label follows on the line:
        # long answers carry on past the end of the underline, into the
        # margin ("Left hamstring, Right calf    Hamstring - 1 day").
        cx1 = w - PAGE_MARGIN
        if max_x1 is not None:
            cx1 = min(cx1, max(cx0 + 8, max_x1 - CROP_PADDING))
        cy0 = max(0, y0 - CROP_PADDING)
        cy1 = min(h, y1 + CROP_PADDING)
    if cx1 - cx0 < 8 or cy1 - cy0 < 8:
        return None
    return cx0, cy0, cx1, cy1


def answer_crop(page_image, label_box, field_kind, max_x1=None):
    box = answer_box(page_image.size, label_box, field_kind, max_x1=max_x1)
    return page_image.crop(box) if box else None


def ceiling_above(data, label_box, answer_x1):
    """Bottom edge of the lowest confidently-printed word that sits wholly
    above the label, within the label-to-answer horizontal span; 0 if none."""
    x0, y0, _, y1 = label_box
    tolerance = (y1 - y0) * 0.25
    ceiling = 0
    for i in range(len(data["text"])):
        if not data["text"][i].strip() or float(data["conf"][i]) < CEILING_WORD_CONF:
            continue
        left, top = data["left"][i], data["top"][i]
        right, bottom = left + data["width"][i], top + data["height"][i]
        if right < x0 or left > answer_x1:
            continue
        if bottom <= y0 + tolerance and top < y0 - tolerance:
            ceiling = max(ceiling, bottom)
    return ceiling


def _strokes(gray_region, line_h):
    """Dark pixels with printed underlines (long horizontal runs) removed."""
    dark = gray_region < INK_DARK
    min_run = max(int(line_h * UNDERLINE_MIN_RUN), 40)
    strokes = dark.copy()
    for row_idx, row in enumerate(dark):
        if not row.any():
            continue
        padded = np.concatenate(([False], row, [False]))
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        for start, stop in zip(edges[::2], edges[1::2]):
            if stop - start >= min_run:
                strokes[row_idx, start:stop] = False
    return strokes


def _raised_box(gray, box, line_h, field_kind, ceiling=0):
    if field_kind == "notes":
        return box
    x0, y0, x1, y1 = box
    band_top = min(max(0, round(y0 - line_h * RAISE_REACH_UP), ceiling), y0)
    limit = max(0, round(y0 - line_h * RAISE_LIMIT_UP))
    strokes = _strokes(gray[limit:y1, x0:x1], line_h)
    labels, count = ndimage.label(strokes, structure=np.ones((3, 3)))
    top = band_top
    if count:
        band_rows = slice(band_top - limit, y1 - limit)
        in_band = np.bincount(labels[band_rows].ravel(), minlength=count + 1)
        for idx, sl in enumerate(ndimage.find_objects(labels), start=1):
            if in_band[idx] >= STROKE_MIN_PIXELS:
                top = min(top, limit + sl[0].start - CROP_PADDING)
    return x0, max(limit, top), x1, y1


def _ink_box(gray, box, line_h):
    """Tighten box to the handwriting inside it: drop printed underlines
    (long horizontal dark runs), then take the bounding box of what dark
    ink is left. None when nothing but underline is there."""
    x0, y0, x1, y1 = box
    strokes = _strokes(gray[y0:y1, x0:x1], line_h)
    ys, xs = np.nonzero(strokes)
    if len(xs) < 20:
        return None
    return (max(x0, x0 + int(xs.min()) - INK_TRIM_MARGIN), max(y0, y0 + int(ys.min()) - INK_TRIM_MARGIN),
            min(x1, x0 + int(xs.max()) + INK_TRIM_MARGIN + 1), min(y1, y0 + int(ys.max()) + INK_TRIM_MARGIN + 1))


def crop_variants(page_image, label_box, field_kind, max_x1=None, gray=None, ceiling=0):
    """Several framings of the same answer region, keyed by CROP_VARIANTS
    name. "label" is the original label-height crop; "raised" reaches up
    over the underline where the handwriting actually sits; "ink" is
    "raised" trimmed to the handwriting itself. A variant that can't be built
    (e.g. "ink" on an empty field) is left out."""
    box = answer_box(page_image.size, label_box, field_kind, max_x1=max_x1)
    if box is None:
        return {}
    line_h = max(label_box[3] - label_box[1], 12)
    if gray is None:
        gray = np.array(page_image.convert("L"))
    raised = _raised_box(gray, box, line_h, field_kind, ceiling)
    variants = {"label": page_image.crop(box), "raised": page_image.crop(raised)}
    ink = _ink_box(gray, raised, line_h)
    if ink and ink[2] - ink[0] >= 8 and ink[3] - ink[1] >= 8:
        variants["ink"] = page_image.crop(ink)
    return variants


def locate_fields(page_image, label_set, data=None):
    """Yield (label, key, kind, field_kind, occurrence, label_box, max_x1,
    ceiling) for every label in label_set found on the page."""
    if data is None:
        data = page_words(page_image)
    lines = line_index(data)
    for label, key, kind in label_set:
        field_kind = field_kind_for(key, kind, label)
        for occ, (box, line_key) in enumerate(label_hits(data, lines, label), start=1):
            max_x1 = next_label_start_x(data, lines.get(line_key, []), box[2])
            answer_x1 = max_x1 if max_x1 is not None else page_image.width
            yield label, key, kind, field_kind, occ, box, max_x1, ceiling_above(data, box, answer_x1)


def open_page(path):
    return Image.open(path).convert("RGB")
