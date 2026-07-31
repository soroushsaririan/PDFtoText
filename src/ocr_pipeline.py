import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz
import numpy as np
import pytesseract
from PIL import Image
from pytesseract import Output

VLM_MODEL_NAME = os.environ.get("VLM_MODEL_NAME", "Qwen/Qwen2.5-VL-3B-Instruct")

RENDER_ZOOM = 3.0
TESSERACT_WORD_CONF_THRESHOLD = 70
MIN_LINE_HEIGHT = 8
CROP_PADDING = 6
SCRIBBLE_DENSITY_THRESHOLD = 0.16
SCRIBBLE_FOLLOWUP_WIDTH = 300

_vlm_processor = None
_vlm_model = None


def _load_vlm():
    global _vlm_processor, _vlm_model
    if _vlm_model is None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        _vlm_processor = AutoProcessor.from_pretrained(VLM_MODEL_NAME)
        _vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(VLM_MODEL_NAME, torch_dtype="auto")
        _vlm_model.eval()
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        _vlm_model.to(device)
    return _vlm_processor, _vlm_model


def _build_prompt(label_context):
    if label_context:
        intro = (
            "This image is a cropped section of a medical case-report form, showing the "
            f'handwritten answer to the printed question: "{label_context}"\n\n'
        )
    else:
        intro = "This image is a cropped section of a handwritten form.\n\n"
    return (
        intro
        + "Read the handwritten answer in the image. If a wrong answer was written and then "
        "crossed out or scribbled over, ignore it and give only the corrected answer written "
        "afterward. Ignore any dotted or dashed guideline running through the field, and ignore "
        "the printed question text if any of it appears in the image.\n\n"
        "Reply with ONLY the answer itself -- no explanation, no repeated question. "
        "If nothing legible is written, reply with exactly: [illegible]"
    )


def _clean_vlm_text(text):
    text = text.strip().strip("\"'")
    return re.sub(r"\s+", " ", text)


def _vlm_read(pil_crop, label_context):
    import torch
    from qwen_vl_utils import process_vision_info

    processor, model = _load_vlm()
    device = next(model.parameters()).device
    prompt = _build_prompt(label_context)
    messages = [{"role": "user", "content": [{"type": "image", "image": pil_crop.convert("RGB")}, {"type": "text", "text": prompt}]}]
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[chat_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return _clean_vlm_text(decoded), 0.0


@dataclass
class OcrLine:
    text: str
    bbox: tuple
    engine: str
    confidence: float


@dataclass
class PageResult:
    page_number: int
    image_path: str
    lines: list = field(default_factory=list)

    @property
    def text(self):
        return "\n".join(l.text for l in self.lines if l.text)


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def render_pdf_to_images(pdf_path, image_dir, zoom=RENDER_ZOOM):
    pdf_path = Path(pdf_path)
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    doc = fitz.open(pdf_path)
    for i in range(len(doc)):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        out_path = image_dir / f"page_{i + 1:02d}.png"
        pix.save(out_path)
        paths.append(out_path)
    doc.close()
    return paths


def render_file_to_images(file_path, image_dir, zoom=RENDER_ZOOM):
    file_path = Path(file_path)
    if file_path.suffix.lower() in IMAGE_EXTS:
        image_dir = Path(image_dir)
        image_dir.mkdir(parents=True, exist_ok=True)
        out_path = image_dir / "page_01.png"
        Image.open(file_path).convert("RGB").save(out_path)
        return [out_path]
    return render_pdf_to_images(file_path, image_dir, zoom=zoom)


def _find_yes_no(words):
    found = {"Yes": None, "No": None}
    for i, w in enumerate(words):
        norm = re.sub(r"[^a-z]", "", w.lower())
        if norm == "yes":
            found["Yes"] = i
        elif norm == "no":
            found["No"] = i
    return found


def _ink_score(rgb, box, margin_x, margin_y):
    x0, y0, x1, y1 = box
    x0 = max(0, x0 - margin_x); y0 = max(0, y0 - margin_y)
    x1 = min(rgb.shape[1], x1 + margin_x); y1 = min(rgb.shape[0], y1 + margin_y)
    if x1 <= x0 or y1 <= y0:
        return 0
    roi = rgb[y0:y1, x0:x1].astype(int)
    r, g, b = roi[..., 0], roi[..., 1], roi[..., 2]
    colored = (b - r > 25) & (b > 70) & (r < 160)
    return int(colored.sum())


def _detect_circled_choice(rgb, found, boxes, line_height):
    present = {label: idx for label, idx in found.items() if idx is not None}
    if not present:
        return None

    if len(present) == 1:
        only_label = next(iter(present))
        return "No" if only_label == "Yes" else "Yes"

    margin_x = max(10, int(line_height * 0.9))
    margin_y = max(8, int(line_height * 0.7))
    scores = {label: _ink_score(rgb, boxes[idx], margin_x, margin_y) for label, idx in present.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] >= 12 else None


def _looks_like_checkbox(w, h):
    if w >= 45 or h >= 45:
        return False
    aspect = w / h if h else 0
    return 0.5 <= aspect <= 1.8


def _read_handwriting_region(pil_image, box, label_context, fallback_conf):
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(pil_image.width, x1), min(pil_image.height, y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    crop = pil_image.crop((x0, y0, x1, y1))
    bbox = (x0, y0, x1 - x0, y1 - y0)
    try:
        text, logprob = _vlm_read(crop, label_context)
    except Exception as exc:
        return OcrLine(text=f"[VLM unavailable: {exc}]", bbox=bbox, engine="handwriting_error", confidence=fallback_conf)
    if not text:
        text = "[illegible]"
    return OcrLine(text=text, bbox=bbox, engine="handwriting", confidence=logprob)


def ocr_page(image_path, use_vlm=True, label_hint=None):
    pil_image = Image.open(image_path).convert("RGB")
    rgb = np.array(pil_image)
    gray = np.array(pil_image.convert("L"))

    data = pytesseract.image_to_data(pil_image, config="--oem 3 --psm 6", output_type=Output.DICT)
    n = len(data["text"])

    line_keys = {}
    for i in range(n):
        conf = float(data["conf"][i])
        if conf < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        line_keys.setdefault(key, []).append(i)

    result_lines = []
    for key, idxs in line_keys.items():
        idxs.sort(key=lambda i: data["left"][i])
        words = [data["text"][i].strip() for i in idxs]
        boxes = [(data["left"][i], data["top"][i], data["left"][i] + data["width"][i], data["top"][i] + data["height"][i]) for i in idxs]
        confs = [float(data["conf"][i]) for i in idxs]
        line_top = min(b[1] for b in boxes)
        line_bottom = max(b[3] for b in boxes)
        line_height = max(line_bottom - line_top, MIN_LINE_HEIGHT)

        found = _find_yes_no(words)
        has_question_mark = any("?" in w for w in words)
        if has_question_mark and (found["Yes"] is not None or found["No"] is not None):
            choice = _detect_circled_choice(rgb, found, boxes, line_height)
            full_text_line = " ".join(w for w in words if w)
            if choice:
                full_text_line += f"  [ANSWER: {choice}]"
            bbox = (boxes[0][0], line_top, boxes[-1][2] - boxes[0][0], line_height)
            result_lines.append(OcrLine(text=full_text_line, bbox=bbox, engine="printed",
                                         confidence=sum(confs) / len(confs)))
            continue

        run, run_type = [], None
        runs = []
        for w, box, c in zip(words, boxes, confs):
            if not w:
                continue
            kind = "printed" if c >= TESSERACT_WORD_CONF_THRESHOLD else "candidate"
            if run and kind != run_type:
                runs.append((run_type, run))
                run = []
            run_type = kind
            run.append((w, box, c))
        if run:
            runs.append((run_type, run))

        printed_y_range = None
        printed_text_context = None
        for kind, items in runs:
            xs0 = min(b[0] for _, b, _ in items)
            ys0 = min(b[1] for _, b, _ in items)
            xs1 = max(b[2] for _, b, _ in items)
            ys1 = max(b[3] for _, b, _ in items)
            mean_conf = sum(c for _, _, c in items) / len(items)

            if kind == "printed":
                text = " ".join(w for w, _, _ in items)
                bbox = (xs0, ys0, xs1 - xs0, ys1 - ys0)
                result_lines.append(OcrLine(text=text, bbox=bbox, engine="printed", confidence=mean_conf))
                printed_y_range = (ys0, ys1)
                printed_text_context = text
                continue

            if not use_vlm:
                continue

            raw_w, raw_h = xs1 - xs0, ys1 - ys0
            if len(items) == 1 and _looks_like_checkbox(raw_w, raw_h):
                continue

            cy0, cy1 = printed_y_range if printed_y_range else (ys0, ys1)
            cpad = max(6, int((cy1 - cy0) * 0.3))

            raw_crop = gray[max(0, ys0):ys1, max(0, xs0):xs1]
            looks_scribbly = raw_crop.size > 0 and (raw_crop < 150).mean() >= SCRIBBLE_DENSITY_THRESHOLD
            search_x1 = min(xs1 + SCRIBBLE_FOLLOWUP_WIDTH, gray.shape[1]) if looks_scribbly else xs1 + CROP_PADDING

            label_context = printed_text_context or label_hint
            line = _read_handwriting_region(
                pil_image, (xs0 - CROP_PADDING, cy0 - cpad, search_x1, cy1 + cpad), label_context, mean_conf,
            )
            if line:
                result_lines.append(line)

    result_lines.sort(key=lambda l: (l.bbox[1] + l.bbox[3], l.bbox[0]))
    return result_lines


def _cache_key(file_path):
    stat = Path(file_path).stat()
    raw = f"{file_path}:{stat.st_size}:{stat.st_mtime}:{VLM_MODEL_NAME}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def run_ocr_on_file(file_path, image_dir, ocr_cache_dir, use_vlm=True, force=False):
    file_path = Path(file_path)
    ocr_cache_dir = Path(ocr_cache_dir)
    ocr_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = ocr_cache_dir / f"{file_path.stem}.{_cache_key(file_path)}.json"

    if cache_path.exists() and not force:
        cached = json.loads(cache_path.read_text())
        pages = []
        for p in cached["pages"]:
            lines = [OcrLine(text=l["text"], bbox=tuple(l["bbox"]), engine=l["engine"], confidence=l["confidence"]) for l in p["lines"]]
            pages.append(PageResult(page_number=p["page_number"], image_path=p["image_path"], lines=lines))
        return pages

    image_paths = render_file_to_images(file_path, Path(image_dir) / file_path.stem)
    pages = []
    for i, img_path in enumerate(image_paths, start=1):
        lines = ocr_page(img_path, use_vlm=use_vlm)
        pages.append(PageResult(page_number=i, image_path=str(img_path), lines=lines))

    serializable = {
        "source": str(file_path),
        "model": VLM_MODEL_NAME,
        "pages": [
            {
                "page_number": p.page_number,
                "image_path": p.image_path,
                "lines": [l.__dict__ for l in p.lines],
            }
            for p in pages
        ],
    }
    cache_path.write_text(json.dumps(serializable, indent=2))
    return pages


def full_text(pages):
    return "\n".join(p.text for p in pages)
