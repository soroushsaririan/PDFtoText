import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import fitz
import numpy as np
import pillow_heif
import pytesseract
from PIL import Image
from pytesseract import Output

# Phone photos of the paper forms are often HEIC; without this PIL can't
# open them at all.
pillow_heif.register_heif_opener()

import consensus
import exercise_log
import field_reader
import prompts

VLM_MODEL_NAME = os.environ.get("VLM_MODEL_NAME", "Qwen/Qwen2.5-VL-3B-Instruct")
TROCR_MODEL_NAME = os.environ.get("TROCR_MODEL_NAME", "microsoft/trocr-base-handwritten")
MLX_MODEL_NAME = os.environ.get("MLX_MODEL_NAME", "mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
HANDWRITING_ENGINES = ("mlx", "vlm", "trocr")

VLM_MAX_NEW_TOKENS = {"notes": 128}
VLM_MAX_NEW_TOKENS_DEFAULT = 32

# Bumped when Exercise Log reading changes, so cached pages redo it.
EXERCISE_LOG_REVISION = 1

RENDER_ZOOM = 3.0
TESSERACT_WORD_CONF_THRESHOLD = 70
MIN_LINE_HEIGHT = 8
CROP_PADDING = 6
SCRIBBLE_DENSITY_THRESHOLD = 0.16
SCRIBBLE_FOLLOWUP_WIDTH = 300

# Qwen2.5-VL's processor rounds each image dimension to the nearest 28px
# patch unit, and only upscales a crop enough to clear its very low minimum
# total-pixel floor (3136px) -- which a thin single-line form-field crop
# (44-64px tall) already does on its own. So a crop that short only ever gets
# ~2 vertical patches to represent the ink, regardless of how legible it is
# to a human. Ground-truth testing against real forms (benchmark/) showed
# pre-upscaling short crops before the model sees them raised accuracy on a
# 44-field real sample from 20% to 32% (net +5), most notably on short
# single-word/initials fields like "tester".
VLM_MIN_CROP_HEIGHT = 160

_vlm_processor = None
_vlm_model = None


def _upscale_for_vlm(pil_crop, min_height=VLM_MIN_CROP_HEIGHT):
    if pil_crop.height >= min_height:
        return pil_crop
    scale = min_height / pil_crop.height
    new_width = max(1, round(pil_crop.width * scale))
    return pil_crop.resize((new_width, min_height), Image.LANCZOS)


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


def _build_prompt(label_context, field_kind=None, style="answer"):
    return prompts.build_prompt(label_context=label_context, field_kind=field_kind, style=style)


def _clean_vlm_text(text):
    text = text.strip().strip("\"'")
    return re.sub(r"\s+", " ", text)


def _release_vlm_memory():
    """MPS's (and CUDA's) caching allocator holds freed GPU memory for reuse
    rather than returning it to the OS. Across many inference calls with
    varying image/prompt shapes (a real handwriting run processes hundreds of
    differently-sized crops), the pool can grow essentially without bound
    instead of reaching a steady state -- observed ballooning to 20+ GB
    resident on a 16 GB machine during a long benchmark loop. Call this after
    every generate() call, not just at the end of a run."""
    import torch
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def _vlm_read(pil_crop, label_context):
    import torch
    from qwen_vl_utils import process_vision_info

    processor, model = _load_vlm()
    device = next(model.parameters()).device
    field_kind = prompts.infer_field_kind(label_context)
    prompt = _build_prompt(label_context, field_kind=field_kind)
    max_new_tokens = VLM_MAX_NEW_TOKENS.get(field_kind, VLM_MAX_NEW_TOKENS_DEFAULT)
    pil_crop = _upscale_for_vlm(pil_crop.convert("RGB"))
    messages = [{"role": "user", "content": [{"type": "image", "image": pil_crop}, {"type": "text", "text": prompt}]}]
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[chat_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(device)
    try:
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    finally:
        del inputs
        _release_vlm_memory()
    return _clean_vlm_text(decoded), 0.0


_mlx_model = None
_mlx_processor = None


def _load_mlx():
    """Apple-Silicon-native 4-bit-quantized 7B, via the MLX framework rather
    than transformers/torch -- a full-precision 7B (~14GB) doesn't fit
    comfortably in 16GB unified memory, but this quantizes down to ~4.5GB."""
    global _mlx_model, _mlx_processor
    if _mlx_model is None:
        from mlx_vlm import load as mlx_load

        _mlx_model, _mlx_processor = mlx_load(MLX_MODEL_NAME)
    return _mlx_model, _mlx_processor


def mlx_generate(pil_crop, prompt, max_new_tokens):
    """generate()'s image= wiring only produces real output when the prompt has
    already been through apply_chat_template(..., num_images=1) -- skipping
    that step silently drops the image and the model just asks to see one."""
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor = _load_mlx()
    templated = apply_chat_template(processor, model.config, prompt, num_images=1)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as f:
        pil_crop.convert("RGB").save(f.name)
        result = generate(model, processor, templated, image=f.name,
                          max_tokens=max_new_tokens, temperature=0.0, verbose=False)
    return result.text.strip()


def _mlx_read(pil_crop, label_context):
    field_kind = prompts.infer_field_kind(label_context)
    prompt = _build_prompt(label_context, field_kind=field_kind)
    max_new_tokens = VLM_MAX_NEW_TOKENS.get(field_kind, VLM_MAX_NEW_TOKENS_DEFAULT)
    return _clean_vlm_text(mlx_generate(_upscale_for_vlm(pil_crop.convert("RGB")), prompt, max_new_tokens)), 0.0


_trocr_processor = None
_trocr_model = None


def _load_trocr():
    global _trocr_processor, _trocr_model
    if _trocr_model is None:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        _trocr_processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_NAME)
        _trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_NAME)
        _trocr_model.eval()
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        _trocr_model.to(device)
    return _trocr_processor, _trocr_model


def _trocr_read(pil_crop, label_context=None):
    import torch

    processor, model = _load_trocr()
    device = next(model.parameters()).device
    pixel_values = processor(images=pil_crop.convert("RGB"), return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_new_tokens=32, do_sample=False)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip(), 0.0


HANDWRITING_READERS = {"mlx": _mlx_read, "vlm": _vlm_read, "trocr": _trocr_read}


@dataclass
class OcrLine:
    text: str
    bbox: tuple
    engine: str
    confidence: float


@dataclass
class FieldRead:
    """One label-anchored handwritten field, read several ways and combined
    (see consensus.decide). readings are [crop variant, prompt style, text]."""
    key: str
    label: str
    occurrence: int
    value: str
    status: str
    readings: list
    bbox: tuple
    crop_path: str = ""


@dataclass
class PageResult:
    page_number: int
    image_path: str
    lines: list = field(default_factory=list)
    field_reads: list = field(default_factory=list)
    log_days: list = field(default_factory=list)
    log_summary: dict = field(default_factory=dict)

    @property
    def text(self):
        return "\n".join(l.text for l in self.lines if l.text)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}


class UnreadableFile(Exception):
    """A form file that can't be rendered at all -- unsupported format, or
    truncated. Raised so one bad file can be reported and skipped instead of
    taking the whole run down with it."""


def looks_like_pdf(path):
    """Whether the file really is a PDF, by signature rather than extension.

    Some forms reach us as a phone photo copy-renamed to ".pdf" (VPT014's
    weekly logs are JPEG and HEIC). Trusting the extension sends those to the
    PDF renderer, where a JPEG happens to open and a HEIC raises.
    """
    try:
        with open(path, "rb") as fh:
            return fh.read(5) == b"%PDF-"
    except OSError:
        return False


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
    """Render one form file to page PNGs, dispatching on what the file
    actually is rather than on its extension."""
    file_path = Path(file_path)
    as_pdf = looks_like_pdf(file_path) if file_path.suffix.lower() == ".pdf" \
        else file_path.suffix.lower() not in IMAGE_EXTS
    try:
        if as_pdf:
            return render_pdf_to_images(file_path, image_dir, zoom=zoom)
        image_dir = Path(image_dir)
        image_dir.mkdir(parents=True, exist_ok=True)
        out_path = image_dir / "page_01.png"
        Image.open(file_path).convert("RGB").save(out_path)
        return [out_path]
    except UnreadableFile:
        raise
    except Exception as exc:
        raise UnreadableFile(f"{file_path.name}: {type(exc).__name__}: {exc}") from exc


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


def _read_handwriting_region(pil_image, box, label_context, fallback_conf, engine="vlm"):
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(pil_image.width, x1), min(pil_image.height, y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    crop = pil_image.crop((x0, y0, x1, y1))
    bbox = (x0, y0, x1 - x0, y1 - y0)
    reader = HANDWRITING_READERS[engine]
    try:
        text, conf = reader(crop, label_context)
    except Exception as exc:
        return OcrLine(text=f"[{engine} unavailable: {exc}]", bbox=bbox, engine="handwriting_error", confidence=fallback_conf)
    if not text:
        text = "[illegible]"
    return OcrLine(text=text, bbox=bbox, engine="handwriting", confidence=conf)


def ocr_page(image_path, use_vlm=True, label_hint=None, engine="vlm"):
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
                pil_image, (xs0 - CROP_PADDING, cy0 - cpad, search_x1, cy1 + cpad), label_context, mean_conf, engine=engine,
            )
            if line:
                result_lines.append(line)

    result_lines.sort(key=lambda l: (l.bbox[1] + l.bbox[3], l.bbox[0]))
    return result_lines


# Each read is (crop variant, prompt style); --passes N uses the first N
# whose crop could be built.
# Ordered best-first: on a disagreement with no majority, earlier wins.
# On 208 hand-checked real fields (benchmark/score_crop_variants.py,
# participant ID excluded) the single reads scored ink 83%, raised 82%,
# label 74%. Two reads, ink then raised: 85% overall, and the 79% of fields
# where both agree were 96% right; the 21% flagged held 26 of the 32
# errors. A third read (label) accepts more fields but at 92%, so the
# default stays at two. discriminative writes up to 400 tokens per read
# where answer writes a few, and didn't beat answer on the 44-field sample.
READ_PLAN = (
    ("ink", "answer"),
    ("raised", "answer"),
    ("label", "answer"),
    ("raised", "discriminative"),
)
STRUCTURED_MAX_NEW_TOKENS = 400

# Structured prompt styles answer in numbered fields; this is the one that
# carries the value, per prompts.py's _FIELDS.
STRUCTURED_VALUE_FIELD = {
    "number": "VALUE AS WRITTEN",
    "pain": "VALUE 0-10",
    "text": "TEXT AS WRITTEN",
    "notes": "TRANSCRIPTION",
}


def structured_value(raw_text, field_kind, style):
    if style == "answer":
        return raw_text
    name = STRUCTURED_VALUE_FIELD.get(field_kind)
    if not name:
        return raw_text
    for line in raw_text.splitlines():
        m = re.match(rf"^\s*(?:\d+\.\s*)?{re.escape(name)}\s*:\s*(.*)$", line, re.I)
        if m:
            return m.group(1).strip()
    return None


# Participant ID comes from the folder name (analyze_participant), so the
# handwritten one isn't worth model time -- it was also the single most
# misread field on the ground-truth sample (7 of 17 misses).
SKIP_HANDWRITTEN_KEYS = {"participant_id"}


def _handwritten_labels():
    from field_extraction import EXIT_LABELS, FINAL_LABELS, INTAKE_LABELS, WEEKLY_LABELS

    seen = {}
    for label_set in (WEEKLY_LABELS, INTAKE_LABELS, FINAL_LABELS, EXIT_LABELS):
        for label, key, kind in label_set:
            if kind != "yesno" and key not in SKIP_HANDWRITTEN_KEYS:
                seen.setdefault(label, (label, key, kind))
    return list(seen.values())


def read_label_fields(page_image, field_dir, passes, generate=None):
    """Read every handwritten (non yes/no) field whose printed label is on
    this page, `passes` times each per READ_PLAN, and combine the readings.
    The raised crop is saved under field_dir for whoever reviews it."""
    generate = generate or mlx_generate
    field_dir = Path(field_dir)
    field_dir.mkdir(parents=True, exist_ok=True)
    gray = np.array(page_image.convert("L"))
    passes = max(1, passes)
    reads = []
    for label, key, kind, field_kind, occ, label_box, max_x1, ceiling in field_reader.locate_fields(
            page_image, _handwritten_labels()):
        crops = field_reader.crop_variants(page_image, label_box, field_kind, max_x1=max_x1, gray=gray, ceiling=ceiling)
        if not crops:
            continue
        readings = []
        # a variant that couldn't be built hands its pass to the next in READ_PLAN
        plan = [(variant, style) for variant, style in READ_PLAN if variant in crops][:passes]
        for variant, style in plan:
            crop = crops[variant]
            budget = STRUCTURED_MAX_NEW_TOKENS if style != "answer" else \
                VLM_MAX_NEW_TOKENS.get(field_kind, VLM_MAX_NEW_TOKENS_DEFAULT)
            prompt = prompts.build_prompt(label_context=label, field_kind=field_kind, style=style)
            try:
                raw = generate(_upscale_for_vlm(crop), prompt, budget)
                text = structured_value(raw, field_kind, style)
            except Exception as exc:
                text = f"[error: {exc}]"
            readings.append([variant, style, None if text is None else _clean_vlm_text(text)])
        decision = consensus.decide([r[2] for r in readings], field_kind=field_kind)
        shown = crops.get("raised") or crops["label"]
        crop_path = field_dir / f"{key}_{occ}.png"
        shown.save(crop_path)
        reads.append(FieldRead(
            key=key, label=label, occurrence=occ, value=decision.value, status=decision.status,
            readings=readings, bbox=tuple(int(v) for v in label_box), crop_path=str(crop_path),
        ))
    return reads


# What to ask for each Exercise Log column. The column headers are long and
# wrap, so the model gets the question rather than the printed header.
LOG_CELL_PROMPTS = {
    "completed": ("Did they mark this box as done? Answer with exactly one word: "
                  "CHECK if it holds a tick or check mark, X if it holds a cross, "
                  "BLANK if it is empty.", "answer", 8),
    "exercise_count": ("How many exercises were completed? Answer with the number only, "
                       "or BLANK if the cell is empty.", "number", 12),
    "running": ("Copy out exactly what is written in this cell about running or other "
                "exercise, including any mileage. Answer BLANK if it is empty.", "notes", 48),
}


def log_page_words(page_image):
    """Tesseract words for an Exercise Log page.

    Deliberately NOT field_reader.page_words: that passes --psm 6 ("one
    uniform block of text"), which suits the session forms but misreads a
    nine-column table -- it found the table on 26 of 54 log pages against
    41 for the default page-segmentation mode.
    """
    return pytesseract.image_to_data(page_image, output_type=Output.DICT)


def read_exercise_log(page_image, field_dir, generate=None):
    """Per-day readings and weekly totals for an Exercise Log page, or
    (None, None) when the page isn't one or its table can't be located."""
    data = log_page_words(page_image)
    if not exercise_log.is_exercise_log(data):
        return None, None
    generate = generate or mlx_generate
    field_dir = Path(field_dir)
    field_dir.mkdir(parents=True, exist_ok=True)

    def read_cell(crop, column, day):
        prompt, field_kind, budget = LOG_CELL_PROMPTS[column]
        crop.save(field_dir / f"day{day}_{column}.png")
        try:
            raw = generate(_upscale_for_vlm(crop), prompt, budget)
        except Exception as exc:
            return f"[error: {exc}]"
        text = _clean_vlm_text(raw)
        return None if text is None or text.strip().upper() == "BLANK" else text

    return exercise_log.read_table(page_image, data, read_cell, columns=tuple(LOG_CELL_PROMPTS))


def _engine_model_name(engine):
    return {"mlx": MLX_MODEL_NAME, "vlm": VLM_MODEL_NAME}.get(engine, TROCR_MODEL_NAME)


def _cache_key(file_path, engine, passes=1):
    stat = Path(file_path).stat()
    raw = f"{file_path}:{stat.st_size}:{stat.st_mtime}:{engine}:{_engine_model_name(engine)}"
    if engine == "vlm":
        raw += f":prompt-{prompts.PROMPT_REVISION}"
    if engine == "mlx":
        plan = READ_PLAN[:max(1, passes)]
        raw += f":prompt-{prompts.PROMPT_REVISION}:plan-{plan}:log-{EXERCISE_LOG_REVISION}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def run_ocr_on_file(file_path, image_dir, ocr_cache_dir, use_vlm=True, force=False, engine="mlx", passes=2):
    if engine not in HANDWRITING_ENGINES:
        raise ValueError(f"Unknown handwriting engine: {engine!r} (expected one of {HANDWRITING_ENGINES})")

    file_path = Path(file_path)
    ocr_cache_dir = Path(ocr_cache_dir)
    ocr_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = ocr_cache_dir / f"{file_path.stem}.{_cache_key(file_path, engine, passes)}.json"

    if cache_path.exists() and not force:
        cached = json.loads(cache_path.read_text())
        pages = []
        for p in cached["pages"]:
            lines = [OcrLine(text=l["text"], bbox=tuple(l["bbox"]), engine=l["engine"], confidence=l["confidence"]) for l in p["lines"]]
            reads = [FieldRead(**{**r, "bbox": tuple(r["bbox"])}) for r in p.get("field_reads", [])]
            pages.append(PageResult(page_number=p["page_number"], image_path=p["image_path"], lines=lines,
                                    field_reads=reads, log_days=p.get("log_days") or [],
                                    log_summary=p.get("log_summary") or {}))
        return pages

    # mlx reads handwriting per printed label (field_reader) rather than per
    # low-confidence tesseract run, so the line pass only needs printed text,
    # circled Yes/No answers and checkboxes.
    label_anchored = engine == "mlx" and use_vlm
    image_paths = render_file_to_images(file_path, Path(image_dir) / file_path.stem)
    pages = []
    for i, img_path in enumerate(image_paths, start=1):
        lines = ocr_page(img_path, use_vlm=use_vlm and not label_anchored, engine=engine)
        reads, log_days, log_summary = [], None, None
        if label_anchored:
            field_dir = Path(img_path).parent / "fields" / f"page_{i:02d}"
            page_image = Image.open(img_path).convert("RGB")
            reads = read_label_fields(page_image, field_dir, passes)
            if not reads:
                # No printed labels on this page -- it may be an Exercise Log,
                # which is a table rather than label/answer pairs.
                log_days, log_summary = read_exercise_log(page_image, field_dir / "log")
        pages.append(PageResult(page_number=i, image_path=str(img_path), lines=lines, field_reads=reads,
                                log_days=log_days or [], log_summary=log_summary or {}))

    serializable = {
        "source": str(file_path),
        "engine": engine,
        "model": _engine_model_name(engine),
        "pages": [
            {
                "page_number": p.page_number,
                "image_path": p.image_path,
                "lines": [l.__dict__ for l in p.lines],
                "field_reads": [r.__dict__ for r in p.field_reads],
                "log_days": p.log_days,
                "log_summary": p.log_summary,
            }
            for p in pages
        ],
    }
    cache_path.write_text(json.dumps(serializable, indent=2))
    return pages


def full_text(pages):
    return "\n".join(p.text for p in pages)
