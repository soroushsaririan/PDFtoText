"""Run each prompt style over every field of the sample forms and dump the
answers side by side (CSV / JSON / Excel + the crops) for review.

    ./.venv/bin/python benchmark/run_prompt_styles.py --dry-run
    ./.venv/bin/python benchmark/run_prompt_styles.py --participants VPT001 --forms weekly
    ./.venv/bin/python benchmark/run_prompt_styles.py --styles answer discriminative --fields mileage pain_worst
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import pytesseract
from PIL import Image
from pytesseract import Output

BENCH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCH_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import prompts
from analyze_participant import discover_files
from field_extraction import (
    EXIT_LABELS,
    FINAL_LABELS,
    INTAKE_LABELS,
    WEEKLY_LABELS,
)
from ocr_pipeline import (
    CROP_PADDING,
    RENDER_ZOOM,
    VLM_MAX_NEW_TOKENS,
    VLM_MAX_NEW_TOKENS_DEFAULT,
    render_file_to_images,
)

FORM_LABEL_SETS = {
    "weekly": WEEKLY_LABELS,
    "intake": INTAKE_LABELS,
    "final": FINAL_LABELS,
    "exit": EXIT_LABELS,
}

ANSWER_CROP_WIDTH = 1100
NOTES_LINE_SPAN = 6
PAGE_MARGIN = 32


def resolve_jobs(participant_dir, forms):
    """(form, pdf_path, label_set, week) tuples, via the pipeline's own file discovery."""
    found = discover_files(participant_dir)
    jobs = []
    if "weekly" in forms:
        for wk in found["weeks"]:
            for pdf_path in wk["files"]:
                if pdf_path.suffix.lower() == ".pdf":
                    jobs.append(("weekly", pdf_path, WEEKLY_LABELS, wk["week"]))
    visits = found["crf_visits"]
    if "intake" in forms:
        for visit_num, pdf_path in visits:
            if visit_num == 1:
                jobs.append(("intake", pdf_path, INTAKE_LABELS, None))
    if "final" in forms and visits:
        last_num, last_path = visits[-1]
        if last_num != 1:
            jobs.append(("final", last_path, FINAL_LABELS, None))
    if "exit" in forms and found["exit_survey"] is not None:
        jobs.append(("exit", found["exit_survey"], EXIT_LABELS, None))
    return jobs


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


def _line_index(data):
    lines = {}
    for i in range(len(data["text"])):
        if float(data["conf"][i]) < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(i)
    return lines


def _norm(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _label_hits(data, lines, label):
    """(box, line_key) for the printed label words, dropping the dotted
    guideline tesseract merges in. line_key lets the caller check what else
    is printed further along the same physical line (see _next_label_start_x)."""
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
                xe = [data["left"][j] + data["width"][j] for j in covered]
                ye = [data["top"][j] + data["height"][j] for j in covered]
                hits.append(((min(xs), min(ys), max(xe), max(ye)), line_key))
                break
    return hits


LABEL_END_RE = re.compile(r"[:?]$")


def _next_label_start_x(data, line_idxs, after_x):
    """Leftmost x, after after_x on this tesseract line, where another
    printed field label starts -- an answer crop should stop there instead
    of bleeding into that field's own answer. Without this, a fixed-width
    crop for e.g. "Participant ID:" can capture "Date: ... Tester: ..." from
    later in the same printed row.

    Every label on these forms ends in ':' or '?' (see field_extraction.py's
    *_LABELS), so a token ending in one of those -- regardless of whether
    it's part of the field currently being cropped -- marks a label start.
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


def _answer_crop(page_image, label_box, field_kind, max_x1=None):
    x0, y0, x1, y1 = label_box
    line_h = max(y1 - y0, 12)
    w, h = page_image.width, page_image.height
    if field_kind == "notes":
        cx0, cx1 = PAGE_MARGIN, w - PAGE_MARGIN
        cy0 = min(h, y1 + CROP_PADDING)
        cy1 = min(h, y1 + line_h * NOTES_LINE_SPAN)
    else:
        cx0 = x1 + CROP_PADDING
        cx1 = min(w - PAGE_MARGIN, x1 + ANSWER_CROP_WIDTH)
        if max_x1 is not None:
            cx1 = min(cx1, max(cx0 + 8, max_x1 - CROP_PADDING))
        cy0 = max(0, y0 - CROP_PADDING)
        cy1 = min(h, y1 + CROP_PADDING)
    if cx1 - cx0 < 8 or cy1 - cy0 < 8:
        return None
    return page_image.crop((cx0, cy0, cx1, cy1))


def _load_generator(model_override):
    import os

    if model_override:
        os.environ["VLM_MODEL_NAME"] = model_override

    import torch
    from qwen_vl_utils import process_vision_info

    from ocr_pipeline import VLM_MODEL_NAME, _load_vlm

    processor, model = _load_vlm()
    device = next(model.parameters()).device

    def generate(crop, prompt, max_new_tokens):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": crop.convert("RGB")},
            {"type": "text", "text": prompt},
        ]}]
        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[chat_text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        decoded = processor.batch_decode(trimmed, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)[0]
        return decoded.strip()

    return generate, VLM_MODEL_NAME


def _ground_truth(participant, form, week=None, base_dir=None):
    # sample_data_content's fixture participants are literally named VPT001,
    # VPT002, ... -- the same IDs real study participants use. Only trust it
    # when we're actually pointed at the generated synthetic sample data;
    # otherwise this would silently hand back fabricated values (fake
    # mileage, fake pain scores) for a real participant's real form.
    if base_dir is None or base_dir.resolve() != (PROJECT_ROOT / "sample_usb").resolve():
        return {}

    try:
        import sample_data_content as sdc
    except ImportError:
        return {}

    entry = sdc.PARTICIPANTS.get(participant)
    if not entry:
        return {}
    if form == "weekly":
        weeks = entry["weeks"]
        wk = next((w for w in weeks if w.get("week") == week), None)
        if wk is None:
            merged = {}
            for w in weeks:
                for k, v in w.items():
                    merged.setdefault(k, []).append(v)
            return {k: " | ".join(str(x) for x in v) for k, v in merged.items()}
        return {k: (v if not isinstance(v, list) else ", ".join(v)) for k, v in wk.items()}
    if form in ("intake", "final"):
        prof = entry.get("profile", {})
        edge = entry["weeks"][0] if form == "intake" else entry["weeks"][-1]
        return {**prof, **{k: (v if not isinstance(v, list) else ", ".join(v)) for k, v in edge.items()}}
    return {k: v for k, v in entry.get("exit", {}).items()}


def collect_fields(pdf_path, form, label_set, styles, out_crops, limit, week, zoom=RENDER_ZOOM):
    pages = render_file_to_images(pdf_path, out_crops / "pages" / pdf_path.stem, zoom=zoom)
    rows = []
    crops_seen = 0
    for page_no, img_path in enumerate(pages, start=1):
        page_image = Image.open(img_path).convert("RGB")
        data = pytesseract.image_to_data(page_image, config="--oem 3 --psm 6", output_type=Output.DICT)
        lines = _line_index(data)
        for label, key, kind in label_set:
            field_kind = field_kind_for(key, kind, label)
            for occ, (box, line_key) in enumerate(_label_hits(data, lines, label), start=1):
                max_x1 = _next_label_start_x(data, lines.get(line_key, []), box[2])
                crop = _answer_crop(page_image, box, field_kind, max_x1=max_x1)
                if crop is None:
                    continue
                wk_tag = f"_wk{week}" if week is not None else ""
                crop_name = f"{pdf_path.stem}{wk_tag}_p{page_no:02d}_{key}_{occ}.png"
                crop_path = out_crops / crop_name
                crop.save(crop_path)
                crops_seen += 1
                try:
                    crop_ref = str(crop_path.relative_to(BENCH_DIR))
                except ValueError:
                    crop_ref = str(crop_path)
                for style in styles:
                    rows.append({
                        "pdf": pdf_path.name,
                        "form_type": form,
                        "week": week,
                        "page": page_no,
                        "field_key": key,
                        "label": label,
                        "field_kind": field_kind,
                        "occurrence": occ,
                        "style": style,
                        "crop": crop_ref,
                        "prompt": prompts.build_prompt(label_context=label, field_kind=field_kind, style=style),
                        "_crop_obj": crop,
                    })
                if limit and crops_seen >= limit:
                    return rows
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-dir", default=str(PROJECT_ROOT / "sample_usb"))
    parser.add_argument("--participants", nargs="*", default=None,
                        help="Participant IDs; default is the first folder found.")
    parser.add_argument("--forms", nargs="*", choices=list(FORM_LABEL_SETS), default=list(FORM_LABEL_SETS))
    parser.add_argument("--weeks", nargs="*", type=int, default=None, help="Restrict weekly forms to these week numbers.")
    parser.add_argument("--styles", nargs="*", choices=list(prompts.PROMPT_STYLES), default=list(prompts.PROMPT_STYLES))
    parser.add_argument("--fields", nargs="*", default=None, help="Only these field keys, e.g. mileage pain_worst.")
    parser.add_argument("--limit", type=int, default=0, help="Cap fields processed per form (0 = no cap).")
    parser.add_argument("--model", default=None, help="VLM override, e.g. Qwen/Qwen2.5-VL-7B-Instruct.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Force a token budget for every style.")
    parser.add_argument("--out", default=str(BENCH_DIR / "prompt_styles"), help="Output path stem.")
    parser.add_argument("--dry-run", action="store_true", help="Build prompts and crops, skip the model.")
    parser.add_argument("--zoom", type=float, default=RENDER_ZOOM,
                         help=f"PDF render resolution multiplier before cropping (default {RENDER_ZOOM}, matches production).")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    participants = args.participants or [next(
        (p.name for p in sorted(base_dir.iterdir()) if p.is_dir() and p.name.startswith("VPT")), None,
    )]
    participants = [p for p in participants if p]
    if not participants:
        print(f"No VPT participant folders under {base_dir}")
        return

    out_stem = Path(args.out).resolve()
    crops_dir = out_stem.parent / f"{out_stem.name}_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for participant in participants:
        pdir = base_dir / participant
        if not pdir.is_dir():
            print(f"  {participant}: folder not found under {base_dir}")
            continue
        for form, pdf_path, label_set, week in resolve_jobs(pdir, args.forms):
            if args.weeks and week is not None and week not in args.weeks:
                continue
            label_set = [t for t in label_set if not args.fields or t[1] in args.fields]
            if not label_set:
                continue
            tag = f"{form} wk{week}" if week is not None else form
            print(f"  {participant}/{tag}: {pdf_path.name}")
            rows = collect_fields(pdf_path, form, label_set, args.styles, crops_dir, args.limit, week, zoom=args.zoom)
            truth = _ground_truth(participant, form, week, base_dir=base_dir)
            truth.setdefault("participant_id", participant)
            for r in rows:
                r["participant"] = participant
                r["ground_truth"] = truth.get(r["field_key"], "")
            all_rows.extend(rows)

    if not all_rows:
        print("Nothing collected.")
        return

    field_count = len({(r["pdf"], r["week"], r["page"], r["field_key"], r["occurrence"]) for r in all_rows})
    print(f"\n{field_count} field crops x {len(args.styles)} styles = {len(all_rows)} prompts")

    if not args.dry_run:
        generate, model_name = _load_generator(args.model)
        print(f"Model: {model_name}\n")
        for i, r in enumerate(all_rows, start=1):
            if args.max_new_tokens:
                budget = args.max_new_tokens
            elif r["style"] == "answer":
                budget = VLM_MAX_NEW_TOKENS.get(r["field_kind"], VLM_MAX_NEW_TOKENS_DEFAULT)
            else:
                budget = 400
            try:
                r["result"] = generate(r["_crop_obj"], r["prompt"], budget)
            except Exception as exc:
                r["result"] = f"[ERROR: {exc}]"
            if i % 10 == 0 or i == len(all_rows):
                print(f"  {i}/{len(all_rows)}")
    else:
        for r in all_rows:
            r["result"] = ""

    for r in all_rows:
        r.pop("_crop_obj", None)

    df = pd.DataFrame(all_rows)
    cols = ["participant", "pdf", "form_type", "week", "page", "field_key", "label", "field_kind",
            "occurrence", "style", "ground_truth", "result", "crop", "prompt"]
    df = df[[c for c in cols if c in df.columns]]

    csv_path = out_stem.with_suffix(".csv")
    json_path = out_stem.with_suffix(".json")
    xlsx_path = out_stem.with_suffix(".xlsx")
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2, force_ascii=False)
    try:
        df.to_excel(xlsx_path, index=False)
    except Exception as exc:
        xlsx_path = f"(skipped: {exc})"

    print(f"\nCSV  : {csv_path}")
    print(f"JSON : {json_path}")
    print(f"Excel: {xlsx_path}")
    print(f"Crops: {crops_dir}")


if __name__ == "__main__":
    main()
