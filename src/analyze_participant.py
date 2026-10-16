import re
from collections import defaultdict
from pathlib import Path

import fitz
from docx import Document

import field_extraction as fx
from ocr_pipeline import run_ocr_on_file, full_text

CRF_RE = re.compile(r"crf\s*visit\s*(\d+)", re.I)
EXIT_RE = re.compile(r"\bexit\b", re.I)
WEEK_RE = re.compile(r"week\s*(\d+)", re.I)
SKIP_DIR_RE = re.compile(r"dari|video", re.I)

# Study staff mark a week that never happened with a placeholder PDF whose
# name says so, e.g. "VPT020 No week 1 VPT session.pdf" or
# "VPT002 No Exercise log week 4.pdf" -- these carry no form fields and
# would otherwise be OCR'd and reported as a "completed" week full of
# "Not documented" values.
NO_SESSION_RE = re.compile(r"\bno\b\s+(week\s*\d*|exercise\s+log)", re.I)

# A participant who never had any weekly log at all gets an empty folder
# named after this pattern (e.g. "No VPT016 Weekly Exercise Log") instead of
# any week folders. Being empty, it's otherwise invisible to file discovery.
NO_LOG_DIR_RE = re.compile(r"^no\s*vpt\d+", re.I)

# Printed emails sometimes get filed alongside a week's forms (e.g.
# "VPT week 2 email communication.pdf"). They carry no form fields, but
# mention form names like "VPT Weekly Visit (2).pdf" in their attachment
# lists, which the label-anchored reader would mistake for the printed
# "Weekly visit" question. Surface them for a person instead of OCRing them.
EMAIL_RE = re.compile(r"\be-?mail\b", re.I)

DOC_EXTS = {".pdf", ".jpg", ".jpeg", ".png"}
NOTE_EXTS = {".docx"}


def discover_participants(base_dir):
    base_dir = Path(base_dir)
    return sorted(
        [p for p in base_dir.iterdir() if p.is_dir() and re.fullmatch(r"VPT\d+", p.name)],
        key=lambda p: p.name,
    )


def _read_docx_text(path):
    try:
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as exc:
        return f"[could not read {path.name}: {exc}]"


def _read_pdf_text(path):
    try:
        doc = fitz.open(str(path))
        try:
            text = "\n".join(page.get_text() for page in doc).strip()
        finally:
            doc.close()
        return text or "[no extractable text]"
    except Exception as exc:
        return f"[could not read {path.name}: {exc}]"


def _find_empty_marker_dirs(participant_dir):
    """Directories like 'No VPT016 Weekly Exercise Log' with nothing (real) inside."""
    markers = []
    for path in sorted(participant_dir.rglob("*")):
        if not path.is_dir() or not NO_LOG_DIR_RE.match(path.name):
            continue
        if any(p.is_file() and p.name != ".DS_Store" for p in path.rglob("*")):
            continue
        markers.append(path)
    return markers


def _walk_relevant_files(participant_dir):
    for path in sorted(participant_dir.rglob("*")):
        if path.is_dir():
            continue
        if any(SKIP_DIR_RE.search(part) for part in path.relative_to(participant_dir).parts[:-1]):
            continue
        ext = path.suffix.lower()
        if ext in DOC_EXTS or ext in NOTE_EXTS:
            yield path


def discover_files(participant_dir):
    participant_dir = Path(participant_dir)

    crf_visits = []
    exit_candidates = []
    week_files = defaultdict(list)
    week_notes = defaultdict(list)
    week_missed_markers = defaultdict(list)
    unmatched = []

    for path in _walk_relevant_files(participant_dir):
        rel = str(path.relative_to(participant_dir))
        ext = path.suffix.lower()

        if EMAIL_RE.search(path.name):
            unmatched.append(path)
            continue

        m_crf = CRF_RE.search(path.name)
        if ext == ".pdf" and m_crf:
            crf_visits.append((int(m_crf.group(1)), path))
            continue

        if ext == ".pdf" and EXIT_RE.search(path.name):
            exit_candidates.append(path)
            continue

        m_week = WEEK_RE.search(rel)
        if m_week:
            week_num = int(m_week.group(1))
            if ext in NOTE_EXTS:
                week_notes[week_num].append(path)
            elif ext == ".pdf" and NO_SESSION_RE.search(path.name):
                week_missed_markers[week_num].append(path)
            else:
                week_files[week_num].append(path)
            continue

        unmatched.append(path)

    # Same CRF visit sometimes exists both at the participant root and
    # copied into the weekly-log folder; keep the first copy in file-walk
    # (alphabetical) order and surface the rest instead of OCRing twice and
    # letting a duplicate silently masquerade as an "Intermediate Visit".
    crf_visits.sort(key=lambda t: t[0])
    # Only dedupe true copies (identical filename filed in two places, e.g.
    # the "weekly exercise log" folder also holding a copy of the root CRF
    # PDF). A deliberately distinct document for the same visit number --
    # e.g. "VPT018 CRF visit 1 repeat.pdf" alongside "VPT018 CRF visit 1.pdf"
    # -- is a real record and must not be dropped.
    deduped_crf_visits = []
    seen_copies = set()
    for visit_num, path in crf_visits:
        copy_key = (visit_num, path.name)
        if copy_key in seen_copies:
            unmatched.append(path)
            continue
        seen_copies.add(copy_key)
        deduped_crf_visits.append((visit_num, path))
    crf_visits = deduped_crf_visits

    exit_candidates.sort()
    exit_survey = exit_candidates[0] if exit_candidates else None
    unmatched.extend(exit_candidates[1:])

    unmatched.extend(_find_empty_marker_dirs(participant_dir))

    week_nums = sorted(set(week_files) | set(week_notes) | set(week_missed_markers))
    weeks = []
    for wn in week_nums:
        weeks.append({
            "week": wn,
            "files": sorted(week_files.get(wn, [])),
            "notes": sorted(week_notes.get(wn, [])),
            "missed_markers": sorted(week_missed_markers.get(wn, [])),
        })

    return {"crf_visits": crf_visits, "exit_survey": exit_survey, "weeks": weeks, "unmatched": unmatched}


def analyze_participant(participant_dir, image_dir, ocr_cache_dir, keywords, use_vlm=True, force=False, engine="mlx",
                        passes=2):
    participant_dir = Path(participant_dir)
    pid = participant_dir.name
    files = discover_files(participant_dir)

    result = {
        "participant_id": pid,
        "intake": None,
        "intermediate_visits": [],
        "final": None,
        "exit": None,
        "weeks": [],
        "keyword_hits": [],
        "review": [],
        "unmatched_files": [str(p.relative_to(participant_dir)) for p in files["unmatched"]],
    }

    def ocr_one(path):
        return run_ocr_on_file(
            path, image_dir=image_dir, ocr_cache_dir=ocr_cache_dir, use_vlm=use_vlm, force=force, engine=engine,
            passes=passes,
        )

    def queue_review(pages, label_set, source, form, week=None):
        for key, entry in fx.field_review(pages, label_set).items():
            result["review"].append({"source": source, "form": form, "week": week, "field": key, **entry})

    def ocr_many(paths):
        pages = []
        for path in paths:
            pages.extend(ocr_one(path))
        return pages

    for visit_num, path in files["crf_visits"]:
        pages = ocr_one(path)
        text = full_text(pages)
        result["keyword_hits"] += [dict(h, source=path.name) for h in fx.keyword_search(text, keywords)]
        if visit_num == 1:
            fields, raw = fx.extract_intake_record(pages)
            queue_review(pages, fx.INTAKE_LABELS, path.name, f"CRF visit {visit_num}")
            result["intake"] = {"visit": visit_num, "file": path.name, "fields": fields}
        else:
            fields, raw = fx.extract_final_record(pages)
            queue_review(pages, fx.FINAL_LABELS, path.name, f"CRF visit {visit_num}")
            entry = {"visit": visit_num, "file": path.name, "fields": fields}
            if visit_num == files["crf_visits"][-1][0]:
                result["final"] = entry
            else:
                result["intermediate_visits"].append(entry)

    if files["exit_survey"] is not None:
        path = files["exit_survey"]
        pages = ocr_one(path)
        text = full_text(pages)
        result["keyword_hits"] += [dict(h, source=path.name) for h in fx.keyword_search(text, keywords)]
        fields, raw = fx.extract_exit_record(pages)
        queue_review(pages, fx.EXIT_LABELS, path.name, "exit survey")
        result["exit"] = {"file": path.name, "fields": fields}

    for wk in files["weeks"]:
        missed_sources = wk["notes"] + wk["missed_markers"]
        if not wk["files"] and missed_sources:
            note_text = "\n".join(
                [_read_docx_text(p) for p in wk["notes"]]
                + [_read_pdf_text(p) for p in wk["missed_markers"]]
            )
            note_names = ", ".join(p.name for p in missed_sources)
            result["weeks"].append({"week": wk["week"], "status": "missed", "note": note_text, "file": note_names, "fields": None})
            continue
        if not wk["files"]:
            result["weeks"].append({"week": wk["week"], "status": "no_file_found", "note": None, "fields": None})
            continue
        pages = ocr_many(wk["files"])
        text = full_text(pages)
        file_names = ", ".join(p.name for p in wk["files"])
        result["keyword_hits"] += [dict(h, source=file_names) for h in fx.keyword_search(text, keywords)]
        fields, raw = fx.extract_weekly_record(pages)
        queue_review(pages, fx.WEEKLY_LABELS, file_names, "weekly", week=wk["week"])
        result["weeks"].append({"week": wk["week"], "status": "completed", "note": None, "fields": fields, "raw": raw, "file": file_names})

    return result
