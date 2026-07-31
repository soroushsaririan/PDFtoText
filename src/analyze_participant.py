import re
from collections import defaultdict
from pathlib import Path

from docx import Document

import field_extraction as fx
from ocr_pipeline import run_ocr_on_file, full_text

CRF_RE = re.compile(r"crf\s*visit\s*(\d+)", re.I)
EXIT_RE = re.compile(r"\bexit\b", re.I)
WEEK_RE = re.compile(r"week\s*(\d+)", re.I)
SKIP_DIR_RE = re.compile(r"dari|video", re.I)

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
    unmatched = []

    for path in _walk_relevant_files(participant_dir):
        rel = str(path.relative_to(participant_dir))
        ext = path.suffix.lower()

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
            else:
                week_files[week_num].append(path)
            continue

        unmatched.append(path)

    crf_visits.sort(key=lambda t: t[0])

    exit_candidates.sort()
    exit_survey = exit_candidates[0] if exit_candidates else None
    unmatched.extend(exit_candidates[1:])

    week_nums = sorted(set(week_files) | set(week_notes))
    weeks = []
    for wn in week_nums:
        weeks.append({
            "week": wn,
            "files": sorted(week_files.get(wn, [])),
            "notes": sorted(week_notes.get(wn, [])),
        })

    return {"crf_visits": crf_visits, "exit_survey": exit_survey, "weeks": weeks, "unmatched": unmatched}


def analyze_participant(participant_dir, image_dir, ocr_cache_dir, keywords, use_vlm=True, force=False):
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
        "unmatched_files": [str(p.relative_to(participant_dir)) for p in files["unmatched"]],
    }

    def ocr_one(path):
        return run_ocr_on_file(path, image_dir=image_dir, ocr_cache_dir=ocr_cache_dir, use_vlm=use_vlm, force=force)

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
            result["intake"] = {"visit": visit_num, "file": path.name, "fields": fields}
        else:
            fields, raw = fx.extract_final_record(pages)
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
        result["exit"] = {"file": path.name, "fields": fields}

    for wk in files["weeks"]:
        if not wk["files"] and wk["notes"]:
            note_text = "\n".join(_read_docx_text(p) for p in wk["notes"])
            note_names = ", ".join(p.name for p in wk["notes"])
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
        result["weeks"].append({"week": wk["week"], "status": "completed", "note": None, "fields": fields, "raw": raw, "file": file_names})

    return result
