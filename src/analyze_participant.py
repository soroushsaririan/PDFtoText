import hashlib
import re
from collections import defaultdict
from pathlib import Path

import fitz
from docx import Document

import exercise_log as el
import field_extraction as fx
from ocr_pipeline import UnreadableFile, run_ocr_on_file, full_text

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


def _content_digest(path, chunk=1 << 20):
    """SHA-256 of a file's bytes, or None if it can't be read -- an
    unreadable file is never treated as a duplicate of anything."""
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(chunk), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def _log_days(pages):
    """Every Exercise Log row read across a week's files, day order kept.

    The weekly totals are derived from these, so a figure a reviewer
    doubts has to be traceable back to the seven cells it came from.
    """
    rows = []
    for page in pages:
        for row in getattr(page, "log_days", None) or []:
            rows.append({"source_page": page.image_path, **row})
    return rows


def discover_files(participant_dir):
    participant_dir = Path(participant_dir)

    crf_visits = []
    exit_candidates = []
    week_files = defaultdict(list)
    week_notes = defaultdict(list)
    week_missed_markers = defaultdict(list)
    unmatched = []
    # Why each file was set aside. A duplicate scan and a file nobody can
    # read both end up here, and a reviewer should not have to open the
    # folder to tell which is which.
    set_aside_because = {}

    def set_aside(path, why):
        unmatched.append(path)
        set_aside_because[path] = why

    for path in _walk_relevant_files(participant_dir):
        rel = str(path.relative_to(participant_dir))
        ext = path.suffix.lower()

        if EMAIL_RE.search(path.name):
            set_aside(path, "printed email, not a study form")
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

        set_aside(path, "no CRF visit number, exit-survey name or week number in the path")

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
            set_aside(path, f"second copy of CRF visit {visit_num}, filed in two places")
            continue
        seen_copies.add(copy_key)
        deduped_crf_visits.append((visit_num, path))
    crf_visits = deduped_crf_visits

    exit_candidates.sort()
    exit_survey = exit_candidates[0] if exit_candidates else None
    for path in exit_candidates[1:]:
        set_aside(path, f"a second exit survey; \"{exit_survey.name}\" was read instead")

    for path in _find_empty_marker_dirs(participant_dir):
        set_aside(path, "folder says there is no log and holds no file")

    # The same weekly scan is sometimes filed twice under different names --
    # VPT014's logs are a phone photo plus a byte-identical copy renamed
    # ".pdf". OCRing both would feed the same answers through field
    # extraction twice, so keep the first copy and surface the rest.
    week_nums = sorted(set(week_files) | set(week_notes) | set(week_missed_markers))
    weeks = []
    for wn in week_nums:
        kept, seen_digests = [], set()
        for path in sorted(week_files.get(wn, [])):
            digest = _content_digest(path)
            if digest is not None and digest in seen_digests:
                twin = next(k for k in kept if _content_digest(k) == digest)
                set_aside(path, f"byte-identical copy of \"{twin.name}\"")
                continue
            seen_digests.add(digest)
            kept.append(path)
        weeks.append({
            "week": wn,
            "files": kept,
            "notes": sorted(week_notes.get(wn, [])),
            "missed_markers": sorted(week_missed_markers.get(wn, [])),
        })

    return {"crf_visits": crf_visits, "exit_survey": exit_survey, "weeks": weeks,
            "unmatched": unmatched, "unmatched_reasons": set_aside_because}


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
        # Exercise Logs that are plainly logs but whose table couldn't be
        # located -- a page someone has to type in rather than a gap in the
        # study's records.
        "unread_logs": [],
        "unmatched_files": [
            f"{p.relative_to(participant_dir)} -- {files['unmatched_reasons'].get(p, 'not recognized')}"
            for p in files["unmatched"]
        ],
    }

    def ocr_one(path):
        """Pages of one form file, or no pages at all if it can't be read --
        a single unreadable file is reported under "Files Not Processed"
        rather than ending the whole run."""
        try:
            return run_ocr_on_file(
                path, image_dir=image_dir, ocr_cache_dir=ocr_cache_dir, use_vlm=use_vlm, force=force, engine=engine,
                passes=passes,
            )
        except UnreadableFile as exc:
            rel = path.relative_to(participant_dir)
            result["unmatched_files"].append(f"{rel} -- unreadable: {exc}")
            return []

    def queue_review(pages, label_set, source, form, week=None):
        for key, entry in fx.field_review(pages, label_set).items():
            result["review"].append({"source": source, "form": form, "week": week, "field": key, **entry})

    def queue_log_review(pages, source, week):
        """Exercise Log figures that are out of range for this population.

        A log cell is read once, so there is no second reading to disagree
        with it -- implausibility is the only flag these can carry.
        """
        for page in pages:
            days = getattr(page, "log_days", None) or []
            for field, detail, cell in el.implausible(days, getattr(page, "log_summary", None)):
                result["review"].append({
                    "source": source, "form": "weekly", "week": week, "field": field,
                    "label": "Exercise Log", "value": detail, "status": "implausible",
                    "readings": cell or "", "crop": page.image_path,
                })

    def ocr_many(paths):
        pages = []
        for path in paths:
            pages.extend(ocr_one(path))
        return pages

    for visit_num, path in files["crf_visits"]:
        pages = ocr_one(path)
        if not pages:
            continue
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
        if pages:
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
        file_names = ", ".join(p.name for p in wk["files"])
        if not pages:
            result["weeks"].append({"week": wk["week"], "status": "unreadable", "note": None,
                                    "fields": None, "file": file_names})
            continue
        text = full_text(pages)
        result["keyword_hits"] += [dict(h, source=file_names) for h in fx.keyword_search(text, keywords)]
        fields, raw = fx.extract_weekly_record(pages)
        queue_review(pages, fx.WEEKLY_LABELS, file_names, "weekly", week=wk["week"])
        log_days = _log_days(pages)
        queue_log_review(pages, file_names, wk["week"])
        result["unread_logs"] += [{"week": wk["week"], "source": file_names, "page": page.image_path}
                                  for page in pages if page.log_status == "table_not_located"]
        result["weeks"].append({"week": wk["week"], "status": "completed", "note": None, "fields": fields,
                                "raw": raw, "file": file_names, "log_days": log_days})

    return result
