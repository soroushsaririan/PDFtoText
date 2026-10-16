import re

import consensus

NOT_DOCUMENTED = "Not documented"
UNCERTAIN = "Uncertain"

ANSWER_RE = re.compile(r"\[ANSWER:\s*(Yes|No)\]", re.I)
CHECK_MARK_RE = re.compile(r"[xXkK]{1,2}[\]\)]")
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

EXERCISE_OPTIONS = [
    "Hip Strength", "Foot Strength", "Core Strength", "Plyometric",
    "Neuromuscular Control", "Stretching", "Other",
]
CUE_OPTIONS = [
    "Faster cadence", "Soft quiet steps", "Slight forward trunk lean",
    "Squeezing the gluteal muscles", "Tucking in the chin",
    "Keeping arms linear", "Abdominal tension", "Landing feet under hips",
    "Taking breaks when needed",
]

# Labels are matched by stripping every non-alphanumeric character and
# testing for a substring of one printed OCR line (field_reader.label_hits),
# so each entry below has to be the form's ACTUAL printed wording, on ONE
# line, and at least 6 characters once normalized. Paraphrasing silently
# costs the whole field: the label is never located, and every value comes
# out "Not documented".
WEEKLY_LABELS = [
    ("Participant ID:", "participant_id", "text"),
    ("Date:", "date", "text"),
    ("Weekly visit:", "weekly_visit", "number"),
    ("Tester:", "tester", "text"),
    ("Days completed the exercise as prescribed on handout", "days_exercised", "number"),
    ("Number of exercises completed in each session", "exercises_per_session", "number"),
    ("Any exercises that were too difficult?", "too_difficult", "yesno"),
    ("Were you confident that you had correct form?", "confident_form", "yesno"),
    # This question wraps onto a second printed line ("Were you able to
    # incorporate any cues on running / form into your training?") and the
    # Yes/No sits on that second line, which is the one to anchor on.
    ("form into your training?", "cues_used", "yesno"),
    ("How many days did you run this week?", "days_ran", "number"),
    ("What was the mileage this week?", "mileage", "number"),
    ("Total minutes running this week (min):", "running_minutes", "number"),
    ("Did you participate in any other exercise", "other_exercise", "yesno"),
    # Weekly says "overall body pain RIGHT NOW?", visit 1 and the final visit
    # say "body pain RIGHT NOW?" -- this suffix matches all three.
    ("body pain RIGHT NOW?", "pain_now", "number"),
    ("your USUAL pain level during the last week?", "pain_usual", "number"),
    ("your BEST pain level during the last week?", "pain_best", "number"),
    ("your WORST pain level during the last week?", "pain_worst", "number"),
    ("Therapist / participant notes:", "notes", "text"),
]

INTAKE_LABELS = [
    ("Participant ID:", "participant_id", "text"),
    ("Tester:", "tester", "text"),
    ("Age (yrs):", "age", "number"),
    # The printed form reads "Injury diagnosis (in known)" -- a typo on the
    # form itself, so match on the part that is stable.
    ("Injury diagnosis", "injury_diagnosis", "text"),
    ("Current injury location:", "injury_location", "text"),
    ("body pain RIGHT NOW?", "pain_now", "number"),
    ("your USUAL pain level during the last week?", "pain_usual", "number"),
    ("your BEST pain level during the last week?", "pain_best", "number"),
    ("your WORST pain level during the last week?", "pain_worst", "number"),
]

FINAL_LABELS = [
    ("Participant ID:", "participant_id", "text"),
    ("Age (yrs):", "age", "number"),
    ("Days completed the exercise as prescribed on handout", "days_exercised", "number"),
    ("Number of exercises completed in each session", "exercises_per_session", "number"),
    ("Any exercises that were too difficult?", "too_difficult", "yesno"),
    # The final visit asks this differently from the weekly form.
    ("confident now that you have correct running form?", "confident_form", "yesno"),
    ("Total distance running per week?", "distance_per_week", "number"),
    ("Total minutes of running a week", "running_minutes", "number"),
    ("body pain RIGHT NOW?", "pain_now", "number"),
    ("your USUAL pain level during the last week?", "pain_usual", "number"),
    ("your BEST pain level during the last week?", "pain_best", "number"),
    ("your WORST pain level during the last week?", "pain_worst", "number"),
    ("Overall summary / therapist notes:", "notes", "text"),
]

EXIT_LABELS = [
    ("Participant ID:", "participant_id", "text"),
    ("Overall satisfaction with the program:", "satisfaction", "text"),
    ("Would you recommend this program?", "would_recommend", "yesno"),
    ("Additional comments:", "comments", "text"),
]


def _clean(text):
    return re.sub(r"\s+", " ", text).strip()


def _looks_illegible(text):
    t = text.strip()
    if (not t) or t in ("[illegible]", "[scratched out]"):
        return True
    # Model-loading failures come through as "[vlm unavailable: ...]" /
    # "[trocr unavailable: ...]" (engine "handwriting_error"); treat those as
    # illegible rather than letting the error string become the field value.
    low = t.lower()
    return low.startswith("[vlm") or "unavailable:" in low


def flatten_lines(pages):
    out = []
    for page in pages:
        for line in page.lines:
            out.append((line.text, line.engine))
    return out


def _label_reads(pages):
    """The first legible label-anchored read per field key across pages
    (falling back to an illegible one). Only the mlx engine produces these."""
    reads = {}
    for page in pages:
        for read in getattr(page, "field_reads", None) or []:
            current = reads.get(read.key)
            if current is None or (current.status == consensus.ILLEGIBLE and read.status != consensus.ILLEGIBLE):
                reads[read.key] = read
    return reads


def _value_from_read(read, kind):
    if read.status == consensus.ILLEGIBLE:
        return UNCERTAIN
    if read.value == consensus.BLANK:
        return NOT_DOCUMENTED
    if kind == "number":
        m = NUMBER_RE.search(read.value)
        return m.group(0) if m else UNCERTAIN
    return _clean(read.value)


def _readings_text(read):
    return "; ".join(f"{variant}/{style}: {text}" for variant, style, text in read.readings)


def field_review(pages, label_set):
    """{key: entry} for every label-anchored field in label_set whose
    readings didn't agree -- the value was kept, but a person should check
    it against the crop."""
    reads = _label_reads(pages)
    review = {}
    for label, key, kind in label_set:
        read = reads.get(key)
        if read is None or kind == "yesno" or read.status == consensus.AGREED:
            continue
        review[key] = {
            "label": label,
            "value": _value_from_read(read, kind),
            "status": read.status,
            "readings": _readings_text(read),
            "crop": read.crop_path,
        }
    return review


def extract_fields(pages, label_set):
    lines = flatten_lines(pages)
    fields = {key: NOT_DOCUMENTED for _, key, _ in label_set}
    raw_ocr = {key: None for _, key, _ in label_set}

    for i, (text, engine) in enumerate(lines):
        for label, key, kind in label_set:
            if not text.startswith(label):
                continue

            if kind == "yesno":
                m = ANSWER_RE.search(text)
                fields[key] = m.group(1) if m else UNCERTAIN
                continue

            raw_parts = []
            answer_parts = []
            j = i + 1
            while j < len(lines) and lines[j][1].startswith("handwriting"):
                text_j = lines[j][0]
                raw_parts.append(text_j)
                j += 1
                if text_j == "[scratched out]":
                    continue
                answer_parts.append(text_j)
                if kind == "number":
                    break

            if raw_parts:
                raw_ocr[key] = " ".join(raw_parts)

            if not answer_parts:
                if raw_parts:
                    fields[key] = UNCERTAIN
                continue

            if all(_looks_illegible(p) for p in answer_parts):
                fields[key] = UNCERTAIN
                continue

            value_text = " ".join(answer_parts)
            if kind == "number":
                m = NUMBER_RE.search(value_text)
                fields[key] = m.group(0) if m else UNCERTAIN
            else:
                fields[key] = _clean(value_text)

    reads = _label_reads(pages)
    for label, key, kind in label_set:
        read = reads.get(key)
        if read is None or kind == "yesno":
            continue
        fields[key] = _value_from_read(read, kind)
        raw_ocr[key] = _readings_text(read)

    return fields, raw_ocr


def extract_checked_options(pages, options):
    lines = flatten_lines(pages)
    checked = []
    for text, engine in lines:
        if engine != "printed":
            continue
        for opt in options:
            if opt.lower() not in text.lower():
                continue
            prefix = text.lower().split(opt.lower())[0]
            if CHECK_MARK_RE.search(prefix) or CHECK_MARK_RE.search(text[: text.lower().find(opt.lower())]):
                checked.append(opt)
    return checked


def extract_weekly_record(pages):
    fields, raw = extract_fields(pages, WEEKLY_LABELS)
    fields["exercise_types"] = extract_checked_options(pages, EXERCISE_OPTIONS) or [NOT_DOCUMENTED]
    fields["cues_list"] = extract_checked_options(pages, CUE_OPTIONS)
    merge_exercise_log(fields, pages)
    return fields, raw


def merge_exercise_log(fields, pages):
    """Fill weekly fields from an Exercise Log table where the session form
    didn't supply them.

    Half the study's participant-weeks have an Exercise Log and no session
    form at all, so without this they carry no data. Where both exist the
    session form wins: it is the clinician-completed record, and the log's
    totals are derived from seven separate handwritten cells.
    """
    for page in pages:
        for key, value in (getattr(page, "log_summary", None) or {}).items():
            if _looks_illegible(str(fields.get(key, ""))) or fields.get(key) in (None, NOT_DOCUMENTED):
                fields[key] = value
    return fields


def extract_intake_record(pages):
    return extract_fields(pages, INTAKE_LABELS)


def extract_final_record(pages):
    return extract_fields(pages, FINAL_LABELS)


def extract_exit_record(pages):
    return extract_fields(pages, EXIT_LABELS)


def load_keywords(path):
    keywords = []
    try:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#"):
                keywords.append(line)
    except FileNotFoundError:
        pass
    return keywords


def keyword_search(text, keywords, context_chars=60):
    hits = []
    lower = text.lower()
    for kw in keywords:
        start = 0
        kw_lower = kw.lower()
        while True:
            idx = lower.find(kw_lower, start)
            if idx == -1:
                break
            snippet = text[max(0, idx - context_chars): idx + len(kw) + context_chars]
            hits.append({"keyword": kw, "context": _clean(snippet)})
            start = idx + len(kw)
    return hits
