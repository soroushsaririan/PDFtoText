import re

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

WEEKLY_LABELS = [
    ("Participant ID:", "participant_id", "text"),
    ("Date:", "date", "text"),
    ("Weekly visit #:", "weekly_visit", "number"),
    ("Days completed exercise as prescribed:", "days_exercised", "number"),
    ("Number of exercises per session:", "exercises_per_session", "number"),
    ("Any exercises too difficult?", "too_difficult", "yesno"),
    ("Confident in correct running form?", "confident_form", "yesno"),
    ("Incorporated running-form cues?", "cues_used", "yesno"),
    ("Days ran this week:", "days_ran", "number"),
    ("Mileage this week:", "mileage", "number"),
    ("Total minutes running this week (min):", "running_minutes", "number"),
    ("Other exercise this week?", "other_exercise", "yesno"),
    ("Pain RIGHT NOW", "pain_now", "number"),
    ("USUAL pain during the last week:", "pain_usual", "number"),
    ("BEST pain during the last week:", "pain_best", "number"),
    ("WORST pain during the last week:", "pain_worst", "number"),
    ("Therapist / participant notes:", "notes", "text"),
]

INTAKE_LABELS = [
    ("Participant ID:", "participant_id", "text"),
    ("Tester:", "tester", "text"),
    ("Age (yrs):", "age", "number"),
    ("Injury diagnosis (if known):", "injury_diagnosis", "text"),
    ("Current injury location:", "injury_location", "text"),
]

FINAL_LABELS = [
    ("Participant ID:", "participant_id", "text"),
    ("Days completed exercise (days/week):", "days_exercised", "number"),
    ("Number of exercises per session:", "exercises_per_session", "number"),
    ("Confident in correct running form?", "confident_form", "yesno"),
    ("Mileage this week:", "mileage", "number"),
    ("Pain RIGHT NOW", "pain_now", "number"),
    ("USUAL pain during the last week:", "pain_usual", "number"),
    ("BEST pain during the last week:", "pain_best", "number"),
    ("WORST pain during the last week:", "pain_worst", "number"),
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
    return (not t) or t in ("[illegible]", "[scratched out]") or t.startswith("[VLM")


def flatten_lines(pages):
    out = []
    for page in pages:
        for line in page.lines:
            out.append((line.text, line.engine))
    return out


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
    return fields, raw


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
