"""Read the weekly Exercise Log -- the day-by-day table participants fill
in between visits.

This is a different document from the weekly VPT session form, with no
"label: answer" pairs for field_reader to anchor on, so none of the
*_LABELS machinery applies. 35 of the study's 72 participant-weeks have an
Exercise Log and no session form, and before this module those weeks
produced no data at all.

The table is nine printed columns by seven day rows:

    Day | Date | Completed the | Number of | What type | Any difficulty | Any new pain | Participation in | Location of
        |      | exercise...   | exercises | of        | performing     | symptoms...  | Running & other? | new pain

Geometry rather than layout analysis: the printed column headers OCR at
93-96% confidence even on a phone photo, so their x positions give the
column boundaries, and the printed day numbers down the left edge give the
row positions. Most pages are photographed at a slight angle, so the table
is not axis-aligned -- the header row's own slope measures that tilt and
every cell is offset by it.
"""

import re
from dataclasses import dataclass

# Column key -> the printed header word to anchor on. Each is unique among
# the headers and OCRs reliably; the multi-word headers wrap unpredictably,
# so only the distinctive word is used.
COLUMN_ANCHORS = [
    ("day", "Day"),
    ("date", "Date"),
    ("completed", "Completed"),
    ("exercise_count", "Number"),
    ("exercise_types", "What"),
    ("difficulty", "difficulty"),
    ("new_pain", "symptoms"),
    ("running", "Participation"),
    ("pain_location", "Location"),
]

DAYS = range(1, 8)
MIN_HEADERS = 5
MIN_HEADER_CONF = 40
MIN_DAY_ANCHORS = 2
# A day row is a plausible fraction of the page height apart; anything
# outside this is a misfit from a stray digit rather than a real table.
MIN_PITCH_FRACTION = 0.025
MAX_PITCH_FRACTION = 0.09
DAY_FIT_TOLERANCE = 0.35
# A day row is about this tall relative to the gap between day numbers;
# handwriting regularly sits above its ruled line, so the crop starts high.
ROW_TOP_SLACK = 0.75
ROW_BOTTOM_SLACK = 0.45

MILES_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:mi\b|mile)", re.I)
BARE_NUMBER_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*$")
DONE_RE = re.compile(r"check|tick|yes|done|✓|✔", re.I)
NOT_DONE_RE = re.compile(r"^\s*(x|×|cross|no|blank|none|-+)\s*$", re.I)


@dataclass
class Table:
    columns: dict      # key -> (x0, x1)
    rows: dict         # day number -> y centre at x_ref
    x_ref: int         # x the row centres are measured at
    slope: float       # dy/dx of the page tilt

    def cell(self, day, column, width, height):
        """Pixel box for one cell, tilt-corrected."""
        x0, x1 = self.columns[column]
        y = self.rows[day] + self.slope * ((x0 + x1) / 2 - self.x_ref)
        pitch = self.row_pitch
        top = y - pitch * ROW_TOP_SLACK
        bottom = y + pitch * ROW_BOTTOM_SLACK
        return (max(0, round(x0)), max(0, round(top)),
                min(width, round(x1)), min(height, round(bottom)))

    @property
    def row_pitch(self):
        ys = [self.rows[d] for d in sorted(self.rows)]
        return (ys[-1] - ys[0]) / (len(ys) - 1) if len(ys) > 1 else 90.0


def _tokens(data):
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if text and float(data["conf"][i]) >= 0:
            yield i, text, float(data["conf"][i])


def header_positions(data):
    """x of each column header found on the page, by column key."""
    found = {}
    for key, anchor in COLUMN_ANCHORS:
        best = None
        for i, text, conf in _tokens(data):
            if text.strip(":") == anchor and conf >= MIN_HEADER_CONF:
                if best is None or data["top"][i] < data["top"][best]:
                    best = i
        if best is not None:
            found[key] = (data["left"][best], data["top"][best])
    return found


def is_exercise_log(data):
    """Whether this page is an Exercise Log rather than a session form."""
    words = {t.strip(":").lower() for _, t, c in _tokens(data) if c >= MIN_HEADER_CONF}
    if not {"exercise", "log"} <= words:
        return False
    return len(header_positions(data)) >= MIN_HEADERS


def _fit(xs, ys):
    """Least-squares (slope, intercept); slope 0 when x never varies."""
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0:
        return 0.0, mean_y
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
    return slope, mean_y - slope * mean_x


def page_slope(headers):
    """Tilt of the table, from the header row's own drift in y."""
    if len(headers) < 2:
        return 0.0
    xs = [x for x, _ in headers.values()]
    ys = [y for _, y in headers.values()]
    return _fit(xs, ys)[0]


def column_bounds(headers, page_width):
    """(x0, x1) per column: from its own header to the next header right."""
    ordered = sorted(headers.items(), key=lambda kv: kv[1][0])
    bounds = {}
    for pos, (key, (x, _)) in enumerate(ordered):
        right = ordered[pos + 1][1][0] if pos + 1 < len(ordered) else page_width
        bounds[key] = (x - 8, right - 8)
    return bounds


def day_rows(data, headers, slope, page_height):
    """y centre per day 1-7, fitted through whatever day numbers OCR'd.

    Tesseract reliably finds only some of the printed day digits -- faint
    print and handwriting crossing the column lose the rest -- so the ones
    it does find are fitted to a line and the remainder interpolated.
    """
    if "day" in headers:
        day_x, day_y = headers["day"]
        right = headers.get("date", (day_x + 140, 0))[0]
    else:
        # A steep enough camera angle loses the outer columns entirely
        # (VPT014's week 3 photo). The day numbers are still printed to the
        # left of every other column, so anchor on the leftmost header found.
        left_key = min(headers, key=lambda k: headers[k][0])
        day_x, day_y = headers[left_key]
        day_x, right = day_x - 420, day_x - 10
        if right <= 0:
            return {}
    seen = {}
    for i, text, conf in _tokens(data):
        if not text.isdigit() or not (1 <= int(text) <= 7):
            continue
        x, y = data["left"][i], data["top"][i]
        if not (day_x - 60 <= x < right and y > day_y + 20):
            continue
        centre = y + data["height"][i] / 2 - slope * (x - day_x)
        seen.setdefault(int(text), centre)
    if len(seen) < MIN_DAY_ANCHORS:
        return {}
    fit = _robust_day_fit(seen, page_height)
    if fit is None:
        return {}
    pitch, intercept = fit
    return {d: pitch * d + intercept for d in DAYS}


def _robust_day_fit(seen, page_height):
    """(pitch, intercept) for day number -> y, ignoring stray digits.

    A handwritten date or a dirt speck that OCRs as "1" in the day column
    will drag a least-squares line far off, which showed up as a fitted row
    pitch of 28px or 525px on pages whose rows are really ~90px apart. So
    every pair of anchors proposes a line, and the line the most anchors
    agree with wins -- with the pitch required to be a plausible fraction
    of the page to begin with.
    """
    lo, hi = page_height * MIN_PITCH_FRACTION, page_height * MAX_PITCH_FRACTION
    anchors = sorted(seen.items())
    best = None
    for i, (day_a, y_a) in enumerate(anchors):
        for day_b, y_b in anchors[i + 1:]:
            pitch = (y_b - y_a) / (day_b - day_a)
            if not lo <= pitch <= hi:
                continue
            intercept = y_a - pitch * day_a
            tol = pitch * DAY_FIT_TOLERANCE
            inliers = [d for d, y in anchors if abs(pitch * d + intercept - y) <= tol]
            if len(inliers) < MIN_DAY_ANCHORS:
                continue
            if best is None or len(inliers) > len(best[0]):
                # refit on just the agreeing anchors
                refit_pitch, refit_intercept = _fit(inliers, [seen[d] for d in inliers])
                best = (inliers, refit_pitch, refit_intercept)
    if best is None:
        return None
    return best[1], best[2]


def find_table(data, page_width, page_height):
    headers = header_positions(data)
    if len(headers) < MIN_HEADERS:
        return None
    slope = page_slope(headers)
    rows = day_rows(data, headers, slope, page_height)
    if not rows:
        return None
    x_ref = headers["day"][0] if "day" in headers else min(x for x, _ in headers.values())
    return Table(columns=column_bounds(headers, page_width), rows=rows,
                 x_ref=x_ref, slope=slope)


def parse_miles(text):
    """Miles recorded in the "Participation in Running & other?" cell.

    The column mixes running with everything else ("2.15 miles",
    "28 mile bike", "400 metre swim"), so only an explicit mile figure
    counts, and a bare number is taken as miles because that is how most
    participants write it.
    """
    if text is None:
        return None
    text = str(text)
    if re.search(r"\b(bike|biking|cycl|swim|yoga|elliptical|row)", text, re.I) \
            and not re.search(r"\bran\b|\brun", text, re.I):
        return None
    hit = MILES_RE.search(text)
    if hit:
        return float(hit.group(1))
    bare = BARE_NUMBER_RE.match(text)
    return float(bare.group(1)) if bare else None


def completed_day(text):
    """True/False/None for one "Completed the exercise as prescribed" cell."""
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None
    if NOT_DONE_RE.match(text):
        return False
    if DONE_RE.search(text):
        return True
    return None


def summarize(days):
    """Weekly totals from the per-day rows.

    Only the questions the Exercise Log actually asks are reported.
    "Days completed the exercise as prescribed" is the same question the
    session form asks, so it fills the same column; the session form's
    pain_now/usual/worst have no counterpart here (the log asks for pain
    *per exercise session*), so they are deliberately left out rather than
    filled with something that means something else.
    """
    marks = [completed_day(d.get("completed")) for d in days]
    miles = [parse_miles(d.get("running")) for d in days]
    counts = [d.get("exercise_count") for d in days]

    out = {}
    if any(m is not None for m in marks):
        out["days_exercised"] = sum(1 for m in marks if m)
    ran = [m for m in miles if m is not None]
    if ran:
        out["days_ran"] = len(ran)
        out["mileage"] = round(sum(ran), 2)
    numeric = [_as_number(c) for c in counts]
    numeric = [n for n in numeric if n is not None]
    if numeric:
        out["exercises_per_session"] = max(set(numeric), key=numeric.count)
    return out


def _as_number(text):
    if text is None:
        return None
    hit = re.search(r"\d+(?:\.\d+)?", str(text))
    return float(hit.group(0)) if hit else None


def read_table(page_image, data, read_cell, columns=("completed", "exercise_count", "running")):
    """Per-day readings for the named columns.

    read_cell(crop, column_key, day) -> str is supplied by the caller so
    this module stays free of any model import, the way field_reader does.
    """
    table = find_table(data, page_image.width, page_image.height)
    if table is None:
        return None, None
    days = []
    for day in DAYS:
        row = {"day": day}
        for column in columns:
            if column not in table.columns:
                continue
            box = table.cell(day, column, page_image.width, page_image.height)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            row[column] = read_cell(page_image.crop(box), column, day)
        days.append(row)
    return days, summarize(days)
