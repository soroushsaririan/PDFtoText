"""Combine several readings of one handwritten field (different crops,
different prompt styles) into a single value plus how much to trust it.

On the 44-field real ground-truth sample, fields where two independent
readings agreed were right ~88% of the time; fields where they disagreed
were right ~18% of the time. So agreement is used as the confidence signal:
an agreed value is accepted, anything else is kept but flagged for review.
"""

import re
from collections import Counter
from dataclasses import dataclass, field

AGREED = "agreed"
REVIEW = "review"
ILLEGIBLE = "illegible"

BLANK = "blank"
ILLEGIBLE_TEXT = "[illegible]"

_BLANK_WORDS = {"blank", "(blank)", "[blank]", "empty", "none", "n/a", ""}
_ILLEGIBLE_WORDS = {"illegible", "(illegible)", "[illegible]", "[scratched out]", "unreadable"}
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

# Plausible ranges per field kind; a reading outside them is treated as a
# misread rather than voted on.
PAIN_RANGE = (0, 10)

# Share of usable readings the winning value needs to count as agreed.
AGREE_SHARE = 2 / 3


@dataclass
class Decision:
    value: str
    status: str
    readings: list = field(default_factory=list)

    @property
    def needs_review(self):
        return self.status != AGREED


def canonical(reading, field_kind=None):
    """Normalized form of one model reading, BLANK, ILLEGIBLE_TEXT, or None
    when the reading is unusable (missing, error, impossible value)."""
    if reading is None or (isinstance(reading, float) and reading != reading):
        return None
    text = re.sub(r"\s+", " ", str(reading).strip().strip("\"'")).lower()
    if text.startswith("[error") or "unavailable:" in text:
        return None
    if text in _BLANK_WORDS:
        return BLANK
    if text in _ILLEGIBLE_WORDS:
        return ILLEGIBLE_TEXT
    if field_kind in ("number", "pain"):
        found = _NUMBER_RE.fullmatch(text) or _NUMBER_RE.search(text)
        if not found:
            return None
        number = float(found.group(0))
        if number < 0:
            return None
        if field_kind == "pain" and not (PAIN_RANGE[0] <= number <= PAIN_RANGE[1]):
            return None
        return str(int(number)) if number.is_integer() else str(number)
    return text


def decide(readings, field_kind=None):
    """Readings are in priority order: on a tie, the earlier one wins."""
    canon = [canonical(r, field_kind) for r in readings]
    usable = [c for c in canon if c is not None and c != ILLEGIBLE_TEXT]
    if not usable:
        return Decision(value=ILLEGIBLE_TEXT, status=ILLEGIBLE, readings=list(readings))
    counts = Counter(usable)
    top = max(counts.values())
    winner = next(c for c in usable if counts[c] == top)
    value = winner
    if winner != BLANK and field_kind not in ("number", "pain"):
        # voted case-insensitively, but report it as the model wrote it
        value = next(re.sub(r"\s+", " ", str(r).strip().strip("\"'"))
                     for r, c in zip(readings, canon) if c == winner)
    agreed = top >= 2 and top / len(usable) >= AGREE_SHARE
    return Decision(value=value, status=AGREED if agreed else REVIEW, readings=list(readings))
