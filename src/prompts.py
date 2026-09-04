"""Prompts for reading one cropped case-report-form field with the VLM.

Styles per field kind: answer (bare value), descriptive, discriminative
(numbered fields), paired (describe then extract). build_prompt() is the
entry point; with no field kind it returns the original answer prompt.
"""

FIELD_KINDS = ("answer", "number", "pain", "yesno", "text", "notes", "checkbox")
PROMPT_STYLES = ("answer", "descriptive", "discriminative", "paired")

# Feeds the OCR cache key; bump when the prompt text below changes.
PROMPT_REVISION = "p1"

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

SHARED_GUARDRAILS = """
Important instructions:

- Read only what is actually handwritten in the field.
- Ignore the printed question text, form labels, box outlines, and any
  dotted or dashed guideline running through the field.
- If a value was written, crossed out or scribbled over, and a new value
  written afterward, report only the final corrected value.
- Do not round, spell-correct, translate, or reformat what is written.
  Report it as written.
- Do not guess a plausible answer from context. Report only what the ink
  shows.
- If nothing legible is written, report exactly: [illegible]
""".strip()

_SUBJECT = {
    "number": "the handwritten number in this field",
    "pain": (
        "the pain rating the client has marked for this item, on a 0 to 10 "
        "scale where 0 is no pain and 10 is the worst pain"
    ),
    "yesno": "which of the printed options (Yes / No) the client has marked",
    "text": "the short handwritten entry in this field",
    "notes": "the handwritten free-text note in this field",
    "checkbox": "which of the printed checkbox options have been marked",
}

_FOCUS = {
    "number": [
        "- the digits written, and whether any is unclear,",
        "- a decimal point or fraction,",
        "- a unit, percent sign, or stray mark next to the number,",
        "- any crossed-out digit followed by a rewrite.",
    ],
    "pain": [
        "- whether a single digit is written, or a number on a printed 0-10",
        "  scale is circled, or a mark sits between two numbers,",
        "- which number the mark lands on,",
        "- any crossed-out mark followed by a new one.",
    ],
    "yesno": [
        "- whether 'Yes' or 'No' is circled, underlined, ticked, or rewritten,",
        "- whether both are marked, or neither,",
        "- any mark that was crossed out and redone.",
    ],
    "text": [
        "- each legible character, and which characters are ambiguous,",
        "- whether the entry is one word or several,",
        "- any crossed-out text followed by a rewrite.",
    ],
    "notes": [
        "- how many lines of writing are present,",
        "- which words or phrases are legible and which are not,",
        "- any crossed-out passage followed by a rewrite.",
    ],
    "checkbox": [
        "- the printed option labels you can read,",
        "- which boxes carry a tick, cross, or fill,",
        "- which marks are faint or ambiguous,",
        "- anything handwritten on an 'Other' line.",
    ],
}

_FIELDS = {
    "number": [
        "1. FIELD LEGIBLE: Yes / Partially / No",
        "2. CORRECTION PRESENT: Yes / No / Uncertain",
        "3. VALUE AS WRITTEN: the digits exactly as written, or [illegible]",
        "4. DECIMAL POINT: Yes / No / Uncertain",
        "5. UNITS OR EXTRA MARKS: any unit or symbol written with the number, or None",
        "6. VISUAL EVIDENCE: the strokes this is based on, in one line",
    ],
    "pain": [
        "1. FIELD LEGIBLE: Yes / Partially / No",
        "2. RESPONSE FORMAT: Digit written / Number circled on scale / Mark between numbers / Other / Uncertain",
        "3. CORRECTION PRESENT: Yes / No / Uncertain",
        "4. VALUE 0-10: the whole number 0 through 10 indicated, or [illegible]",
        "5. OUT OF RANGE: Yes / No",
        "6. VISUAL EVIDENCE: the strokes this is based on, in one line",
    ],
    "yesno": [
        "1. FIELD LEGIBLE: Yes / No",
        "2. RESPONSE METHOD: Circled / Underlined / Checkbox ticked / Word written / Other / Uncertain",
        "3. CORRECTION PRESENT: Yes / No / Uncertain",
        "4. ANSWER: Yes / No / Both marked / Neither marked / Uncertain",
        "5. VISUAL EVIDENCE: the strokes this is based on, in one line",
    ],
    "text": [
        "1. FIELD LEGIBLE: Yes / Partially / No",
        "2. CORRECTION PRESENT: Yes / No / Uncertain",
        "3. TEXT AS WRITTEN: verbatim transcription, or [illegible]",
        "4. UNCERTAIN CHARACTERS: characters you are unsure of, or None",
        "5. VISUAL EVIDENCE: the strokes this is based on, in one line",
    ],
    "notes": [
        "1. WRITING PRESENT: Yes / No",
        "2. LEGIBILITY: Fully legible / Partially legible / Mostly illegible",
        "3. CORRECTION PRESENT: Yes / No / Uncertain",
        "4. TRANSCRIPTION: the full text as written, keeping line breaks, or [illegible]",
        "5. UNCERTAIN SEGMENTS: words you are unsure of, quoted, or None",
        "6. VISUAL EVIDENCE: the strokes this is based on, in one line",
    ],
    "checkbox": [
        "1. OPTIONS VISIBLE: the printed option labels you can read",
        "2. MARKED OPTIONS: options whose box carries a tick, cross, or fill, or None",
        "3. AMBIGUOUS MARKS: options where a mark may or may not be present, or None",
        "4. HANDWRITTEN ADDITIONS: text written on an 'Other' line, or None",
        "5. VISUAL EVIDENCE: the marks this is based on, in one line",
    ],
}

_ANSWER_HINT = {
    "number": "The expected answer is a number.",
    "pain": "The expected answer is a whole number from 0 to 10.",
    "yesno": "The expected answer is Yes or No; if one option is circled or ticked, report that word.",
    "text": "The expected answer is a short line of text.",
    "notes": "The answer may run to several lines; transcribe all of it.",
    "checkbox": "Report each printed option that is ticked, one per line.",
}


def _intro(label_context):
    if label_context:
        return (
            "This image is a cropped section of a handwritten medical "
            "case-report form. The printed question for this field is:\n"
            f'"{label_context}"'
        )
    return "This image is a cropped section of a handwritten medical case-report form."


def _answer_prompt(label_context, field_kind):
    if label_context:
        intro = (
            "This image is a cropped section of a medical case-report form, showing the "
            f'handwritten answer to the printed question: "{label_context}"\n\n'
        )
    else:
        intro = "This image is a cropped section of a handwritten form.\n\n"
    hint = _ANSWER_HINT.get(field_kind)
    hint = (hint + " ") if hint else ""
    return (
        intro
        + "Read the handwritten answer in the image. " + hint + "If a wrong answer was written and then "
        "crossed out or scribbled over, ignore it and give only the corrected answer written "
        "afterward. Ignore any dotted or dashed guideline running through the field, and ignore "
        "the printed question text if any of it appears in the image.\n\n"
        "Reply with ONLY the answer itself -- no explanation, no repeated question. "
        "If nothing legible is written, reply with exactly: [illegible]"
    )


def _descriptive_prompt(field_kind, label_context):
    focus = "\n".join(_FOCUS[field_kind])
    return (
        f"{_intro(label_context)}\n\n"
        f"Describe only what is visibly written in this field: {_SUBJECT[field_kind]}.\n\n"
        f"Focus on:\n{focus}\n\n"
        f"{SHARED_GUARDRAILS}\n\n"
        "Keep the description to two or three sentences. Do not give a final "
        "transcription in this step."
    )


def _discriminative_prompt(field_kind, label_context):
    fields = "\n".join(_FIELDS[field_kind])
    tail = ""
    if field_kind == "checkbox":
        tail = "\n\nThe printed options for this field are:\n" + _options_block(label_context)
    return (
        f"{_intro(label_context)}\n\n"
        f"Your task is to extract {_SUBJECT[field_kind]}.\n\n"
        f"Fill in exactly these fields:\n\n"
        f"{fields}{tail}\n\n"
        f"{SHARED_GUARDRAILS}\n\n"
        "Return the answer using exactly the numbered fields above and nothing else."
    )


def _paired_prompt(field_kind, label_context):
    focus = "\n".join(_FOCUS[field_kind])
    fields = "\n".join(_FIELDS[field_kind])
    headers = "\n".join(line.split(":", 1)[0] + ":" for line in _FIELDS[field_kind])
    return (
        f"{_intro(label_context)}\n\n"
        "Work in two steps.\n\n"
        "PART A -- OBSERVATION\n"
        f"Briefly describe only what is visibly written in this field: {_SUBJECT[field_kind]}.\n"
        f"{focus}\n\n"
        "PART B -- EXTRACTION\n"
        "Based only on Part A, fill in exactly these fields:\n"
        f"{fields}\n\n"
        f"{SHARED_GUARDRAILS}\n\n"
        "Return your answer as:\n\n"
        "PART A -- OBSERVATION\n"
        "[two or three sentences]\n\n"
        "PART B -- EXTRACTION\n"
        f"{headers}"
    )


def _options_block(label_context):
    label = (label_context or "").lower()
    if "cue" in label:
        options = CUE_OPTIONS
    elif "exercise" in label:
        options = EXERCISE_OPTIONS
    else:
        options = sorted(set(EXERCISE_OPTIONS) | set(CUE_OPTIONS))
    return "\n".join(f"- {opt}" for opt in options)


_BUILDERS = {
    "descriptive": _descriptive_prompt,
    "discriminative": _discriminative_prompt,
    "paired": _paired_prompt,
}


_NUMBER_LABEL_CUES = (
    "mileage", "minutes", "days ", "days:", "number of", "how many", "age",
    "visit #", "count",
)
_TEXT_LABEL_CUES = (
    "date", "participant id", "tester", "name", "location", "diagnosis",
    "satisfaction",
)


def infer_field_kind(label_context):
    """Field kind from the printed question text, or None when unmatched."""
    if not label_context:
        return None
    text = label_context.lower()
    if "pain" in text:
        return "pain"
    if any(cue in text for cue in ("notes", "comment", "summary", "describe", "explain")):
        return "notes"
    if text.rstrip().endswith("?"):
        return "yesno"
    if any(cue in text for cue in _NUMBER_LABEL_CUES):
        return "number"
    if any(cue in text for cue in _TEXT_LABEL_CUES):
        return "text"
    return None


def build_prompt(label_context=None, field_kind=None, style="answer"):
    """Compose the prompt for one cropped field."""
    if style not in PROMPT_STYLES:
        raise ValueError(f"Unknown prompt style: {style!r} (expected one of {PROMPT_STYLES})")

    if style == "answer" or field_kind in (None, "answer"):
        return _answer_prompt(label_context, field_kind)

    if field_kind not in _FIELDS:
        raise ValueError(f"Unknown field kind: {field_kind!r} (expected one of {FIELD_KINDS})")

    return _BUILDERS[style](field_kind, label_context)


PROMPTS = {
    (kind, style): _BUILDERS[style](kind, None)
    for kind in _FIELDS
    for style in _BUILDERS
}
