"""Labels have to be the form's real printed wording.

field_reader.label_hits strips non-alphanumerics and looks for the label as
a substring of ONE printed OCR line. A paraphrased label therefore fails
silently -- no error, no warning, the field just reports "Not documented"
forever. That is exactly what happened: WEEKLY_LABELS was written from a
paraphrase of the form and only 4 of 17 labels ever matched, so six of the
eight weekly summary columns were empty for every participant.

The lines below are the printed text of the blank forms (no participant
answers), transcribed from the study CRFs. Each label must be locatable in
them, and must survive the >= 6 normalized characters that label_hits
requires before it will match anything at all.
"""

import pytest

import field_extraction as fx
import field_reader

# Printed questions, one string per physical line on the form.
WEEKLY_FORM_LINES = [
    "Weekly Virtual Physical Therapy (VPT) session Case Report Form",
    "Participant ID: Date:",
    "Weekly visit: 1 2 3 4 Tester:",
    "Adherence to home exercise program",
    "Days completed the exercise as prescribed on handout (#)",
    "Number of exercises completed in each session (#)",
    "What type of exercises did you perform? Hip Strength",
    "Any exercises that were too difficult? Yes No",
    "Were you confident that you had correct form? Yes No",
    "Were you able to incorporate any cues on running",
    "form into your training? Yes No",
    "Participation in Running & Other Exercise",
    "How many days did you run this week? (#)",
    "What was the mileage this week? (#)",
    "Did you participate in any other exercise (s)? Yes No",
    "Musculoskeletal Pain",
    "Since last visit, did you experience any new pain symptoms? Yes No",
    "How would you rate your overall body pain RIGHT NOW? (pts)",
    "How would you rate your USUAL pain level during the last week? (pts)",
    "How would you rate your BEST pain level during the last week? (pts)",
    "How would you rate your WORST pain level during the last week? (pts)",
]

INTAKE_FORM_LINES = [
    "Visit 1 Case Report Form",
    "Participant ID: Date: Tester:",
    "Age (yrs)",
    "Total minutes of running a week (min)",
    "Total distance running per week? (miles)",
    "Injury diagnosis (in known)",
    "Current injury location",
    "How would you rate your body pain RIGHT NOW? (pts)",
    "How would you rate your USUAL pain level during the last week? (pts)",
    "How would you rate your BEST pain level during the last week? (pts)",
    "How would you rate your WORST pain level during the last week? (pts)",
]

FINAL_FORM_LINES = [
    "Final Visit (week 25) Case Report Form",
    "Age (yrs)",
    "Total minutes of running a week (min)",
    "Total distance running per week? (miles)",
    "How would you rate your body pain RIGHT NOW? (pts)",
    "How would you rate your USUAL pain level during the last week? (pts)",
    "How would you rate your BEST pain level during the last week? (pts)",
    "How would you rate your WORST pain level during the last week? (pts)",
    "Days completed the exercise as prescribed on handout (days per week)",
    "Number of exercises completed in each session (# per session)",
    "Any exercises that were too difficult? Yes No",
    "Are you confident now that you have correct running form? Yes No",
]

# Printed on the form but with no field to read, so no label targets them.
UNPRINTED = {
    "weekly": {"date", "running_minutes", "notes"},
    "intake": set(),
    "final": {"participant_id", "notes"},
}


def fake_ocr(lines):
    """A tesseract image_to_data dict for printed lines, one word per box."""
    data = {k: [] for k in
            ("text", "conf", "left", "top", "width", "height", "block_num", "par_num", "line_num")}
    for line_no, line in enumerate(lines):
        x = 0
        for word in line.split():
            data["text"].append(word)
            data["conf"].append(96.0)
            data["left"].append(x)
            data["top"].append(line_no * 40)
            data["width"].append(len(word) * 12)
            data["height"].append(24)
            data["block_num"].append(1)
            data["par_num"].append(1)
            data["line_num"].append(line_no)
            x += len(word) * 12 + 10
    return data


def located(labels, form_lines):
    data = fake_ocr(form_lines)
    lines = field_reader.line_index(data)
    return {key for label, key, _ in labels if field_reader.label_hits(data, lines, label)}


@pytest.mark.parametrize("name, labels, form_lines", [
    ("weekly", fx.WEEKLY_LABELS, WEEKLY_FORM_LINES),
    ("intake", fx.INTAKE_LABELS, INTAKE_FORM_LINES),
    ("final", fx.FINAL_LABELS, FINAL_FORM_LINES),
])
def test_every_label_is_found_on_its_form(name, labels, form_lines):
    expected = {key for _, key, _ in labels} - UNPRINTED[name]
    assert located(labels, form_lines) >= expected


@pytest.mark.parametrize("name, labels", [
    ("weekly", fx.WEEKLY_LABELS),
    ("intake", fx.INTAKE_LABELS),
    ("final", fx.FINAL_LABELS),
    ("exit", fx.EXIT_LABELS),
])
def test_labels_are_long_enough_to_match_at_all(name, labels):
    """label_hits refuses anything under 6 normalized characters, so a
    shorter label can never match however correct its wording."""
    too_short = [label for label, key, _ in labels
                 if len(field_reader._norm(label)) < 6 and key not in UNPRINTED.get(name, set())]
    assert too_short == []


def test_pain_now_matches_both_printed_variants():
    """Weekly says "overall body pain RIGHT NOW?", the visit forms say "body
    pain RIGHT NOW?" -- one label has to cover both."""
    label = next(l for l, k, _ in fx.WEEKLY_LABELS if k == "pain_now")
    for line in ("How would you rate your overall body pain RIGHT NOW? (pts)",
                 "How would you rate your body pain RIGHT NOW? (pts)"):
        data = fake_ocr([line])
        assert field_reader.label_hits(data, field_reader.line_index(data), label)
