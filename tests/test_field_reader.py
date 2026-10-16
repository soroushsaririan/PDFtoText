import numpy as np
from PIL import Image, ImageDraw

import field_reader as fr

LINE_H = 30


def _words(*words):
    """Fake pytesseract image_to_data output: (text, left, top, width, height, conf)."""
    data = {k: [] for k in ("text", "left", "top", "width", "height", "conf", "block_num", "par_num", "line_num")}
    for n, (text, left, top, width, height, conf) in enumerate(words):
        data["text"].append(text)
        data["left"].append(left)
        data["top"].append(top)
        data["width"].append(width)
        data["height"].append(height)
        data["conf"].append(conf)
        data["block_num"].append(1)
        data["par_num"].append(1)
        data["line_num"].append(top // LINE_H)
    return data


def _form_page():
    """White page: a label at y 200-230 with an underline at y 228 running
    right of it, a "previous row" answer stroke entirely above the field's
    band, and this field's answer: a tall stroke rising from the underline
    well above the label line."""
    page = Image.new("RGB", (1200, 400), "white")
    d = ImageDraw.Draw(page)
    d.line((320, 228, 900, 228), fill="black", width=2)        # underline
    d.line((500, 120, 500, 150), fill="black", width=4)        # previous row's answer
    d.line((600, 160, 600, 226), fill="black", width=4)        # this field's answer
    return page


class TestCeilingAbove:
    def test_finds_bottom_of_printed_line_above(self):
        data = _words(("Days", 100, 150, 60, 30, 95), ("Mileage", 100, 200, 120, 30, 95))
        assert fr.ceiling_above(data, (100, 200, 300, 230), answer_x1=900) == 180

    def test_ignores_low_confidence_words(self):
        data = _words(("~", 100, 150, 60, 30, 20))
        assert fr.ceiling_above(data, (100, 200, 300, 230), answer_x1=900) == 0

    def test_ignores_words_outside_the_horizontal_span(self):
        data = _words(("Elsewhere", 1000, 150, 100, 30, 95))
        assert fr.ceiling_above(data, (100, 200, 300, 230), answer_x1=900) == 0

    def test_ignores_words_on_the_same_line(self):
        data = _words(("Mileage", 100, 200, 120, 30, 95))
        assert fr.ceiling_above(data, (100, 200, 300, 230), answer_x1=900) == 0


class TestCropVariants:
    def test_label_variant_matches_answer_crop(self):
        page = _form_page()
        crops = fr.crop_variants(page, (100, 200, 300, 230), "number")
        expected = fr.answer_crop(page, (100, 200, 300, 230), "number")
        assert np.array_equal(np.array(crops["label"]), np.array(expected))

    def test_raised_takes_in_a_stroke_rising_above_the_band(self):
        page = _form_page()
        box = fr.answer_box(page.size, (100, 200, 300, 230), "number")
        crops = fr.crop_variants(page, (100, 200, 300, 230), "number", ceiling=190)
        top = box[3] - crops["raised"].height
        # the answer stroke starts at y=160, above the ceiling, and is kept whole
        assert top <= 160
        # the previous row's stroke (y 120-150) never touches the band and stays out
        assert top > 150

    def test_raised_stays_at_the_band_without_rising_strokes(self):
        page = Image.new("RGB", (1200, 400), "white")
        ImageDraw.Draw(page).line((320, 228, 900, 228), fill="black", width=2)
        crops = fr.crop_variants(page, (100, 200, 300, 230), "number", ceiling=190)
        box = fr.answer_box(page.size, (100, 200, 300, 230), "number")
        assert box[3] - crops["raised"].height == 190

    def test_ink_variant_skipped_when_only_the_underline_is_there(self):
        page = Image.new("RGB", (1200, 400), "white")
        ImageDraw.Draw(page).line((320, 228, 900, 228), fill="black", width=2)
        crops = fr.crop_variants(page, (100, 200, 300, 230), "number")
        assert "ink" not in crops

    def test_ink_variant_hugs_the_handwriting(self):
        crops = fr.crop_variants(_form_page(), (100, 200, 300, 230), "number", ceiling=190)
        assert crops["ink"].width < crops["raised"].width

    def test_notes_crops_are_not_raised(self):
        page = _form_page()
        crops = fr.crop_variants(page, (100, 200, 300, 230), "notes")
        assert crops["raised"].size == crops["label"].size

    def test_degenerate_region_gives_nothing(self):
        page = Image.new("RGB", (400, 400), "white")
        assert fr.crop_variants(page, (100, 200, 390, 230), "number", max_x1=395) == {}


class TestNextLabelStartX:
    def test_stops_at_the_next_printed_label(self):
        data = _words(("Weekly", 171, 200, 100, 30, 91), ("visit:", 279, 200, 57, 30, 96),
                      ("3", 603, 200, 20, 30, 96), ("Tester:", 923, 200, 90, 30, 96))
        assert fr.next_label_start_x(data, range(4), after_x=336) == 923

    def test_handwriting_misread_as_ending_in_colon_or_question_mark_is_not_a_label(self):
        # a circled "2" came back as "(2?", a scrawled "VPT" as "Vf?" -- both cut
        # the answer crop off before the actual answer
        data = _words(("(2?", 468, 200, 60, 30, 69), ("Vf?", 520, 200, 40, 30, 39),
                      ("3:", 605, 200, 20, 30, 28), ("Tester:", 923, 200, 90, 30, 96))
        assert fr.next_label_start_x(data, range(4), after_x=336) == 923
