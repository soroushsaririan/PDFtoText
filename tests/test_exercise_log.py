"""Exercise Log table geometry and weekly totals.

Everything here is model-free: the table is located from tesseract word
boxes, so the tests build those boxes directly.
"""

from PIL import Image

import exercise_log as el


def words(entries):
    """A tesseract image_to_data dict from (text, x, y, w, h) tuples."""
    data = {k: [] for k in ("text", "conf", "left", "top", "width", "height",
                            "block_num", "par_num", "line_num")}
    for n, (text, x, y, w, h) in enumerate(entries):
        data["text"].append(text)
        data["conf"].append(95.0)
        data["left"].append(x)
        data["top"].append(y)
        data["width"].append(w)
        data["height"].append(h)
        data["block_num"].append(1)
        data["par_num"].append(1)
        data["line_num"].append(n)
    return data


HEADERS = [
    ("Exercise", 560, 460, 180, 32), ("Log", 760, 460, 76, 32),
    ("Day", 586, 657, 64, 34), ("Date", 696, 655, 84, 27),
    ("Completed", 846, 650, 183, 32), ("Number", 1122, 648, 146, 31),
    ("What", 1377, 645, 122, 42), ("difficulty", 1737, 640, 155, 35),
    ("symptoms", 2107, 635, 182, 34), ("Participation", 2545, 630, 228, 38),
    ("Location", 2994, 625, 151, 32),
]


def page(extra=()):
    return words(HEADERS + list(extra))


def day_digits(days, first_y=1008, pitch=92):
    return [(str(d), 578, first_y + (d - 1) * pitch, 20, 28) for d in days]


class TestDetection:
    def test_recognizes_an_exercise_log(self):
        assert el.is_exercise_log(page()) is True

    def test_a_session_form_is_not_an_exercise_log(self):
        assert el.is_exercise_log(words([("Participant", 560, 460, 180, 32),
                                         ("Weekly", 760, 460, 76, 32)])) is False

    def test_all_nine_columns_are_located(self):
        assert set(el.header_positions(page())) == {
            "day", "date", "completed", "exercise_count", "exercise_types",
            "difficulty", "new_pain", "running", "pain_location"}


class TestRowFitting:
    def test_missing_day_numbers_are_interpolated(self):
        """Tesseract typically finds only some of the printed day digits."""
        table = el.find_table(page(day_digits([1, 3, 7])), 4032, 2268)
        ys = [table.rows[d] for d in el.DAYS]
        gaps = [round(b - a) for a, b in zip(ys, ys[1:])]
        assert gaps == [92] * 6

    def test_a_stray_digit_does_not_drag_the_fit(self):
        """A handwritten date landing in the day column used to produce a
        row pitch of 28px or 525px on a page whose rows are 92px apart."""
        stray = [("4", 575, 1040, 18, 26)]  # far from where day 4 belongs
        table = el.find_table(page(day_digits([1, 2, 3, 5, 7]) + stray), 4032, 2268)
        ys = [table.rows[d] for d in el.DAYS]
        gaps = [round(b - a) for a, b in zip(ys, ys[1:])]
        assert gaps == [92] * 6

    def test_too_few_day_anchors_gives_no_table(self):
        assert el.find_table(page(day_digits([4])), 4032, 2268) is None

    def test_an_implausible_pitch_is_rejected(self):
        """Two digits 445px apart on an 1836px page cannot both be day rows."""
        far = [("1", 235, 835, 18, 26), ("2", 234, 1280, 18, 26)]
        assert el.find_table(page(far), 2376, 1836) is None


class TestTilt:
    def test_the_header_row_measures_the_page_tilt(self):
        slope = el.page_slope(el.header_positions(page()))
        assert -0.05 < slope < 0

    def test_cells_are_offset_by_the_tilt(self):
        """A photographed form is not axis-aligned: the right-hand columns
        sit higher than the left, so their crops must too."""
        table = el.find_table(page(day_digits([1, 4, 7])), 4032, 2268)
        left = table.cell(1, "completed", 4032, 2268)
        right = table.cell(1, "running", 4032, 2268)
        assert right[1] < left[1]


class TestMiles:
    def test_plain_mileage(self):
        assert el.parse_miles("2.15 miles") == 2.15

    def test_a_bare_number_is_taken_as_miles(self):
        assert el.parse_miles("3") == 3.0

    def test_cross_training_is_not_running_mileage(self):
        assert el.parse_miles("28 mile bike") is None
        assert el.parse_miles("400 metre swim") is None

    def test_a_run_described_with_other_exercise_still_counts(self):
        assert el.parse_miles("ran 4 miles, then yoga") == 4.0

    def test_empty(self):
        assert el.parse_miles(None) is None
        assert el.parse_miles("") is None


class TestCompletedMark:
    def test_a_check_is_done(self):
        assert el.completed_day("CHECK") is True

    def test_a_cross_is_not_done(self):
        assert el.completed_day("X") is False

    def test_blank_is_unknown(self):
        assert el.completed_day("") is None
        assert el.completed_day(None) is None


class TestSummarize:
    def test_counts_days_and_totals_mileage(self):
        days = [
            {"completed": "CHECK", "running": "2.15 miles", "exercise_count": "5"},
            {"completed": "CHECK", "running": "", "exercise_count": "5"},
            {"completed": "X", "running": "3 miles", "exercise_count": ""},
            {"completed": "CHECK", "running": "28 mile bike", "exercise_count": "5"},
        ]
        got = el.summarize(days)
        assert got["days_exercised"] == 3
        assert got["days_ran"] == 2
        assert got["mileage"] == 5.15
        assert got["exercises_per_session"] == 5

    def test_pain_columns_are_not_invented(self):
        """The log asks about pain per exercise session, which is not the
        session form's pain_now/usual/worst -- so it must not fill them."""
        got = el.summarize([{"completed": "CHECK", "new_pain": "4"}])
        assert "pain_now" not in got and "pain_usual" not in got and "pain_worst" not in got

    def test_an_empty_log_reports_nothing(self):
        assert el.summarize([{"completed": None, "running": None}]) == {}


class TestReadTable:
    def test_reads_each_requested_cell_once_per_day(self):
        calls = []

        def read_cell(crop, column, day):
            calls.append((column, day))
            return "CHECK" if column == "completed" else None

        img = Image.new("RGB", (4032, 2268), "white")
        days, summary = el.read_table(img, page(day_digits([1, 4, 7])), read_cell,
                                      columns=("completed", "running"))
        assert len(days) == 7
        assert sorted(calls) == sorted([(c, d) for c in ("completed", "running") for d in el.DAYS])
        assert summary["days_exercised"] == 7

    def test_a_page_with_no_table_reads_nothing(self):
        img = Image.new("RGB", (4032, 2268), "white")
        days, summary = el.read_table(img, page(), lambda *a: "CHECK")
        assert days is None and summary is None
