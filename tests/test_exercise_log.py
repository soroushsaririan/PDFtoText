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


class TestRanDay:
    def test_a_run_without_a_distance_still_counts(self):
        """Half the entries name a run and no mileage: "20 minutes
        running" is a running day even though days_ran used to need miles."""
        assert el.ran_day("20 minutes running") is True
        assert el.ran_day("30 minutes running, 1 hour biking") is True

    def test_a_bare_distance_counts(self):
        assert el.ran_day("2.15 miles") is True

    def test_cross_training_alone_does_not(self):
        assert el.ran_day("Bike Ride: 55 miles, 3 hours 30 mins.") is False
        assert el.ran_day("Biking 10.6 miles, 38 minutes") is False

    def test_empty(self):
        assert el.ran_day(None) is False
        assert el.ran_day("  ") is False


class TestMinutes:
    def test_plain_minutes(self):
        assert el.parse_minutes("4.33 miles Run, 40 minutes") == 40

    def test_abbreviated(self):
        assert el.parse_minutes("4.2 mile run (43 mins),") == 43

    def test_hours_and_minutes(self):
        assert el.parse_minutes("31.08 mile run, 6 hours 5 minutes") == 365

    def test_a_clock_time_is_a_duration(self):
        assert round(el.parse_minutes("Running, 2.42 miles, 27:50")) == 28

    def test_an_explicit_wording_wins_over_a_clock(self):
        assert el.parse_minutes("ran 1 hour 5 minutes (1:05)") == 65

    def test_a_mixed_cell_contributes_nothing(self):
        """Otherwise the hour on the bike lands in the week's running minutes."""
        assert el.parse_minutes("30 minutes running, 1 hour biking") is None

    def test_no_duration(self):
        assert el.parse_minutes("2.15 miles") is None
        assert el.parse_minutes(None) is None


class TestSummarizeZeros:
    def test_a_zero_count_is_not_a_session(self):
        """A week reading 0, 2, 0, 2, 3 exercises used to report 0 per
        session -- the tie between 0 and 2 was broken by set order."""
        days = [{"exercise_count": c} for c in ("0", None, "2", "0", "2", None, "3")]
        got = el.summarize(days)
        assert got["exercises_per_session"] == 2
        assert got["days_exercised"] == 3

    def test_a_week_of_only_zeros_reports_no_session_size(self):
        got = el.summarize([{"exercise_count": "0"}, {"exercise_count": "0"}])
        assert "exercises_per_session" not in got
        assert got["days_exercised"] == 0

    def test_a_tie_is_resolved_the_same_way_every_time(self):
        got = [el.summarize([{"exercise_count": "5"}, {"exercise_count": "2"}])
               ["exercises_per_session"] for _ in range(20)]
        assert set(got) == {2}

    def test_a_written_count_stands_in_for_an_unticked_box(self):
        """Most participants leave the tick box empty and write a number."""
        days = [{"completed": "", "exercise_count": "4"},
                {"completed": "", "exercise_count": None}]
        assert el.summarize(days)["days_exercised"] == 1

    def test_a_tick_wins_over_a_missing_count(self):
        assert el.summarize([{"completed": "CHECK", "exercise_count": None}])["days_exercised"] == 1

    def test_running_minutes_are_totalled_over_the_week(self):
        days = [{"running": "4.33 miles Run, 40 minutes"},
                {"running": "5.75 miles run, 55 minutes"},
                {"running": "Biking 10.6 miles, 38 minutes"}]
        got = el.summarize(days)
        assert got["running_minutes"] == 95
        assert got["days_ran"] == 2
        assert got["mileage"] == 10.08

    def test_a_run_with_no_distance_gives_days_but_no_mileage(self):
        got = el.summarize([{"running": "20 minutes running"}])
        assert got["days_ran"] == 1
        assert "mileage" not in got


class TestImplausible:
    def test_an_inflated_digit_is_flagged(self):
        """"4.1 miles" read as "41 miles" put a 52-mile week in the output."""
        flags = el.implausible([{"day": 1, "running": "4 miles"}, {"day": 7, "running": "41 miles"}])
        assert [f[0] for f in flags] == ["mileage"]
        assert "day 7" in flags[0][1]

    def test_an_ordinary_week_is_not_flagged(self):
        assert el.implausible([{"day": d, "running": "3 miles"} for d in el.DAYS]) == []

    def test_a_long_week_is_flagged_even_when_each_day_is_plausible(self):
        days = [{"day": d, "running": "20 miles"} for d in el.DAYS]
        assert any(f[0] == "mileage" and "over the week" in f[1] for f in el.implausible(days))

    def test_an_absurd_exercise_count_is_flagged(self):
        flags = el.implausible([{"day": 2, "exercise_count": "55"}])
        assert [f[0] for f in flags] == ["exercises_per_session"]

    def test_cross_training_mileage_is_not_flagged_as_running(self):
        assert el.implausible([{"day": 1, "running": "Bike Ride: 55 miles, 3 hours 30 mins."}]) == []


class TestDistanceWrittenNextToRun:
    """VPT003 logs every day as "<miles> run; <miles> bike", which the
    mile-word patterns alone read as no mileage at all."""

    def test_a_distance_before_the_word_run(self):
        assert el.parse_miles("5 run") == 5.0

    def test_no_space_is_still_a_distance(self):
        assert el.parse_miles("6run") == 6.0

    def test_the_run_figure_wins_over_the_cross_training_one(self):
        assert el.parse_miles("8.5 run, 15 bike") == 8.5
        assert el.parse_miles("6.5 run, 10 bike") == 6.5

    def test_a_run_day_even_without_a_space(self):
        assert el.ran_day("6run") is True

    def test_minutes_are_not_read_as_miles(self):
        assert el.parse_miles("20 minutes running") is None
