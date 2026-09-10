import pytest

import field_extraction as fx
from field_extraction import NOT_DOCUMENTED, UNCERTAIN

# A small label set covering all four kinds the extractor handles.
LABELS = [
    ("Participant ID:", "participant_id", "text"),
    ("Days ran this week:", "days_ran", "number"),
    ("Confident in correct running form?", "confident_form", "yesno"),
    ("Therapist / participant notes:", "notes", "text"),
]


class TestClean:
    def test_collapses_internal_whitespace(self):
        assert fx._clean("  a\t b\n c ") == "a b c"

    def test_all_whitespace_becomes_empty(self):
        assert fx._clean("   \n\t ") == ""


class TestLooksIllegible:
    @pytest.mark.parametrize(
        "value",
        [
            "",
            "   ",
            "[illegible]",
            "[scratched out]",
            "[VLM error: boom]",
            "[vlm unavailable: out of memory]",
            "[trocr unavailable: model not found]",
        ],
    )
    def test_illegible_markers(self, value):
        assert fx._looks_illegible(value)

    @pytest.mark.parametrize("value", ["3", "VPT001", "felt fine today"])
    def test_real_values(self, value):
        assert not fx._looks_illegible(value)


class TestExtractFields:
    def test_absent_label_reports_not_documented(self, page):
        fields, raw = fx.extract_fields([page()], LABELS)
        assert fields["participant_id"] == NOT_DOCUMENTED
        assert raw["participant_id"] is None

    def test_text_field_taken_from_following_handwriting(self, line, page):
        p = page(
            line("Participant ID:", "printed"),
            line("VPT001", "handwriting"),
        )
        fields, raw = fx.extract_fields([p], LABELS)
        assert fields["participant_id"] == "VPT001"
        assert raw["participant_id"] == "VPT001"

    def test_number_field_pulls_numeric_token_out_of_prose(self, line, page):
        p = page(
            line("Days ran this week:", "printed"),
            line("about 4 days", "handwriting"),
        )
        fields, _ = fx.extract_fields([p], LABELS)
        assert fields["days_ran"] == "4"

    def test_number_field_stops_at_first_handwriting_line(self, line, page):
        p = page(
            line("Days ran this week:", "printed"),
            line("3", "handwriting"),
            line("5", "handwriting"),
        )
        fields, raw = fx.extract_fields([p], LABELS)
        assert fields["days_ran"] == "3"
        assert raw["days_ran"] == "3"

    def test_number_field_without_digits_is_uncertain(self, line, page):
        p = page(
            line("Days ran this week:", "printed"),
            line("none", "handwriting"),
        )
        fields, _ = fx.extract_fields([p], LABELS)
        assert fields["days_ran"] == UNCERTAIN

    def test_yesno_reads_inline_answer_marker(self, line, page):
        p = page(line("Confident in correct running form?  [ANSWER: Yes]", "printed"))
        fields, _ = fx.extract_fields([p], LABELS)
        assert fields["confident_form"] == "Yes"

    def test_yesno_without_marker_is_uncertain(self, line, page):
        p = page(line("Confident in correct running form?", "printed"))
        fields, _ = fx.extract_fields([p], LABELS)
        assert fields["confident_form"] == UNCERTAIN

    def test_illegible_handwriting_is_uncertain_but_kept_in_raw(self, line, page):
        p = page(
            line("Participant ID:", "printed"),
            line("[illegible]", "handwriting"),
        )
        fields, raw = fx.extract_fields([p], LABELS)
        assert fields["participant_id"] == UNCERTAIN
        assert raw["participant_id"] == "[illegible]"

    def test_scratched_out_value_is_dropped_in_favor_of_rewrite(self, line, page):
        p = page(
            line("Participant ID:", "printed"),
            line("[scratched out]", "handwriting"),
            line("VPT002", "handwriting"),
        )
        fields, raw = fx.extract_fields([p], LABELS)
        assert fields["participant_id"] == "VPT002"
        assert raw["participant_id"] == "[scratched out] VPT002"

    def test_engine_error_line_does_not_become_the_field_value(self, line, page):
        p = page(
            line("Participant ID:", "printed"),
            line("[vlm unavailable: out of memory]", "handwriting_error"),
        )
        fields, _ = fx.extract_fields([p], LABELS)
        assert fields["participant_id"] == UNCERTAIN

    def test_notes_field_joins_consecutive_handwriting_lines(self, line, page):
        p = page(
            line("Therapist / participant notes:", "printed"),
            line("felt good", "handwriting"),
            line("no pain today", "handwriting"),
        )
        fields, _ = fx.extract_fields([p], LABELS)
        assert fields["notes"] == "felt good no pain today"

    def test_label_line_with_no_handwriting_after_it(self, line, page):
        p = page(
            line("Participant ID:", "printed"),
            line("Date: 2026-01-05", "printed"),
        )
        fields, raw = fx.extract_fields([p], LABELS)
        assert fields["participant_id"] == NOT_DOCUMENTED
        assert raw["participant_id"] is None


class TestExtractCheckedOptions:
    def test_x_marked_option_is_detected(self, line, page):
        p = page(
            line("[X] Hip Strength", "printed"),
            line("[ ] Core Strength", "printed"),
        )
        got = fx.extract_checked_options([p], ["Hip Strength", "Core Strength"])
        assert got == ["Hip Strength"]

    def test_non_printed_lines_are_ignored(self, line, page):
        p = page(line("[X] Hip Strength", "handwriting"))
        assert fx.extract_checked_options([p], ["Hip Strength"]) == []

    def test_unmarked_option_is_excluded(self, line, page):
        p = page(line("[ ] Stretching", "printed"))
        assert fx.extract_checked_options([p], ["Stretching"]) == []

    def test_known_limitation_mark_bleeds_across_a_shared_ocr_line(self, line, page):
        # When Tesseract merges two checkboxes onto one line, the X in front of
        # the first option also satisfies the second option's prefix test.
        # This documents current behavior; see the polish notes.
        p = page(line("[X] Hip Strength [ ] Core Strength", "printed"))
        got = fx.extract_checked_options([p], ["Hip Strength", "Core Strength"])
        assert got == ["Hip Strength", "Core Strength"]


class TestRecordWrappers:
    def test_weekly_record_attaches_option_lists(self, line, page):
        p = page(
            line("Participant ID:", "printed"),
            line("VPT001", "handwriting"),
            line("[X] Hip Strength", "printed"),
        )
        fields, _ = fx.extract_weekly_record([p])
        assert fields["participant_id"] == "VPT001"
        assert fields["exercise_types"] == ["Hip Strength"]
        assert fields["cues_list"] == []

    def test_weekly_record_defaults_exercise_types_when_none_checked(self, page):
        fields, _ = fx.extract_weekly_record([page()])
        assert fields["exercise_types"] == [NOT_DOCUMENTED]

    def test_intake_final_exit_wrappers_return_their_label_keys(self, page):
        for extractor, expected_key in (
            (fx.extract_intake_record, "injury_diagnosis"),
            (fx.extract_final_record, "days_exercised"),
            (fx.extract_exit_record, "would_recommend"),
        ):
            fields, _ = extractor([page()])
            assert expected_key in fields
            assert fields[expected_key] == NOT_DOCUMENTED


class TestLoadKeywords:
    def test_reads_lines_skipping_comments_and_blanks(self, tmp_path):
        f = tmp_path / "kw.txt"
        f.write_text("# a comment\n\npain\n  swelling  \n#another\n", encoding="utf-8")
        assert fx.load_keywords(str(f)) == ["pain", "swelling"]

    def test_missing_file_returns_empty_list(self, tmp_path):
        assert fx.load_keywords(str(tmp_path / "nope.txt")) == []


class TestKeywordSearch:
    def test_case_insensitive_hit_with_surrounding_context(self):
        text = "The participant reported sharp PAIN in the left knee after running."
        hits = fx.keyword_search(text, ["pain"])
        assert len(hits) == 1
        assert hits[0]["keyword"] == "pain"
        assert "knee" in hits[0]["context"]

    def test_every_occurrence_is_reported(self):
        assert len(fx.keyword_search("pain pain pain", ["pain"])) == 3

    def test_no_match_yields_no_hits(self):
        assert fx.keyword_search("all clear", ["pain"]) == []

    def test_context_window_is_bounded_by_context_chars(self):
        text = "x" * 200 + "pain" + "y" * 200
        hit = fx.keyword_search(text, ["pain"], context_chars=10)[0]
        assert hit["context"] == "x" * 10 + "pain" + "y" * 10
