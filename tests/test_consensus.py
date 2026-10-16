import pytest

import consensus
from consensus import AGREED, BLANK, ILLEGIBLE, ILLEGIBLE_TEXT, REVIEW


class TestCanonical:
    @pytest.mark.parametrize("reading", [None, float("nan"), "[error: boom]", "[mlx unavailable: oom]"])
    def test_unusable_readings(self, reading):
        assert consensus.canonical(reading) is None

    @pytest.mark.parametrize("reading", ["blank", "(blank)", "  Blank ", ""])
    def test_blank_words(self, reading):
        assert consensus.canonical(reading) == BLANK

    @pytest.mark.parametrize("reading", ["illegible", "[illegible]", "[scratched out]"])
    def test_illegible_words(self, reading):
        assert consensus.canonical(reading) == ILLEGIBLE_TEXT

    def test_numbers_normalize(self):
        assert consensus.canonical("4.0", "number") == "4"
        assert consensus.canonical("about 12 miles", "number") == "12"
        assert consensus.canonical("2.5", "number") == "2.5"

    def test_number_without_digits_is_unusable(self):
        assert consensus.canonical("lots", "number") is None

    def test_pain_outside_0_to_10_is_unusable(self):
        assert consensus.canonical("14", "pain") is None
        assert consensus.canonical("10", "pain") == "10"
        assert consensus.canonical("0", "pain") == "0"

    def test_text_is_case_and_space_insensitive(self):
        assert consensus.canonical("  Outer   Left knee ", "text") == "outer left knee"


class TestDecide:
    def test_two_matching_readings_agree(self):
        d = consensus.decide(["7", "7.0"], field_kind="pain")
        assert (d.value, d.status) == ("7", AGREED)
        assert not d.needs_review

    def test_two_different_readings_need_review_and_keep_the_first(self):
        d = consensus.decide(["7", "1"], field_kind="pain")
        assert (d.value, d.status) == ("7", REVIEW)
        assert d.needs_review

    def test_single_reading_never_counts_as_agreed(self):
        assert consensus.decide(["3"], field_kind="number").status == REVIEW

    def test_illegible_readings_dont_vote_but_dont_block(self):
        d = consensus.decide(["[illegible]", "4", "4"], field_kind="number")
        assert (d.value, d.status) == ("4", AGREED)

    def test_one_legible_among_illegible_is_kept_but_flagged(self):
        d = consensus.decide(["[illegible]", "4"], field_kind="number")
        assert (d.value, d.status) == ("4", REVIEW)

    def test_all_illegible(self):
        d = consensus.decide(["[illegible]", None], field_kind="text")
        assert (d.value, d.status) == (ILLEGIBLE_TEXT, ILLEGIBLE)

    def test_two_of_three_agrees(self):
        d = consensus.decide(["ssh", "ssk", "SSH"], field_kind="text")
        assert (d.value, d.status) == ("ssh", AGREED)

    def test_two_of_four_is_not_enough(self):
        d = consensus.decide(["ssh", "ssk", "SSH", "sh"], field_kind="text")
        assert (d.value, d.status) == ("ssh", REVIEW)

    def test_text_value_keeps_the_models_casing(self):
        d = consensus.decide(["VPT020", "vpt020"], field_kind="text")
        assert (d.value, d.status) == ("VPT020", AGREED)

    def test_blank_agreement(self):
        d = consensus.decide(["blank", "(blank)"], field_kind="number")
        assert (d.value, d.status) == (BLANK, AGREED)

    def test_impossible_pain_reading_is_dropped_from_the_vote(self):
        d = consensus.decide(["14", "4", "4"], field_kind="pain")
        assert (d.value, d.status) == ("4", AGREED)
