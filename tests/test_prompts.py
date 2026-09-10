import pytest

import prompts


class TestInferFieldKind:
    @pytest.mark.parametrize(
        "label,kind",
        [
            ("Pain RIGHT NOW (0-10):", "pain"),
            ("USUAL pain during the last week:", "pain"),
            ("Therapist / participant notes:", "notes"),
            ("Additional comments:", "notes"),
            ("Overall summary / therapist notes:", "notes"),
            ("Any exercises too difficult?", "yesno"),
            ("Days ran this week:", "number"),
            ("Number of exercises per session:", "number"),
            ("Age (yrs):", "number"),
            ("Participant ID:", "text"),
            ("Date:", "text"),
            ("Injury diagnosis (if known):", "text"),
        ],
    )
    def test_kind_from_label_text(self, label, kind):
        assert prompts.infer_field_kind(label) == kind

    def test_unrecognized_label_is_none(self):
        assert prompts.infer_field_kind("Left knee circumference") is None

    def test_empty_label_is_none(self):
        assert prompts.infer_field_kind("") is None

    def test_pain_takes_precedence_over_question_mark(self):
        assert prompts.infer_field_kind("Worst pain this week?") == "pain"


class TestBuildPrompt:
    def test_unknown_style_raises(self):
        with pytest.raises(ValueError):
            prompts.build_prompt(style="verbose")

    def test_answer_is_the_default_style_and_names_the_label(self):
        out = prompts.build_prompt(label_context="Days ran this week:", field_kind="number")
        assert "Days ran this week:" in out
        assert out == prompts.build_prompt(
            label_context="Days ran this week:", field_kind="number", style="answer"
        )

    def test_none_field_kind_falls_back_to_answer_prompt(self):
        out = prompts.build_prompt(label_context="X:", field_kind=None, style="descriptive")
        assert "Read the handwritten answer" in out

    def test_descriptive_style_for_a_known_kind(self):
        out = prompts.build_prompt(
            label_context="Pain now:", field_kind="pain", style="descriptive"
        )
        assert "two or three sentences" in out

    def test_unknown_field_kind_with_a_real_style_raises(self):
        with pytest.raises(ValueError):
            prompts.build_prompt(field_kind="mystery", style="descriptive")


class TestOptionsBlock:
    def test_cue_label_lists_only_cues(self):
        block = prompts._options_block("Running-form cues used:")
        assert "Faster cadence" in block
        assert "Hip Strength" not in block

    def test_exercise_label_lists_only_exercises(self):
        block = prompts._options_block("Exercise types this week:")
        assert "Hip Strength" in block
        assert "Faster cadence" not in block

    def test_ambiguous_label_lists_the_union(self):
        block = prompts._options_block("something else entirely")
        assert "Hip Strength" in block
        assert "Faster cadence" in block
