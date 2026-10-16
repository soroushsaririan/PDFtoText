from PIL import Image

import ocr_pipeline as ocr


class TestUpscaleForVlm:
    def test_leaves_tall_enough_crops_unchanged(self):
        img = Image.new("RGB", (200, 200))
        out = ocr._upscale_for_vlm(img)
        assert out.size == (200, 200)

    def test_exactly_at_threshold_is_unchanged(self):
        img = Image.new("RGB", (100, ocr.VLM_MIN_CROP_HEIGHT))
        out = ocr._upscale_for_vlm(img)
        assert out.size == (100, ocr.VLM_MIN_CROP_HEIGHT)

    def test_upscales_short_crops_preserving_aspect_ratio(self):
        img = Image.new("RGB", (864, 44))
        out = ocr._upscale_for_vlm(img)
        assert out.height == ocr.VLM_MIN_CROP_HEIGHT
        assert out.width == round(864 * ocr.VLM_MIN_CROP_HEIGHT / 44)

    def test_respects_an_explicit_min_height(self):
        img = Image.new("RGB", (100, 20))
        out = ocr._upscale_for_vlm(img, min_height=40)
        assert out.height == 40
        assert out.width == 200


class TestReleaseVlmMemory:
    def test_runs_without_error_regardless_of_device(self):
        # No MPS/CUDA on the test machine is a valid case too -- this must
        # never raise, since _vlm_read() calls it after every single field.
        ocr._release_vlm_memory()


class TestStructuredValue:
    def test_answer_style_is_passed_through(self):
        assert ocr.structured_value("7", "pain", "answer") == "7"

    def test_pulls_the_value_field_out_of_a_structured_reply(self):
        raw = "1. DESCRIPTION: a single digit\n2. VALUE 0-10: 7\n3. CONFIDENCE: high"
        assert ocr.structured_value(raw, "pain", "discriminative") == "7"

    def test_missing_value_field_gives_none(self):
        assert ocr.structured_value("I can't tell", "number", "discriminative") is None


class TestReadLabelFields:
    def test_reads_each_found_label_per_plan_and_combines(self, tmp_path, monkeypatch):
        import field_reader

        page = Image.new("RGB", (1200, 400), "white")
        monkeypatch.setattr(field_reader, "locate_fields", lambda image, labels: iter([
            ("Days ran this week:", "days_ran", "number", "number", 1, (100, 200, 300, 230), None, 0),
        ]))
        calls = []

        def fake_generate(crop, prompt, budget):
            calls.append(prompt)
            return "2. VALUE AS WRITTEN: 4" if len(calls) == 3 else "4"

        # a page with no ink: the ink variant can't be built, so the next crop takes its pass
        reads = ocr.read_label_fields(page, tmp_path, passes=3, generate=fake_generate)
        assert len(calls) == 3
        assert [(r.key, r.value, r.status) for r in reads] == [("days_ran", "4", "agreed")]
        assert reads[0].readings == [["raised", "answer", "4"], ["label", "answer", "4"],
                                     ["raised", "discriminative", "4"]]
        assert (tmp_path / "days_ran_1.png").exists()

    def test_a_failing_read_is_recorded_not_raised(self, tmp_path, monkeypatch):
        import field_reader

        page = Image.new("RGB", (1200, 400), "white")
        monkeypatch.setattr(field_reader, "locate_fields", lambda image, labels: iter([
            ("Days ran this week:", "days_ran", "number", "number", 1, (100, 200, 300, 230), None, 0),
        ]))

        def broken(crop, prompt, budget):
            raise RuntimeError("out of memory")

        reads = ocr.read_label_fields(page, tmp_path, passes=1, generate=broken)
        assert reads[0].status == "illegible"
        assert reads[0].readings[0][2].startswith("[error")
