import fitz
import pytest
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


class TestLooksLikePdf:
    def test_a_real_pdf_is_recognized(self, tmp_path):
        p = tmp_path / "form.pdf"
        p.write_bytes(b"%PDF-1.7\nrest of the file")
        assert ocr.looks_like_pdf(p) is True

    def test_a_jpeg_renamed_to_pdf_is_not(self, tmp_path):
        p = tmp_path / "form.pdf"
        p.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")
        assert ocr.looks_like_pdf(p) is False

    def test_a_missing_file_is_not(self, tmp_path):
        assert ocr.looks_like_pdf(tmp_path / "nope.pdf") is False


class TestRenderFileToImages:
    """Form files are dispatched on their actual format: study staff hand us
    phone photos copy-renamed to ".pdf"."""

    def test_a_jpeg_renamed_to_pdf_is_rendered_as_an_image(self, tmp_path):
        src = tmp_path / "Exercise Log week 1 .pdf"
        Image.new("RGB", (120, 80), "white").save(src, format="JPEG")

        got = ocr.render_file_to_images(src, tmp_path / "out")

        assert [p.name for p in got] == ["page_01.png"]
        assert Image.open(got[0]).size == (120, 80)

    def test_a_heic_renamed_to_pdf_is_rendered_as_an_image(self, tmp_path):
        src = tmp_path / "Exercise Log week 3 .pdf"
        Image.new("RGB", (64, 48), "white").save(src, format="HEIF")

        got = ocr.render_file_to_images(src, tmp_path / "out")

        assert Image.open(got[0]).size == (64, 48)

    def test_an_unreadable_file_raises_unreadable_file(self, tmp_path):
        src = tmp_path / "truncated.pdf"
        src.write_bytes(b"%PDF-1.4\nnot really a pdf at all")

        with pytest.raises(ocr.UnreadableFile) as caught:
            ocr.render_file_to_images(src, tmp_path / "out")

        assert "truncated.pdf" in str(caught.value)

    def test_a_real_pdf_still_goes_through_the_pdf_renderer(self, tmp_path):
        src = tmp_path / "form.pdf"
        doc = fitz.open()
        doc.new_page()
        doc.new_page()
        doc.save(str(src))
        doc.close()

        got = ocr.render_file_to_images(src, tmp_path / "out")

        assert [p.name for p in got] == ["page_01.png", "page_02.png"]


class TestUprightLogPage:
    """VPT003's four Exercise Logs are scanned sideways; upright they read
    as blank pages. Tesseract is stubbed out so the test is about which
    turn of the page gets picked, not about OCR."""

    HEADERS = [
        ("Exercise", 560, 460), ("Log", 760, 460), ("Day", 586, 657),
        ("Date", 696, 655), ("Completed", 846, 650), ("Number", 1122, 648),
        ("What", 1377, 645), ("difficulty", 1737, 640), ("symptoms", 2107, 635),
    ]

    def _words(self, entries):
        data = {k: [] for k in ("text", "conf", "left", "top", "width", "height",
                                "block_num", "par_num", "line_num")}
        for n, (text, x, y) in enumerate(entries):
            data["text"].append(text)
            data["conf"].append(95.0)
            data["left"].append(x)
            data["top"].append(y)
            data["width"].append(60)
            data["height"].append(30)
            for k in ("block_num", "par_num"):
                data[k].append(1)
            data["line_num"].append(n)
        return data

    def _readable_at(self, angle, monkeypatch, extra=()):
        """Stub tesseract so only `angle` yields a readable table."""
        landscape = self._words(self.HEADERS + list(extra))
        blank = self._words([])

        def words(image):
            return landscape if image.width > image.height else blank

        monkeypatch.setattr(ocr, "log_page_words", words)
        return landscape, blank

    def test_a_sideways_page_is_turned_upright(self, monkeypatch):
        days = [(str(d), 578, 1008 + (d - 1) * 92) for d in (1, 4, 7)]
        self._readable_at(270, monkeypatch, days)
        page = Image.new("RGB", (1836, 2376), "white")
        upright, data = ocr.upright_log_page(page)
        assert upright is not None and upright.width > upright.height
        assert data["text"][0] == "Exercise"

    def test_an_upright_page_is_left_alone(self, monkeypatch):
        days = [(str(d), 578, 1008 + (d - 1) * 92) for d in (1, 4, 7)]
        self._readable_at(0, monkeypatch, days)
        page = Image.new("RGB", (2376, 1836), "white")
        upright, _ = ocr.upright_log_page(page)
        assert upright is page

    def test_a_page_that_is_not_a_log_is_rejected_in_every_turn(self, monkeypatch):
        monkeypatch.setattr(ocr, "log_page_words", lambda image: self._words([("Participant", 10, 10)]))
        page = Image.new("RGB", (1836, 2376), "white")
        assert ocr.upright_log_page(page) == (None, None)

    def test_a_turn_that_locates_the_table_beats_one_that_only_sees_headers(self, monkeypatch):
        """Without the day numbers a sideways page can still match enough
        header words to look like a log, so headers alone can't decide."""
        day_rows = [(str(d), 578, 1008 + (d - 1) * 92) for d in (1, 4, 7)]
        with_rows = self._words(self.HEADERS + day_rows)
        headers_only = self._words(self.HEADERS)

        def words(image):
            return with_rows if image.height > image.width else headers_only

        monkeypatch.setattr(ocr, "log_page_words", words)
        page = Image.new("RGB", (2376, 1836), "white")
        upright, _ = ocr.upright_log_page(page)
        assert upright.height > upright.width
