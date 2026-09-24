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
