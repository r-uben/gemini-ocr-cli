"""Tests for OCR processor module."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from gemini_ocr.config import Config
from gemini_ocr.processor import OCRProcessor, OCRResult


class TestOCRResult:
    """Tests for OCRResult dataclass."""

    def test_success_true(self):
        result = OCRResult(
            file_path=Path("test.pdf"),
            pages=["Extracted content"],
            success=True,
            processing_time=1.0,
        )
        assert result.success is True
        assert result.text == "Extracted content"
        assert result.page_count == 1

    def test_success_false(self):
        result = OCRResult(
            file_path=Path("test.pdf"),
            pages=[],
            success=False,
            error="Processing failed",
            processing_time=1.0,
        )
        assert result.success is False
        assert result.error == "Processing failed"


class TestOCRProcessor:
    """Tests for OCRProcessor class."""

    @pytest.fixture
    def processor(self, mock_config, mock_genai_client):
        with patch("gemini_ocr.processor.genai") as mock_genai:
            mock_genai.Client.return_value = mock_genai_client
            processor = OCRProcessor(mock_config)
            processor.client = mock_genai_client
            return processor

    def test_init_creates_client(self, mock_config):
        with patch("gemini_ocr.processor.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            OCRProcessor(mock_config)
            mock_genai.Client.assert_called_once_with(api_key=mock_config.api_key)

    def test_init_raises_without_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            env = {
                k: v for k, v in os.environ.items() if k not in ("GEMINI_API_KEY", "GOOGLE_API_KEY")
            }
            with patch.dict(os.environ, env, clear=True):
                config = Config()
                config.api_key = ""
                with pytest.raises(ValueError, match="API key not set"):
                    OCRProcessor(config)

    def test_process_image(self, processor, sample_image):
        result = processor.process_image(sample_image)
        assert result.success
        assert "Extracted text" in result.text

    def test_process_image_with_custom_prompt(self, processor, sample_image):
        result = processor.process_image(sample_image, custom_prompt="Extract only numbers")
        assert processor.client.models.generate_content.called

    def test_process_pdf(self, processor, sample_pdf):
        result = processor.process_pdf(sample_pdf, show_progress=False)
        assert result.file_path == sample_pdf
        assert isinstance(result.text, str)

    def test_process_file_detects_image(self, processor, sample_image):
        with patch.object(processor, "process_image") as mock_process:
            mock_process.return_value = OCRResult(
                file_path=sample_image, pages=["content"], success=True, processing_time=0.1
            )
            processor.process_file(sample_image)
            mock_process.assert_called_once()

    def test_process_file_detects_pdf(self, processor, sample_pdf):
        with patch.object(processor, "process_pdf") as mock_process:
            mock_process.return_value = OCRResult(
                file_path=sample_pdf, pages=["content"], success=True, processing_time=0.1
            )
            processor.process_file(sample_pdf)
            mock_process.assert_called_once()

    def test_process_file_raises_for_unsupported(self, processor, tmp_path):
        unsupported = tmp_path / "test.xyz"
        unsupported.touch()
        with pytest.raises(ValueError, match="Unsupported file type"):
            processor.process_file(unsupported)

    def test_is_retryable_rate_limit(self):
        assert OCRProcessor._is_retryable(Exception("429 rate limit"))

    def test_is_retryable_server_error(self):
        assert OCRProcessor._is_retryable(TimeoutError("timed out"))

    def test_is_retryable_not_retryable(self):
        assert not OCRProcessor._is_retryable(ValueError("invalid input"))


class TestOCRProcessorNative:
    """Tests for native whole-PDF processing (the DEFAULT 'auto' path)."""

    @pytest.fixture
    def processor(self, mock_config, mock_genai_client):
        with patch("gemini_ocr.processor.genai") as mock_genai:
            mock_genai.Client.return_value = mock_genai_client
            processor = OCRProcessor(mock_config)
            processor.client = mock_genai_client
            return processor

    def test_native_is_default_single_call(self, processor, sample_pdf):
        # Default mode is "auto": whole-PDF = ONE Gemini call. A complete response
        # (one marker per actual page) must NOT trigger fallback => exactly 1 call.
        from tests.conftest import make_gemini_response

        processor.client.models.generate_content.return_value = make_gemini_response(
            "## Page 1\n\nFirst page text.\n\n## Page 2\n\nSecond page text."
        )
        result = processor.process_pdf(sample_pdf, show_progress=False)
        assert result.success
        assert result.mode == "whole_pdf"
        assert result.fell_back_from_whole_pdf is False
        assert processor.client.models.generate_content.call_count == 1
        # Uploaded once via the Files API, then deleted (no leak).
        assert processor.client.files.upload.call_count == 1
        assert processor.client.files.delete.call_count == 1

    def test_native_splits_response_on_page_markers(self, processor, sample_pdf):
        from tests.conftest import make_gemini_response

        processor.client.models.generate_content.return_value = make_gemini_response(
            "## Page 1\n\nAlpha content.\n\n## Page 2\n\nBeta content."
        )
        result = processor.process_pdf(sample_pdf, show_progress=False)
        assert result.success
        assert result.page_count == 2
        assert "Alpha content." in result.pages[0]
        assert "Beta content." in result.pages[1]
        # Marker lines are dropped (contract re-adds canonical headers).
        assert "## Page" not in result.pages[0]

    def test_whole_pdf_no_markers_falls_back_to_single_page(self, processor, sample_pdf):
        # In forced whole_pdf mode (no fallback), an unsplittable blob is kept as
        # one page rather than discarded.
        from tests.conftest import make_gemini_response

        processor.client.models.generate_content.return_value = make_gemini_response(
            "Just one undivided blob with no page markers."
        )
        result = processor.process_pdf(sample_pdf, show_progress=False, mode="whole_pdf")
        assert result.success
        assert result.page_count == 1
        assert result.mode == "whole_pdf"
        assert "undivided blob" in result.pages[0]
        # No fallback in forced whole_pdf mode: a single whole-PDF call only.
        assert processor.client.models.generate_content.call_count == 1

    def test_native_empty_response_is_failure(self, processor, sample_pdf):
        # An empty extraction raises in _extract_text -> caught -> FAILED.
        processor.client.models.generate_content.side_effect = RuntimeError(
            "Empty response: finish_reason=STOP"
        )
        result = processor.process_pdf(sample_pdf, show_progress=False, mode="whole_pdf")
        assert not result.success
        assert result.page_count == 0
        assert result.status.value == "failed"
        assert result.mode == "whole_pdf"

    def test_uses_per_page_when_config_mode_set(self, mock_config, mock_genai_client, sample_pdf):
        # config.pdf_mode="per_page" must route process_pdf to the per-page path.
        mock_config.pdf_mode = "per_page"
        with patch("gemini_ocr.processor.genai") as mock_genai:
            mock_genai.Client.return_value = mock_genai_client
            processor = OCRProcessor(mock_config)
            processor.client = mock_genai_client
        result = processor.process_pdf(sample_pdf, show_progress=False)
        # Per-page on the 2-page sample => 2 calls, no Files-API upload.
        assert result.page_count == 2
        assert result.mode == "per_page"
        assert processor.client.models.generate_content.call_count == 2
        assert processor.client.files.upload.call_count == 0


class TestOCRProcessorAutoFallback:
    """Tests for auto-mode truncation detection + per-page fallback."""

    @pytest.fixture
    def processor(self, mock_config, mock_genai_client):
        with patch("gemini_ocr.processor.genai") as mock_genai:
            mock_genai.Client.return_value = mock_genai_client
            processor = OCRProcessor(mock_config)
            processor.client = mock_genai_client
            return processor

    def test_fallback_on_max_tokens_finish_reason(self, processor, sample_pdf):
        # Whole-PDF returns a length-limited finish reason -> auto-fallback.
        from tests.conftest import make_gemini_response

        first = {"done": False}

        def gen(*args, **kwargs):
            if not first["done"]:
                first["done"] = True
                # Whole-PDF call: a complete-looking 2-page blob, but cut off.
                resp = make_gemini_response("## Page 1\n\nA\n\n## Page 2\n\nB (truncated")
                resp.candidates[0].finish_reason = "MAX_TOKENS"
                return resp
            # Subsequent per-page calls succeed.
            return make_gemini_response("per-page text")

        processor.client.models.generate_content.side_effect = gen
        result = processor.process_pdf(sample_pdf, show_progress=False)  # auto (default)
        assert result.success
        assert result.mode == "per_page"
        assert result.fell_back_from_whole_pdf is True
        # 1 whole-PDF call + 2 per-page calls = 3.
        assert processor.client.models.generate_content.call_count == 3

    def test_fallback_on_page_shortfall(self, processor, sample_pdf):
        # Whole-PDF returns markers for only 1 of the 2 pages -> auto-fallback.
        from tests.conftest import make_gemini_response

        first = {"done": False}

        def gen(*args, **kwargs):
            if not first["done"]:
                first["done"] = True
                # Only ONE page marker for a 2-page PDF (tail dropped), STOP finish.
                return make_gemini_response("## Page 1\n\nOnly the first page made it.")
            return make_gemini_response("per-page text")

        processor.client.models.generate_content.side_effect = gen
        result = processor.process_pdf(sample_pdf, show_progress=False)  # auto
        assert result.success
        assert result.mode == "per_page"
        assert result.fell_back_from_whole_pdf is True
        assert result.page_count == 2  # per-page recovered both pages
        assert processor.client.models.generate_content.call_count == 3

    def test_no_fallback_when_complete(self, processor, sample_pdf):
        # A complete whole-PDF response (all pages, STOP) -> no fallback, 1 call.
        from tests.conftest import make_gemini_response

        processor.client.models.generate_content.return_value = make_gemini_response(
            "## Page 1\n\nA\n\n## Page 2\n\nB"
        )
        result = processor.process_pdf(sample_pdf, show_progress=False)  # auto
        assert result.mode == "whole_pdf"
        assert result.fell_back_from_whole_pdf is False
        assert processor.client.models.generate_content.call_count == 1

    def test_whole_pdf_mode_does_not_fall_back(self, processor, sample_pdf):
        # Forced whole_pdf: even a truncated response is kept (no fallback).
        from tests.conftest import make_gemini_response

        resp = make_gemini_response("## Page 1\n\nonly one page")
        resp.candidates[0].finish_reason = "MAX_TOKENS"
        processor.client.models.generate_content.return_value = resp
        result = processor.process_pdf(sample_pdf, show_progress=False, mode="whole_pdf")
        assert result.mode == "whole_pdf"
        assert result.fell_back_from_whole_pdf is False
        assert processor.client.models.generate_content.call_count == 1


# NOTE: The truncation-signal logic (``is_truncated``) and the native-response
# page splitter (``split_native_pages``) now live in the shared
# ``ocr-output-contract`` package and are unit-tested there. gemini only tests how
# those helpers behave WHEN DRIVEN by its processor (native/auto-fallback tests
# below), not the helpers in isolation.


class TestOCRProcessorPerPage:
    """Tests for per-page PDF processing (opt-in via --per-page)."""

    @pytest.fixture
    def processor(self, mock_config, mock_genai_client):
        with patch("gemini_ocr.processor.genai") as mock_genai:
            mock_genai.Client.return_value = mock_genai_client
            processor = OCRProcessor(mock_config)
            processor.client = mock_genai_client
            return processor

    def test_pdf_produces_one_result_per_page(self, processor, sample_pdf):
        # sample_pdf fixture has 2 pages -> 2 Gemini calls, 2 page entries.
        result = processor.process_pdf(sample_pdf, show_progress=False, per_page=True)
        assert result.success
        assert result.page_count == 2
        assert processor.client.models.generate_content.call_count == 2

    def test_pdf_page_failure_is_tracked(self, processor, sample_pdf):
        from tests.conftest import make_gemini_response

        calls = {"n": 0}

        def gen(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ValueError("page 1 boom")
            return make_gemini_response("ok")

        processor.client.models.generate_content.side_effect = gen
        result = processor.process_pdf(sample_pdf, show_progress=False, per_page=True)
        assert not result.success
        assert 1 in result.page_errors
        assert result.status.value in ("partial", "failed")


class TestOCRProcessorSaveResults:
    """Tests for saving OCR results."""

    @pytest.fixture
    def processor(self, mock_config, mock_genai_client):
        with patch("gemini_ocr.processor.genai") as mock_genai:
            mock_genai.Client.return_value = mock_genai_client
            processor = OCRProcessor(mock_config)
            processor.client = mock_genai_client
            return processor

    def test_save_results_creates_per_document_folder(self, processor, tmp_path, sample_image):
        result = OCRResult(
            file_path=sample_image, pages=["Test content"], success=True, processing_time=1.5
        )
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        output_path = processor.save_results(result, output_dir, "sample.png")

        # Should be in per-document folder mirroring the relative key
        assert output_path.parent.name == "sample"
        assert output_path.name == "sample.md"
        assert output_path.exists()

    def test_save_results_clean_markdown_with_page_header(self, processor, tmp_path, sample_image):
        result = OCRResult(
            file_path=sample_image, pages=["Test content"], success=True, processing_time=1.5
        )
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        output_path = processor.save_results(result, output_dir, "sample.png")
        content = output_path.read_text()

        # Clean markdown — page header, no YAML frontmatter
        assert "## Page 1" in content
        assert "Test content" in content
        assert not content.lstrip().startswith("---")
        assert "OCR Results" not in content

    def test_save_results_handles_failure(self, processor, tmp_path, sample_image):
        result = OCRResult(
            file_path=sample_image,
            pages=[],
            success=False,
            error="API timeout",
            processing_time=1.0,
        )
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        output_path = processor.save_results(result, output_dir, "sample.png")
        content = output_path.read_text()
        # Failure marker present; raw error not leaked into the .md body.
        assert "OCR Failed" in content
        assert "API timeout" not in content

    def test_save_results_with_extracted_images(self, processor, tmp_path, sample_pdf):
        # A 1x1 PNG so PIL can open/re-encode it.
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (1, 1), "white").save(buf, format="PNG")
        png_bytes = buf.getvalue()

        result = OCRResult(
            file_path=sample_pdf,
            pages=["Content"],
            success=True,
            processing_time=1.0,
            extracted_images=[
                {"page": 1, "index": 1, "data": png_bytes, "ext": "png", "width": 1, "height": 1}
            ],
        )
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        md_path = processor.save_results(result, output_dir, "sample.pdf")

        figures_dir = output_dir / "sample" / "figures"
        assert figures_dir.exists()
        # Canonical figure naming + resolving link.
        assert (figures_dir / "figure_1_page1.png").exists()
        assert "figures/figure_1_page1.png" in md_path.read_text()


class TestOCRProcessorErrorHandling:
    """Tests for error handling in processor."""

    @pytest.fixture
    def processor(self, mock_config, mock_genai_client):
        with patch("gemini_ocr.processor.genai") as mock_genai:
            mock_genai.Client.return_value = mock_genai_client
            processor = OCRProcessor(mock_config)
            processor.client = mock_genai_client
            return processor

    def test_api_error_handling(self, processor, sample_image):
        processor.client.models.generate_content.side_effect = ValueError("API Error")
        result = processor.process_image(sample_image)
        assert result.success is False
        assert result.error is not None

    def test_file_size_validation(self, processor, mock_config, tmp_path):
        mock_config.max_file_size_mb = 0.0001
        large_file = tmp_path / "large.pdf"
        large_file.write_bytes(b"x" * 10000)
        with pytest.raises(ValueError, match="exceeds maximum"):
            processor.process_file(large_file)
