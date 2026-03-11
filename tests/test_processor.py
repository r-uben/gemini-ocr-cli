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
            text="Extracted content",
            success=True,
            processing_time=1.0,
        )
        assert result.success is True

    def test_success_false(self):
        result = OCRResult(
            file_path=Path("test.pdf"),
            text="",
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
            env = {k: v for k, v in os.environ.items()
                   if k not in ("GEMINI_API_KEY", "GOOGLE_API_KEY")}
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
                file_path=sample_image, text="content", success=True, processing_time=0.1
            )
            processor.process_file(sample_image)
            mock_process.assert_called_once()

    def test_process_file_detects_pdf(self, processor, sample_pdf):
        with patch.object(processor, "process_pdf") as mock_process:
            mock_process.return_value = OCRResult(
                file_path=sample_pdf, text="content", success=True, processing_time=0.1
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


class TestOCRProcessorUpload:
    """Tests for file upload functionality."""

    @pytest.fixture
    def processor(self, mock_config, mock_genai_client):
        with patch("gemini_ocr.processor.genai") as mock_genai:
            mock_genai.Client.return_value = mock_genai_client
            processor = OCRProcessor(mock_config)
            processor.client = mock_genai_client
            return processor

    def test_upload_file(self, processor, sample_pdf):
        processor.client.files.upload.return_value.state = "ACTIVE"
        uploaded = processor._upload_file(sample_pdf)
        processor.client.files.upload.assert_called_once()
        assert uploaded is not None

    def test_upload_file_waits_for_processing(self, processor, sample_pdf):
        mock_file = MagicMock()
        mock_file.state = "PROCESSING"
        processor.client.files.upload.return_value = mock_file
        active_file = MagicMock()
        active_file.state = "ACTIVE"
        processor.client.files.get.return_value = active_file
        processor._upload_file(sample_pdf)
        processor.client.files.get.assert_called()

    def test_upload_file_raises_on_failure(self, processor, sample_pdf):
        mock_file = MagicMock()
        mock_file.state = "FAILED"
        mock_file.name = "test-file"
        processor.client.files.upload.return_value = mock_file
        with pytest.raises(RuntimeError, match="File upload failed"):
            processor._upload_file(sample_pdf)


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
            file_path=sample_image, text="Test content", success=True, processing_time=1.5
        )
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        output_path = processor.save_results(result, output_dir)

        # Should be in per-document folder
        assert output_path.parent.name == "sample"
        assert output_path.name == "sample.md"
        assert output_path.exists()

    def test_save_results_clean_markdown(self, processor, tmp_path, sample_image):
        result = OCRResult(
            file_path=sample_image, text="Test content", success=True, processing_time=1.5
        )
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        output_path = processor.save_results(result, output_dir)
        content = output_path.read_text()

        # Clean markdown — no headers, no metadata
        assert content == "Test content"
        assert "OCR Results" not in content
        assert "Processing Time" not in content

    def test_save_results_handles_failure(self, processor, tmp_path, sample_image):
        result = OCRResult(
            file_path=sample_image, text="", success=False, error="API timeout", processing_time=1.0
        )
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        output_path = processor.save_results(result, output_dir)
        content = output_path.read_text()
        assert "OCR Failed" in content
        assert "API timeout" in content

    def test_save_results_with_extracted_images(self, processor, tmp_path, sample_pdf):
        result = OCRResult(
            file_path=sample_pdf,
            text="Content",
            success=True,
            processing_time=1.0,
            extracted_images=[
                {"page": 1, "index": 1, "data": b"\x89PNG\r\n", "ext": "png", "width": 100, "height": 100}
            ],
        )
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        processor.save_results(result, output_dir)

        figures_dir = output_dir / "sample" / "figures"
        assert figures_dir.exists()
        assert (figures_dir / "page1_img1.png").exists()


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
