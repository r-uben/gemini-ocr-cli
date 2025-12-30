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
        """Test success property when True."""
        result = OCRResult(
            file_path=Path("test.pdf"),
            text="Extracted content",
            success=True,
            processing_time=1.0,
        )
        assert result.success is True

    def test_success_false(self):
        """Test success property when False."""
        result = OCRResult(
            file_path=Path("test.pdf"),
            text="",
            success=False,
            error="Processing failed",
            processing_time=1.0,
        )
        assert result.success is False

    def test_total_pages_estimate(self):
        """Test total_pages estimation."""
        # ~3000 chars per page estimate
        result = OCRResult(
            file_path=Path("test.pdf"),
            text="x" * 9000,  # ~3 pages
            success=True,
            processing_time=1.0,
        )
        assert result.total_pages == 3

    def test_total_pages_minimum_one(self):
        """Test total_pages returns at least 1."""
        result = OCRResult(
            file_path=Path("test.pdf"),
            text="short",
            success=True,
            processing_time=1.0,
        )
        assert result.total_pages == 1

    def test_total_pages_empty_text(self):
        """Test total_pages with empty text."""
        result = OCRResult(
            file_path=Path("test.pdf"),
            text="",
            success=False,
            processing_time=1.0,
        )
        assert result.total_pages == 0


class TestOCRProcessor:
    """Tests for OCRProcessor class."""

    @pytest.fixture
    def processor(self, mock_config, mock_genai_client):
        """Create a processor with mocked dependencies."""
        with patch("gemini_ocr.processor.genai") as mock_genai:
            mock_genai.Client.return_value = mock_genai_client
            processor = OCRProcessor(mock_config)
            processor.client = mock_genai_client
            return processor

    def test_init_creates_client(self, mock_config):
        """Test processor initialization creates Gemini client."""
        with patch("gemini_ocr.processor.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client

            processor = OCRProcessor(mock_config)

            mock_genai.Client.assert_called_once_with(api_key=mock_config.api_key)

    def test_init_raises_without_api_key(self):
        """Test initialization raises without API key."""
        with patch.dict(os.environ, {}, clear=True):
            env = {k: v for k, v in os.environ.items()
                   if k not in ("GEMINI_API_KEY", "GOOGLE_API_KEY")}
            with patch.dict(os.environ, env, clear=True):
                config = Config()
                config.api_key = ""

                with pytest.raises(ValueError, match="API key not set"):
                    OCRProcessor(config)

    def test_process_image(self, processor, sample_image):
        """Test processing a single image."""
        result = processor.process_image(sample_image)

        assert result.success
        assert "Extracted text" in result.text

    def test_process_image_with_custom_prompt(self, processor, sample_image):
        """Test processing with custom prompt."""
        custom_prompt = "Extract only numbers"

        result = processor.process_image(sample_image, custom_prompt=custom_prompt)

        # Verify the API was called
        assert processor.client.models.generate_content.called

    def test_process_pdf(self, processor, sample_pdf):
        """Test processing a PDF file."""
        result = processor.process_pdf(sample_pdf, show_progress=False)

        assert result.file_path == sample_pdf
        # Native PDF upload returns a single result
        assert isinstance(result.text, str)

    def test_process_file_detects_image(self, processor, sample_image):
        """Test process_file routes images correctly."""
        with patch.object(processor, "process_image") as mock_process:
            mock_process.return_value = OCRResult(
                file_path=sample_image,
                text="content",
                success=True,
                processing_time=0.1,
            )

            processor.process_file(sample_image)

            mock_process.assert_called_once()

    def test_process_file_detects_pdf(self, processor, sample_pdf):
        """Test process_file routes PDFs correctly."""
        with patch.object(processor, "process_pdf") as mock_process:
            mock_process.return_value = OCRResult(
                file_path=sample_pdf,
                text="content",
                success=True,
                processing_time=0.1,
            )

            processor.process_file(sample_pdf)

            mock_process.assert_called_once()

    def test_process_file_raises_for_unsupported(self, processor, tmp_path):
        """Test process_file raises for unsupported file types."""
        unsupported = tmp_path / "test.xyz"
        unsupported.touch()

        with pytest.raises(ValueError, match="Unsupported file type"):
            processor.process_file(unsupported)

    def test_describe_figure(self, processor, sample_image):
        """Test figure description."""
        description = processor.describe_figure(sample_image)

        assert isinstance(description, str)
        processor.client.models.generate_content.assert_called()


class TestOCRProcessorUpload:
    """Tests for file upload functionality."""

    @pytest.fixture
    def processor(self, mock_config, mock_genai_client):
        """Create a processor with mocked dependencies."""
        with patch("gemini_ocr.processor.genai") as mock_genai:
            mock_genai.Client.return_value = mock_genai_client
            processor = OCRProcessor(mock_config)
            processor.client = mock_genai_client
            return processor

    def test_upload_file(self, processor, sample_pdf):
        """Test file upload to Gemini API."""
        # Mock file state as ACTIVE (already processed)
        processor.client.files.upload.return_value.state = "ACTIVE"

        uploaded = processor._upload_file(sample_pdf)

        processor.client.files.upload.assert_called_once()
        assert uploaded is not None

    def test_upload_file_waits_for_processing(self, processor, sample_pdf):
        """Test upload waits for file processing."""
        mock_file = MagicMock()
        # First call returns PROCESSING, second returns ACTIVE
        mock_file.state = "PROCESSING"
        processor.client.files.upload.return_value = mock_file

        # Mock get to return ACTIVE state
        active_file = MagicMock()
        active_file.state = "ACTIVE"
        processor.client.files.get.return_value = active_file

        uploaded = processor._upload_file(sample_pdf)

        # Should have called get to check status
        processor.client.files.get.assert_called()

    def test_upload_file_raises_on_failure(self, processor, sample_pdf):
        """Test upload raises on processing failure."""
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
        """Create a processor with mocked dependencies."""
        with patch("gemini_ocr.processor.genai") as mock_genai:
            mock_genai.Client.return_value = mock_genai_client
            processor = OCRProcessor(mock_config)
            processor.client = mock_genai_client
            return processor

    def test_save_results_creates_markdown(self, processor, tmp_path, sample_image):
        """Test save_results creates markdown file."""
        result = OCRResult(
            file_path=sample_image,
            text="Test content",
            success=True,
            processing_time=1.5,
        )

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        output_path = processor.save_results(result, output_dir)

        assert output_path.exists()
        assert output_path.suffix == ".md"

        content = output_path.read_text()
        assert "Test content" in content
        assert "OCR Results" in content

    def test_save_results_includes_metadata(self, processor, tmp_path, sample_image):
        """Test save_results includes processing metadata."""
        result = OCRResult(
            file_path=sample_image,
            text="Content",
            success=True,
            processing_time=2.5,
        )

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        output_path = processor.save_results(result, output_dir)
        content = output_path.read_text()

        assert "Processing Time" in content
        assert "2.5" in content

    def test_save_results_handles_failure(self, processor, tmp_path, sample_image):
        """Test save_results handles failed results."""
        result = OCRResult(
            file_path=sample_image,
            text="",
            success=False,
            error="API timeout",
            processing_time=1.0,
        )

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        output_path = processor.save_results(result, output_dir)
        content = output_path.read_text()

        assert "OCR Failed" in content
        assert "API timeout" in content


class TestOCRProcessorErrorHandling:
    """Tests for error handling in processor."""

    @pytest.fixture
    def processor(self, mock_config, mock_genai_client):
        """Create a processor with mocked dependencies."""
        with patch("gemini_ocr.processor.genai") as mock_genai:
            mock_genai.Client.return_value = mock_genai_client
            processor = OCRProcessor(mock_config)
            processor.client = mock_genai_client
            return processor

    def test_api_error_handling(self, processor, sample_image):
        """Test handling of API errors."""
        processor.client.models.generate_content.side_effect = Exception("API Error")

        result = processor.process_image(sample_image)

        assert result.success is False
        # Error is wrapped by retry logic
        assert result.error is not None
        assert "failed" in result.error.lower() or "error" in result.error.lower()

    def test_file_size_validation(self, processor, mock_config, tmp_path):
        """Test file size validation."""
        mock_config.max_file_size_mb = 0.0001  # Very small

        large_file = tmp_path / "large.pdf"
        large_file.write_bytes(b"x" * 10000)

        with pytest.raises(ValueError, match="exceeds maximum"):
            processor.process_file(large_file)


class TestRetryLogic:
    """Tests for retry functionality."""

    def test_retry_decorator_import(self):
        """Test retry decorator can be imported."""
        from gemini_ocr.retry import retry, RetryError
        assert callable(retry)
        assert issubclass(RetryError, Exception)

    def test_is_retryable_error(self):
        """Test retryable error detection."""
        from gemini_ocr.retry import is_retryable_error

        assert is_retryable_error(Exception("rate limit exceeded"))
        assert is_retryable_error(Exception("429 Too Many Requests"))
        assert is_retryable_error(Exception("500 Internal Server Error"))
        assert is_retryable_error(Exception("connection timeout"))
        assert not is_retryable_error(Exception("invalid input"))
