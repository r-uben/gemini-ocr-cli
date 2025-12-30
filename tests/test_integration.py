"""Integration tests requiring a real GEMINI_API_KEY.

Run with: pytest tests/test_integration.py -v

These tests are skipped if GEMINI_API_KEY is not set.
"""

import os
from pathlib import Path

import pytest

from tests.conftest import skip_without_api_key


@pytest.fixture
def real_config():
    """Create a real config using actual API key."""
    from gemini_ocr.config import Config

    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set")

    return Config.from_env()


@pytest.fixture
def real_processor(real_config):
    """Create a real processor with actual API access."""
    from gemini_ocr.processor import OCRProcessor

    return OCRProcessor(real_config)


@skip_without_api_key
class TestIntegrationPDF:
    """Integration tests for PDF processing."""

    def test_process_pdf_real(self, real_processor, sample_pdf):
        """Test processing a real PDF with actual API."""
        result = real_processor.process_pdf(sample_pdf, show_progress=False)

        assert result.success, f"Failed: {result.error}"
        assert len(result.text) > 0
        # Check for expected content from sample PDF
        assert "Page" in result.text or "Test" in result.text

    def test_process_pdf_extract_task(self, real_processor, sample_pdf):
        """Test extract task on PDF."""
        result = real_processor.process_pdf(
            sample_pdf, task="extract", show_progress=False
        )

        assert result.success, f"Failed: {result.error}"
        assert len(result.text) > 0


@skip_without_api_key
class TestIntegrationImage:
    """Integration tests for image processing."""

    def test_process_image_real(self, real_processor, sample_image):
        """Test processing a real image with actual API."""
        result = real_processor.process_image(sample_image)

        assert result.success, f"Failed: {result.error}"
        # Image might have no text (white image), so just check it ran
        assert result.processing_time > 0

    def test_describe_figure_real(self, real_processor, sample_image):
        """Test figure description with actual API."""
        description = real_processor.describe_figure(sample_image)

        assert len(description) > 0
        # Should describe it as some kind of image/visualization
        assert any(
            word in description.lower()
            for word in ["image", "white", "blank", "simple", "plain"]
        )


@skip_without_api_key
class TestIntegrationEndToEnd:
    """End-to-end integration tests."""

    def test_full_process_workflow(self, real_processor, sample_pdf, tmp_path):
        """Test the full process workflow."""
        output_dir = tmp_path / "output"

        # Process the PDF
        result = real_processor.process_pdf(sample_pdf, show_progress=False)
        assert result.success

        # Save results
        output_dir.mkdir()
        output_path = real_processor.save_results(result, output_dir)

        # Verify output
        assert output_path.exists()
        content = output_path.read_text()
        assert "OCR Results" in content
        assert result.text in content

    def test_batch_processing(self, real_processor, sample_pdf, sample_image, tmp_path):
        """Test processing multiple files."""
        # Create input directory with files
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        import shutil
        shutil.copy(sample_pdf, input_dir / "test.pdf")
        shutil.copy(sample_image, input_dir / "test.png")

        # Process each file
        results = []
        for file_path in input_dir.iterdir():
            result = real_processor.process_file(file_path, show_progress=False)
            results.append(result)

        # Verify all processed
        success_count = sum(1 for r in results if r.success)
        assert success_count >= 1  # At least PDF should succeed


@skip_without_api_key
class TestIntegrationErrorHandling:
    """Integration tests for error handling."""

    def test_invalid_file_handling(self, real_processor, tmp_path):
        """Test handling of invalid files."""
        # Create a fake PDF (invalid content)
        fake_pdf = tmp_path / "fake.pdf"
        fake_pdf.write_text("This is not a real PDF")

        result = real_processor.process_pdf(fake_pdf, show_progress=False)

        # Should fail gracefully
        assert result.success is False
        assert result.error is not None

    def test_large_file_rejection(self, real_processor, tmp_path):
        """Test rejection of files exceeding size limit."""
        real_processor.config.max_file_size_mb = 0.0001  # Very small

        large_file = tmp_path / "large.pdf"
        large_file.write_bytes(b"x" * 1000)

        with pytest.raises(ValueError, match="exceeds maximum"):
            real_processor.process_file(large_file)
