"""Integration tests requiring a real GEMINI_API_KEY.

Run with: pytest tests/test_integration.py -v

These tests are skipped if GEMINI_API_KEY is not set.
"""

import os
import shutil
from pathlib import Path

import pytest

from tests.conftest import skip_without_api_key


@pytest.fixture
def real_config():
    from gemini_ocr.config import Config

    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set")
    return Config.from_env()


@pytest.fixture
def real_processor(real_config):
    from gemini_ocr.processor import OCRProcessor

    return OCRProcessor(real_config)


@skip_without_api_key
class TestIntegrationPDF:
    def test_process_pdf_real(self, real_processor, sample_pdf):
        result = real_processor.process_pdf(sample_pdf, show_progress=False)
        assert result.success, f"Failed: {result.error}"
        assert len(result.text) > 0

    def test_process_pdf_extract_task(self, real_processor, sample_pdf):
        result = real_processor.process_pdf(sample_pdf, task="extract", show_progress=False)
        assert result.success, f"Failed: {result.error}"
        assert len(result.text) > 0


@skip_without_api_key
class TestIntegrationImage:
    def test_process_image_real(self, real_processor, sample_image):
        result = real_processor.process_image(sample_image)
        assert result.success, f"Failed: {result.error}"
        assert result.processing_time > 0


@skip_without_api_key
class TestIntegrationEndToEnd:
    def test_full_process_workflow(self, real_processor, sample_pdf, tmp_path):
        output_dir = tmp_path / "output"
        result = real_processor.process_pdf(sample_pdf, show_progress=False)
        assert result.success

        output_dir.mkdir()
        output_path = real_processor.save_results(result, output_dir)

        assert output_path.exists()
        content = output_path.read_text()
        # Clean markdown — just the OCR text
        assert result.text in content

    def test_batch_processing(self, real_processor, sample_pdf, sample_image, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        shutil.copy(sample_pdf, input_dir / "test.pdf")
        shutil.copy(sample_image, input_dir / "test.png")

        results = []
        for file_path in input_dir.iterdir():
            result = real_processor.process_file(file_path, show_progress=False)
            results.append(result)

        success_count = sum(1 for r in results if r.success)
        assert success_count >= 1
