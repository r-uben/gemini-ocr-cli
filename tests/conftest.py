"""Pytest configuration and fixtures."""

import os
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

# Mark all tests as unit by default
def pytest_collection_modifyitems(items):
    for item in items:
        if "integration" not in item.keywords:
            item.add_marker(pytest.mark.unit)


# Skip integration tests if no API key
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: mark test as requiring GEMINI_API_KEY"
    )


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_image(fixtures_dir: Path, tmp_path: Path) -> Path:
    """Create a sample test image."""
    img_path = tmp_path / "sample.png"
    img = Image.new("RGB", (100, 100), color="white")
    img.save(img_path)
    return img_path


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Create a sample 2-page PDF using PyMuPDF."""
    import fitz

    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()

    # Page 1
    page1 = doc.new_page(width=612, height=792)
    page1.insert_text((72, 72), "Page 1: Test Document", fontsize=24)
    page1.insert_text((72, 120), "This is sample text for testing OCR.", fontsize=12)

    # Page 2
    page2 = doc.new_page(width=612, height=792)
    page2.insert_text((72, 72), "Page 2: More Content", fontsize=24)
    page2.insert_text((72, 120), "Additional text on the second page.", fontsize=12)

    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def sample_pdf_with_table(tmp_path: Path) -> Path:
    """Create a PDF with a simple table."""
    import fitz

    pdf_path = tmp_path / "table.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)

    page.insert_text((72, 72), "Table Example", fontsize=18)

    # Draw simple table
    table_text = """
| Column A | Column B | Column C |
|----------|----------|----------|
| Row 1A   | Row 1B   | Row 1C   |
| Row 2A   | Row 2B   | Row 2C   |
"""
    page.insert_text((72, 120), table_text, fontsize=10)

    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def mock_config():
    """Create a mock config with test values."""
    from gemini_ocr.config import Config

    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-api-key"}):
        config = Config()
        config.api_key = "test-api-key"
        config.model = "gemini-3.0-flash"
        config.verbose = False
        yield config


@pytest.fixture
def mock_genai_client():
    """Create a mock Gemini client."""
    mock_client = MagicMock()

    # Mock file upload
    mock_file = MagicMock()
    mock_file.name = "files/test-file-id"
    mock_file.state = "ACTIVE"
    mock_client.files.upload.return_value = mock_file

    # Mock generate_content
    mock_response = MagicMock()
    mock_response.text = "Extracted text from document"
    mock_client.models.generate_content.return_value = mock_response

    return mock_client


@pytest.fixture
def api_key_available() -> bool:
    """Check if GEMINI_API_KEY is available for integration tests."""
    return bool(os.environ.get("GEMINI_API_KEY"))


def skip_without_api_key(func):
    """Decorator to skip integration tests without API key."""
    return pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY not set"
    )(pytest.mark.integration(func))
