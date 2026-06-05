"""Pytest configuration and fixtures."""

import os
from pathlib import Path
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
    config.addinivalue_line("markers", "integration: mark test as requiring GEMINI_API_KEY")


@pytest.fixture(autouse=True)
def _reset_console_quiet():
    """Reset the module-level rich Consoles' ``quiet`` state around each test.

    ``gemini_ocr.cli`` and ``gemini_ocr.processor`` hold module-level
    ``Console()`` singletons, and ``--quiet`` flips ``console.quiet = True`` on
    them. Without restoring that, a test exercising ``--quiet`` silences every
    later test's captured output. This guard keeps CLI tests order-independent.
    """
    from gemini_ocr import cli as _cli
    from gemini_ocr import processor as _proc

    saved = (_cli.console.quiet, _proc.console.quiet)
    try:
        yield
    finally:
        _cli.console.quiet, _proc.console.quiet = saved


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
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

    page1 = doc.new_page(width=612, height=792)
    page1.insert_text((72, 72), "Page 1: Test Document", fontsize=24)
    page1.insert_text((72, 120), "This is sample text for testing OCR.", fontsize=12)

    page2 = doc.new_page(width=612, height=792)
    page2.insert_text((72, 72), "Page 2: More Content", fontsize=24)
    page2.insert_text((72, 120), "Additional text on the second page.", fontsize=12)

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
        config.model = "gemini-3-flash-preview"
        config.verbose = False
        config.quiet = False
        config.max_workers = 1
        config.max_retries = 3
        config.retry_base_delay = 0.01
        yield config


def make_gemini_response(text: str = "Extracted text from document") -> MagicMock:
    """Build a GenerateContentResponse mock whose parts yield `text`.

    The processor extracts text by walking ``candidates[0].content.parts`` (the
    ``.text`` shortcut is unreliable for thinking models), so mocks must supply a
    real parts structure rather than only ``response.text``.
    """
    part = MagicMock()
    part.text = text
    part.thought = False
    content = MagicMock()
    content.parts = [part]
    candidate = MagicMock()
    candidate.content = content
    candidate.finish_reason = "STOP"
    resp = MagicMock()
    resp.candidates = [candidate]
    return resp


@pytest.fixture
def mock_genai_client():
    """Create a mock Gemini client."""
    mock_client = MagicMock()

    # Mock file upload
    mock_file = MagicMock()
    mock_file.name = "files/test-file-id"
    mock_file.state = "ACTIVE"
    mock_client.files.upload.return_value = mock_file

    # Mock generate_content with a walkable parts structure
    mock_client.models.generate_content.return_value = make_gemini_response()

    return mock_client


@pytest.fixture
def api_key_available() -> bool:
    """Check if GEMINI_API_KEY is available for integration tests."""
    return bool(os.environ.get("GEMINI_API_KEY"))


def skip_without_api_key(func):
    """Decorator to skip integration tests without API key."""
    return pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set"
    )(pytest.mark.integration(func))
