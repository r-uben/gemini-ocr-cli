"""Basic import tests."""

import pytest


def test_import_package():
    """Test that package can be imported."""
    import gemini_ocr
    assert hasattr(gemini_ocr, "__version__")


def test_import_processor():
    """Test that processor can be imported."""
    from gemini_ocr import OCRProcessor
    assert OCRProcessor is not None


def test_import_config():
    """Test that config can be imported."""
    from gemini_ocr import Config
    assert Config is not None


def test_import_cli():
    """Test that CLI can be imported."""
    from gemini_ocr.cli import cli
    assert cli is not None
