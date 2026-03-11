"""Basic import tests."""


def test_import_package():
    import gemini_ocr

    assert hasattr(gemini_ocr, "__version__")


def test_import_processor():
    from gemini_ocr import OCRProcessor

    assert OCRProcessor is not None


def test_import_config():
    from gemini_ocr import Config

    assert Config is not None


def test_import_cli():
    from gemini_ocr.cli import cli

    assert cli is not None


def test_import_metadata():
    from gemini_ocr.metadata import MetadataManager

    assert MetadataManager is not None
