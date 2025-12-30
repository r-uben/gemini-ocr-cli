"""Tests for utility functions."""

from pathlib import Path

import pytest

from gemini_ocr.utils import (
    determine_output_path,
    format_file_size,
    get_supported_files,
    is_image_file,
    is_pdf_file,
    is_supported_file,
    sanitize_filename,
)


class TestFileTypeDetection:
    """Tests for file type detection functions."""

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("test.pdf", True),
            ("test.PDF", True),
            ("test.jpg", True),
            ("test.jpeg", True),
            ("test.png", True),
            ("test.webp", True),
            ("test.gif", True),
            ("test.bmp", True),
            ("test.tiff", True),
            ("test.tif", True),
            ("test.txt", False),
            ("test.docx", False),
            ("test", False),
        ],
    )
    def test_is_supported_file(self, filename: str, expected: bool):
        """Test supported file detection."""
        assert is_supported_file(Path(filename)) == expected

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("test.jpg", True),
            ("test.jpeg", True),
            ("test.png", True),
            ("test.PNG", True),
            ("test.webp", True),
            ("test.gif", True),
            ("test.pdf", False),
            ("test.txt", False),
        ],
    )
    def test_is_image_file(self, filename: str, expected: bool):
        """Test image file detection."""
        assert is_image_file(Path(filename)) == expected

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("test.pdf", True),
            ("test.PDF", True),
            ("document.pdf", True),
            ("test.jpg", False),
            ("test.txt", False),
        ],
    )
    def test_is_pdf_file(self, filename: str, expected: bool):
        """Test PDF file detection."""
        assert is_pdf_file(Path(filename)) == expected


class TestSanitizeFilename:
    """Tests for filename sanitization."""

    @pytest.mark.parametrize(
        "input_name,expected",
        [
            ("normal_file", "normal_file"),
            ("file with spaces", "file_with_spaces"),
            ("file<>:\"/\\|?*name", "file_name"),
            ("multiple   spaces", "multiple_spaces"),
            ("___leading_trailing___", "leading_trailing"),
            ("", "unnamed"),
        ],
    )
    def test_sanitize_filename(self, input_name: str, expected: str):
        """Test filename sanitization."""
        assert sanitize_filename(input_name) == expected

    def test_sanitize_filename_max_length(self):
        """Test filename truncation to max length."""
        long_name = "a" * 300
        result = sanitize_filename(long_name, max_length=200)
        assert len(result) == 200

    def test_sanitize_filename_no_max_length(self):
        """Test filename without max length restriction."""
        long_name = "a" * 300
        result = sanitize_filename(long_name, max_length=None)
        assert len(result) == 300


class TestFormatFileSize:
    """Tests for file size formatting."""

    @pytest.mark.parametrize(
        "size_bytes,expected",
        [
            (0, "0.0 B"),
            (500, "500.0 B"),
            (1024, "1.0 KB"),
            (1536, "1.5 KB"),
            (1048576, "1.0 MB"),
            (1073741824, "1.0 GB"),
            (1099511627776, "1.0 TB"),
        ],
    )
    def test_format_file_size(self, size_bytes: int, expected: str):
        """Test file size formatting."""
        assert format_file_size(size_bytes) == expected


class TestDetermineOutputPath:
    """Tests for output path determination."""

    def test_output_path_for_file(self, tmp_path):
        """Test output path when input is a file."""
        input_file = tmp_path / "test.pdf"
        input_file.touch()

        result = determine_output_path(input_file)

        assert result == tmp_path / "gemini_ocr_output"
        assert result.exists()

    def test_output_path_for_directory(self, tmp_path):
        """Test output path when input is a directory."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        result = determine_output_path(input_dir)

        assert result == input_dir / "gemini_ocr_output"
        assert result.exists()

    def test_output_path_custom(self, tmp_path):
        """Test custom output path."""
        input_file = tmp_path / "test.pdf"
        input_file.touch()
        custom_output = tmp_path / "custom_output"

        result = determine_output_path(input_file, output_path=custom_output)

        assert result == custom_output
        assert result.exists()

    def test_output_path_with_timestamp(self, tmp_path):
        """Test output path with timestamp."""
        input_file = tmp_path / "test.pdf"
        input_file.touch()

        result = determine_output_path(input_file, add_timestamp=True)

        # Should contain timestamp pattern
        assert "gemini_ocr_output_" in str(result)


class TestGetSupportedFiles:
    """Tests for finding supported files."""

    def test_get_supported_files_recursive(self, tmp_path):
        """Test recursive file discovery."""
        # Create structure
        (tmp_path / "root.pdf").touch()
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "nested.pdf").touch()
        (tmp_path / "sub" / "image.png").touch()
        (tmp_path / "ignored.txt").touch()

        files = get_supported_files(tmp_path, recursive=True)

        assert len(files) == 3
        names = [f.name for f in files]
        assert "root.pdf" in names
        assert "nested.pdf" in names
        assert "image.png" in names
        assert "ignored.txt" not in names

    def test_get_supported_files_non_recursive(self, tmp_path):
        """Test non-recursive file discovery."""
        (tmp_path / "root.pdf").touch()
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "nested.pdf").touch()

        files = get_supported_files(tmp_path, recursive=False)

        assert len(files) == 1
        assert files[0].name == "root.pdf"

    def test_get_supported_files_empty_directory(self, tmp_path):
        """Test empty directory returns empty list."""
        files = get_supported_files(tmp_path)
        assert files == []

    def test_get_supported_files_sorted(self, tmp_path):
        """Test files are returned sorted."""
        (tmp_path / "c.pdf").touch()
        (tmp_path / "a.pdf").touch()
        (tmp_path / "b.pdf").touch()

        files = get_supported_files(tmp_path)

        names = [f.name for f in files]
        assert names == ["a.pdf", "b.pdf", "c.pdf"]
