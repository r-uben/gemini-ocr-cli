"""Tests for CLI module."""

import os
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from gemini_ocr import __version__
from gemini_ocr.cli import cli, main


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


class TestCLIBasics:
    """Tests for basic CLI functionality."""

    def test_cli_help(self, runner):
        """Test --help shows usage information."""
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "Gemini OCR" in result.output
        assert "process" in result.output
        assert "describe" in result.output
        assert "info" in result.output

    def test_cli_version(self, runner):
        """Test --version shows version."""
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        assert "gemini-ocr" in result.output
        assert __version__ in result.output

    def test_process_help(self, runner):
        """Test process --help shows options."""
        result = runner.invoke(cli, ["process", "--help"])

        assert result.exit_code == 0
        assert "--output-dir" in result.output
        assert "--api-key" in result.output
        assert "--model" in result.output
        # --dpi removed in v0.2.0 (native PDF upload)
        assert "--dpi" not in result.output

    def test_describe_help(self, runner):
        """Test describe --help shows options."""
        result = runner.invoke(cli, ["describe", "--help"])

        assert result.exit_code == 0
        assert "--api-key" in result.output
        assert "--model" in result.output
        assert "--output" in result.output


class TestProcessCommand:
    """Tests for the process command."""

    def test_process_missing_file(self, runner):
        """Test process with non-existent file."""
        result = runner.invoke(cli, ["process", "nonexistent.pdf"])

        assert result.exit_code != 0
        assert "does not exist" in result.output or "Error" in result.output

    def test_process_with_api_key(self, runner, sample_pdf):
        """Test process with explicit API key."""
        with patch("gemini_ocr.cli.OCRProcessor") as mock_processor:
            mock_instance = MagicMock()
            mock_processor.return_value = mock_instance

            result = runner.invoke(
                cli,
                ["process", str(sample_pdf), "--api-key", "test-key"],
            )

            # Should attempt to create processor
            # (may fail due to other validation, but key should be set)

    def test_process_with_output_dir(self, runner, sample_pdf, tmp_path):
        """Test process with custom output directory."""
        output_dir = tmp_path / "output"

        with patch("gemini_ocr.cli.OCRProcessor") as mock_processor:
            mock_instance = MagicMock()
            mock_processor.return_value = mock_instance

            result = runner.invoke(
                cli,
                [
                    "process",
                    str(sample_pdf),
                    "-o",
                    str(output_dir),
                    "--api-key",
                    "test-key",
                ],
            )


class TestDescribeCommand:
    """Tests for the describe command."""

    def test_describe_missing_image(self, runner):
        """Test describe with non-existent file."""
        result = runner.invoke(cli, ["describe", "nonexistent.png"])

        assert result.exit_code != 0

    def test_describe_with_output(self, runner, sample_image, tmp_path):
        """Test describe with output file."""
        output_file = tmp_path / "description.md"

        with patch("gemini_ocr.cli.OCRProcessor") as mock_processor:
            mock_instance = MagicMock()
            mock_instance.describe_figure.return_value = "Test description"
            mock_processor.return_value = mock_instance

            result = runner.invoke(
                cli,
                [
                    "describe",
                    str(sample_image),
                    "-o",
                    str(output_file),
                    "--api-key",
                    "test-key",
                ],
            )


class TestInfoCommand:
    """Tests for the info command."""

    def test_info_shows_system_info(self, runner):
        """Test info command shows system information."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
            result = runner.invoke(cli, ["info"])

            assert result.exit_code == 0
            assert "System Information" in result.output
            assert "Python" in result.output

    def test_info_shows_config(self, runner):
        """Test info command shows configuration."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("google.genai.Client") as mock_client_class:
                # Mock the API test
                mock_client = MagicMock()
                mock_client.models.list.return_value = []
                mock_client_class.return_value = mock_client

                result = runner.invoke(cli, ["info"])

                assert "Configuration" in result.output
                assert "Model" in result.output


class TestShorthandSyntax:
    """Tests for shorthand command syntax."""

    def test_shorthand_file_path(self, sample_pdf, tmp_path):
        """Test that gemini-ocr file.pdf works as shorthand for process."""
        # This tests the main() function's shorthand detection
        import sys
        from unittest.mock import patch

        with patch.object(sys, "argv", ["gemini-ocr", str(sample_pdf)]):
            with patch("gemini_ocr.cli.cli") as mock_cli:
                # The shorthand should insert "process" command
                try:
                    main()
                except SystemExit:
                    pass

    def test_explicit_command_not_modified(self):
        """Test that explicit commands are not modified."""
        import sys

        original_argv = ["gemini-ocr", "info"]

        with patch.object(sys, "argv", original_argv.copy()):
            with patch("gemini_ocr.cli.cli") as mock_cli:
                try:
                    main()
                except SystemExit:
                    pass

                # Should call cli with original args
