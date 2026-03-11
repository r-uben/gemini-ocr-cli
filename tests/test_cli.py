"""Tests for CLI module."""

import os
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from gemini_ocr import __version__
from gemini_ocr.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


class TestCLIBasics:
    """Tests for basic CLI functionality."""

    def test_cli_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Gemini OCR" in result.output

    def test_cli_version(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_cli_shows_options(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "--output-dir" in result.output
        assert "--api-key" in result.output
        assert "--model" in result.output
        assert "--dry-run" in result.output
        assert "--quiet" in result.output
        assert "--workers" in result.output


class TestProcessCommand:
    """Tests for file processing."""

    def test_process_missing_file(self, runner):
        result = runner.invoke(cli, ["nonexistent.pdf"])
        assert result.exit_code != 0

    def test_process_with_api_key(self, runner, sample_pdf):
        with patch("gemini_ocr.cli.OCRProcessor") as mock_processor:
            mock_instance = MagicMock()
            mock_processor.return_value = mock_instance
            runner.invoke(cli, [str(sample_pdf), "--api-key", "test-key"])

    def test_process_with_output_dir(self, runner, sample_pdf, tmp_path):
        output_dir = tmp_path / "output"
        with patch("gemini_ocr.cli.OCRProcessor") as mock_processor:
            mock_instance = MagicMock()
            mock_processor.return_value = mock_instance
            runner.invoke(cli, [str(sample_pdf), "-o", str(output_dir), "--api-key", "test-key"])


class TestDryRun:
    """Tests for --dry-run mode."""

    def test_dry_run_single_file(self, runner, sample_pdf):
        result = runner.invoke(cli, [str(sample_pdf), "--dry-run"])
        assert result.exit_code == 0
        assert "dry run" in result.output.lower()

    def test_dry_run_no_api_key_needed(self, runner, sample_pdf):
        with patch.dict(os.environ, {}, clear=True):
            env = {k: v for k, v in os.environ.items()
                   if k not in ("GEMINI_API_KEY", "GOOGLE_API_KEY")}
            with patch.dict(os.environ, env, clear=True):
                result = runner.invoke(cli, [str(sample_pdf), "--dry-run"])
                assert result.exit_code == 0


class TestInfoFlag:
    """Tests for --info flag."""

    def test_info_shows_system_info(self, runner):
        # --info requires INPUT_PATH to not be required, but our CLI requires it
        # So we pass a dummy path along with --info
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
            result = runner.invoke(cli, ["--info", "."])
            assert result.exit_code == 0
            assert "System Information" in result.output
            assert "Python" in result.output

    def test_info_shows_config(self, runner):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("google.genai.Client") as mock_client_class:
                mock_client = MagicMock()
                mock_client.models.list.return_value = []
                mock_client_class.return_value = mock_client
                result = runner.invoke(cli, ["--info", "."])
                assert "Configuration" in result.output
                assert "Model" in result.output


class TestQuietMode:
    """Tests for --quiet mode."""

    def test_quiet_suppresses_banner(self, runner, sample_pdf):
        with patch("gemini_ocr.cli.OCRProcessor") as mock_processor:
            mock_instance = MagicMock()
            mock_processor.return_value = mock_instance
            result = runner.invoke(
                cli, [str(sample_pdf), "--api-key", "test-key", "--quiet"]
            )
            assert "Gemini OCR" not in result.output
