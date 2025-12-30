"""Tests for configuration module."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from gemini_ocr.config import Config


class TestConfigAPIKey:
    """Tests for API key resolution."""

    def test_api_key_from_gemini_env(self):
        """Test API key from GEMINI_API_KEY."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-key"}, clear=True):
            config = Config()
            assert config.api_key == "gemini-key"

    def test_api_key_from_google_env(self):
        """Test fallback to GOOGLE_API_KEY."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "google-key"}, clear=True):
            config = Config()
            assert config.api_key == "google-key"

    def test_api_key_priority(self):
        """Test GEMINI_API_KEY takes priority over GOOGLE_API_KEY."""
        with patch.dict(
            os.environ,
            {"GEMINI_API_KEY": "gemini-key", "GOOGLE_API_KEY": "google-key"},
            clear=True,
        ):
            config = Config()
            assert config.api_key == "gemini-key"

    def test_api_key_empty_when_not_set(self):
        """Test API key is empty when not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove any existing keys
            env = {k: v for k, v in os.environ.items()
                   if k not in ("GEMINI_API_KEY", "GOOGLE_API_KEY")}
            with patch.dict(os.environ, env, clear=True):
                config = Config()
                assert config.api_key == ""

    def test_validate_api_key_raises_when_empty(self):
        """Test validate_api_key raises ValueError when empty."""
        with patch.dict(os.environ, {}, clear=True):
            env = {k: v for k, v in os.environ.items()
                   if k not in ("GEMINI_API_KEY", "GOOGLE_API_KEY")}
            with patch.dict(os.environ, env, clear=True):
                config = Config()
                config.api_key = ""
                with pytest.raises(ValueError, match="Gemini API key not set"):
                    config.validate_api_key()


class TestConfigDefaults:
    """Tests for default configuration values."""

    def test_default_model(self):
        """Test default model name."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test"}, clear=True):
            config = Config()
            assert config.model == "gemini-3.0-flash"

    def test_default_dpi(self):
        """Test default DPI value."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test"}, clear=True):
            config = Config()
            assert config.dpi == 200

    def test_default_max_file_size(self):
        """Test default max file size."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test"}, clear=True):
            config = Config()
            assert config.max_file_size_mb == 20.0

    def test_default_include_images(self):
        """Test default include_images setting."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test"}, clear=True):
            config = Config()
            assert config.include_images is True


class TestConfigFileValidation:
    """Tests for file validation."""

    def test_validate_file_size_passes(self, tmp_path):
        """Test file size validation passes for small files."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test"}, clear=True):
            config = Config()
            config.max_file_size_mb = 10.0

            # Create a small file
            test_file = tmp_path / "small.txt"
            test_file.write_text("small content")

            # Should not raise
            config.validate_file_size(test_file)

    def test_validate_file_size_raises_for_large_file(self, tmp_path):
        """Test file size validation raises for large files."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test"}, clear=True):
            config = Config()
            config.max_file_size_mb = 0.0001  # Very small limit

            # Create a file that exceeds limit
            test_file = tmp_path / "large.txt"
            test_file.write_text("x" * 1000)

            with pytest.raises(ValueError, match="exceeds maximum"):
                config.validate_file_size(test_file)


class TestConfigFromEnv:
    """Tests for loading config from .env file."""

    def test_from_env_loads_dotenv(self, tmp_path):
        """Test from_env loads .env file."""
        env_file = tmp_path / ".env"
        env_file.write_text("GEMINI_API_KEY=dotenv-key\n")

        # Clear existing env vars to ensure .env takes effect
        env = {k: v for k, v in os.environ.items()
               if k not in ("GEMINI_API_KEY", "GOOGLE_API_KEY")}
        with patch.dict(os.environ, env, clear=True):
            config = Config.from_env(env_file)
            assert config.api_key == "dotenv-key"

    def test_from_env_without_file(self):
        """Test from_env works without .env file."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "env-key"}, clear=True):
            config = Config.from_env()
            assert config.api_key == "env-key"
