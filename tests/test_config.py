"""Tests for configuration module."""

import os
from unittest.mock import patch

import pytest

from gemini_ocr.config import Config


class TestConfigAPIKey:
    """Tests for API key resolution."""

    def test_api_key_from_gemini_env(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-key"}, clear=True):
            config = Config()
            assert config.api_key == "gemini-key"

    def test_api_key_from_google_env(self):
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "google-key"}, clear=True):
            config = Config()
            assert config.api_key == "google-key"

    def test_api_key_priority(self):
        with patch.dict(
            os.environ,
            {"GEMINI_API_KEY": "gemini-key", "GOOGLE_API_KEY": "google-key"},
            clear=True,
        ):
            config = Config()
            assert config.api_key == "gemini-key"

    def test_api_key_empty_when_not_set(self):
        with patch.dict(os.environ, {}, clear=True):
            env = {k: v for k, v in os.environ.items()
                   if k not in ("GEMINI_API_KEY", "GOOGLE_API_KEY")}
            with patch.dict(os.environ, env, clear=True):
                config = Config()
                assert config.api_key == ""

    def test_validate_api_key_raises_when_empty(self):
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
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test"}, clear=True):
            config = Config()
            assert config.model == "gemini-3-flash-preview"

    def test_default_max_file_size(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test"}, clear=True):
            config = Config()
            assert config.max_file_size_mb == 50.0

    def test_default_include_images(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test"}, clear=True):
            config = Config()
            assert config.include_images is True

    def test_default_max_workers(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test"}, clear=True):
            config = Config()
            assert config.max_workers == 1

    def test_default_max_retries(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test"}, clear=True):
            config = Config()
            assert config.max_retries == 3

    def test_default_quiet(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test"}, clear=True):
            config = Config()
            assert config.quiet is False


class TestConfigFileValidation:
    """Tests for file validation."""

    def test_validate_file_size_passes(self, tmp_path):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test"}, clear=True):
            config = Config()
            test_file = tmp_path / "small.txt"
            test_file.write_text("small content")
            config.validate_file_size(test_file)

    def test_validate_file_size_raises_for_large_file(self, tmp_path):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test"}, clear=True):
            config = Config()
            config.max_file_size_mb = 0.0001
            test_file = tmp_path / "large.txt"
            test_file.write_text("x" * 1000)
            with pytest.raises(ValueError, match="exceeds maximum"):
                config.validate_file_size(test_file)


class TestConfigFromEnv:
    """Tests for loading config from .env file."""

    def test_from_env_loads_dotenv(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("GEMINI_API_KEY=dotenv-key\n")
        env = {k: v for k, v in os.environ.items()
               if k not in ("GEMINI_API_KEY", "GOOGLE_API_KEY")}
        with patch.dict(os.environ, env, clear=True):
            config = Config.from_env(env_file)
            assert config.api_key == "dotenv-key"

    def test_from_env_without_file(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "env-key"}, clear=True):
            config = Config.from_env()
            assert config.api_key == "env-key"
