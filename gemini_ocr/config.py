"""Configuration management for Gemini OCR CLI."""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Configuration settings for Gemini OCR."""

    model_config = SettingsConfigDict(
        env_prefix="GEMINI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Configuration - checks GEMINI_API_KEY first, then GOOGLE_API_KEY
    api_key: str = Field(default="", description="Google Gemini API key")

    @field_validator("api_key", mode="before")
    @classmethod
    def resolve_api_key(cls, v: str) -> str:
        """Resolve API key from multiple sources.

        Priority:
        1. Explicitly passed value (non-empty)
        2. GEMINI_API_KEY environment variable
        3. GOOGLE_API_KEY environment variable (fallback)
        """
        if v:  # Explicitly provided value
            return v

        # Check environment variables
        gemini_key = os.environ.get("GEMINI_API_KEY", "")
        if gemini_key:
            return gemini_key

        google_key = os.environ.get("GOOGLE_API_KEY", "")
        if google_key:
            return google_key

        return ""

    # Model Configuration
    model: str = Field(
        default="gemini-3.0-flash",
        description="Gemini model to use for OCR",
    )

    # Processing Configuration
    include_images: bool = Field(
        default=True,
        description="Extract and save images from documents",
    )
    save_original_images: bool = Field(
        default=True,
        description="Save original input images alongside results",
    )
    dpi: int = Field(
        default=200,
        description="DPI for PDF rendering",
    )
    max_file_size_mb: float = Field(
        default=20.0,
        description="Maximum file size in MB",
    )

    # Output Configuration
    output_dir: Optional[Path] = Field(
        default=None,
        description="Default output directory",
    )

    # Runtime
    verbose: bool = Field(default=False, description="Enable verbose output")

    @classmethod
    def from_env(cls, env_file: Optional[Path] = None) -> "Config":
        """Load configuration from environment or .env file."""
        if env_file and env_file.exists():
            from dotenv import load_dotenv

            load_dotenv(env_file)

        return cls()

    def validate_api_key(self) -> None:
        """Validate that API key is set."""
        if not self.api_key:
            raise ValueError(
                "Gemini API key not set. "
                "Set GEMINI_API_KEY environment variable or pass --api-key"
            )

    def validate_file_size(self, file_path: Path) -> None:
        """Validate file size is within limits."""
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            raise ValueError(
                f"File size ({file_size_mb:.2f} MB) exceeds maximum "
                f"allowed size ({self.max_file_size_mb} MB)"
            )
