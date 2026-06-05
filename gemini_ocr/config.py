"""Configuration management for Gemini OCR CLI."""

import os
from pathlib import Path

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

    # API Configuration
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
        if v:
            return v
        gemini_key = os.environ.get("GEMINI_API_KEY", "")
        if gemini_key:
            return gemini_key
        google_key = os.environ.get("GOOGLE_API_KEY", "")
        if google_key:
            return google_key
        return ""

    # Model Configuration
    model: str = Field(
        default="gemini-3-flash-preview",
        description="Gemini model to use for OCR",
    )

    # Processing Configuration
    #: PDF processing mode:
    #:   "auto"      — whole-PDF upload (one call), auto-fall-back to per-page if
    #:                 the whole-PDF response is truncated (DEFAULT).
    #:   "whole_pdf" — force whole-PDF (one call), NO fallback (accept truncation).
    #:   "per_page"  — force page-by-page (one call per page) always.
    pdf_mode: str = Field(
        default="auto",
        description="PDF processing mode: auto | whole_pdf | per_page",
    )

    @field_validator("pdf_mode", mode="before")
    @classmethod
    def normalize_pdf_mode(cls, v: str) -> str:
        """Normalize/validate the PDF mode (tolerant of hyphens / case)."""
        if not v:
            return "auto"
        norm = str(v).strip().lower().replace("-", "_")
        if norm not in {"auto", "whole_pdf", "per_page"}:
            raise ValueError(f"Invalid pdf_mode {v!r}; expected auto, whole_pdf, or per_page")
        return norm

    include_images: bool = Field(
        default=True,
        description="Extract and save images from documents",
    )
    save_original_images: bool = Field(
        default=True,
        description="Save original input images alongside results",
    )
    max_file_size_mb: float = Field(
        default=50.0,
        description="Maximum file size in MB",
    )

    # Concurrency
    max_workers: int = Field(default=1, description="Number of concurrent workers")
    max_retries: int = Field(default=3, description="Max retry attempts for transient errors")
    retry_base_delay: float = Field(default=1.0, description="Base delay for exponential backoff")

    # Output Configuration
    output_dir: Path | None = Field(
        default=None,
        description="Default output directory",
    )

    # Runtime
    verbose: bool = Field(default=False, description="Enable verbose output")
    quiet: bool = Field(default=False, description="Suppress all output except paths")

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> "Config":
        """Load configuration from environment or .env file."""
        if env_file and env_file.exists():
            from dotenv import load_dotenv

            load_dotenv(env_file)
        return cls()

    def validate_api_key(self) -> None:
        """Validate that API key is set."""
        if not self.api_key:
            raise ValueError(
                "Gemini API key not set. Set GEMINI_API_KEY environment variable or pass --api-key"
            )

    def validate_file_size(self, file_path: Path) -> None:
        """Validate file size is within limits."""
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            raise ValueError(
                f"File size ({file_size_mb:.2f} MB) exceeds maximum "
                f"allowed size ({self.max_file_size_mb} MB)"
            )
