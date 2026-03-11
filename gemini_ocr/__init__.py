"""Gemini OCR CLI - Document processing using Google Gemini's vision capabilities."""

__version__ = "0.3.0"

from gemini_ocr.config import Config
from gemini_ocr.processor import OCRProcessor

__all__ = ["OCRProcessor", "Config", "__version__"]
