"""Gemini OCR CLI - Document processing using Google Gemini's vision capabilities."""

__version__ = "0.2.1"

from gemini_ocr.processor import OCRProcessor
from gemini_ocr.config import Config

__all__ = ["OCRProcessor", "Config", "__version__"]
