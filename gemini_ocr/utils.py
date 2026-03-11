"""Utility functions for Gemini OCR CLI."""

import logging
import re
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Supported file extensions
SUPPORTED_IMAGES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".tif"}
SUPPORTED_DOCUMENTS = {".pdf"}
SUPPORTED_EXTENSIONS = SUPPORTED_IMAGES | SUPPORTED_DOCUMENTS


def setup_logging(level: str = "INFO", verbose: bool = False) -> None:
    """Configure logging for the application."""
    log_level = logging.DEBUG if verbose else getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def is_supported_file(file_path: Path) -> bool:
    """Check if file type is supported."""
    return file_path.suffix.lower() in SUPPORTED_EXTENSIONS


def is_image_file(file_path: Path) -> bool:
    """Check if file is an image."""
    return file_path.suffix.lower() in SUPPORTED_IMAGES


def is_pdf_file(file_path: Path) -> bool:
    """Check if file is a PDF."""
    return file_path.suffix.lower() in SUPPORTED_DOCUMENTS


def get_supported_files(directory: Path, recursive: bool = True) -> list[Path]:
    """Get all supported files in a directory, excluding output directories."""
    pattern = "**/*" if recursive else "*"
    files = []
    for file_path in directory.glob(pattern):
        if (
            file_path.is_file()
            and is_supported_file(file_path)
            and "gemini_ocr_output" not in file_path.parts
        ):
            files.append(file_path)
    return sorted(files)


def sanitize_filename(filename: str, max_length: int | None = 200) -> str:
    """Sanitize filename for safe filesystem usage."""
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", filename)
    sanitized = re.sub(r"\s+", "_", sanitized)
    sanitized = re.sub(r"_+", "_", sanitized)
    sanitized = sanitized.strip("_")
    if max_length and len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    return sanitized or "unnamed"


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def determine_output_path(
    input_path: Path,
    output_path: Path | None = None,
) -> Path:
    """Determine the output directory path."""
    if output_path:
        base_output = output_path
    elif input_path.is_file():
        base_output = input_path.parent / "gemini_ocr_output"
    else:
        base_output = input_path / "gemini_ocr_output"
    base_output.mkdir(parents=True, exist_ok=True)
    return base_output


def extract_pdf_images(pdf_path: Path) -> list[dict[str, Any]]:
    """Extract embedded images from PDF."""
    doc = fitz.open(pdf_path)
    extracted = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        image_list = page.get_images()

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                extracted.append(
                    {
                        "page": page_idx + 1,
                        "index": img_idx + 1,
                        "data": base_image["image"],
                        "ext": base_image["ext"],
                        "width": base_image.get("width"),
                        "height": base_image.get("height"),
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to extract image {img_idx} from page {page_idx}: {e}")

    doc.close()
    return extracted


def get_pdf_page_count(pdf_path: Path) -> int:
    """Get page count of a PDF without fully loading it."""
    doc = fitz.open(pdf_path)
    count = len(doc)
    doc.close()
    return count
