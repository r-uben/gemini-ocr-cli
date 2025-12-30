"""Utility functions for Gemini OCR CLI."""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
from PIL import Image

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


def get_supported_files(directory: Path, recursive: bool = True) -> List[Path]:
    """Get all supported files in a directory."""
    pattern = "**/*" if recursive else "*"
    files = []
    for file_path in directory.glob(pattern):
        if file_path.is_file() and is_supported_file(file_path):
            files.append(file_path)
    return sorted(files)


def sanitize_filename(filename: str, max_length: Optional[int] = 200) -> str:
    """Sanitize filename for safe filesystem usage."""
    # Remove or replace invalid characters
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
    output_path: Optional[Path] = None,
    add_timestamp: bool = False,
) -> Path:
    """Determine the output directory path."""
    if output_path:
        base_output = output_path
    elif input_path.is_file():
        base_output = input_path.parent / "gemini_ocr_output"
    else:
        base_output = input_path / "gemini_ocr_output"

    if add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output = base_output.parent / f"{base_output.name}_{timestamp}"

    base_output.mkdir(parents=True, exist_ok=True)
    return base_output


def pdf_to_images(
    pdf_path: Path,
    dpi: int = 200,
    pages: Optional[List[int]] = None,
) -> List[Image.Image]:
    """Convert PDF pages to PIL Images."""
    doc = fitz.open(pdf_path)
    images = []

    page_indices = pages if pages else range(len(doc))

    for page_idx in page_indices:
        if page_idx >= len(doc):
            logger.warning(f"Page {page_idx} out of range, skipping")
            continue

        page = doc[page_idx]
        # Calculate zoom factor for desired DPI (default PDF is 72 DPI)
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix)

        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    doc.close()
    return images


def extract_pdf_images(pdf_path: Path) -> List[Dict[str, Any]]:
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


def load_metadata(output_dir: Path) -> Dict[str, Any]:
    """Load existing metadata from output directory."""
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"files_processed": [], "errors": [], "total_processing_time": 0}


def save_metadata(
    output_dir: Path,
    processed_files: List[Dict],
    processing_time: float,
    errors: List[Dict],
) -> None:
    """Save processing metadata to JSON file."""
    metadata_path = output_dir / "metadata.json"

    # Load existing metadata and merge
    existing = load_metadata(output_dir)
    existing_files_set = {item["file"] for item in existing["files_processed"]}

    # Add new processed files
    for item in processed_files:
        if item["file"] not in existing_files_set:
            existing["files_processed"].append(item)

    # Add new errors
    existing["errors"].extend(errors)
    existing["total_processing_time"] += processing_time
    existing["last_updated"] = datetime.now().isoformat()

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, default=str)
