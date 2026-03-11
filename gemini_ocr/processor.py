"""Core OCR processing module using Google Gemini with native PDF support."""

import io
import logging
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from PIL import Image
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

from gemini_ocr.config import Config
from gemini_ocr.metadata import MetadataManager
from gemini_ocr.utils import (
    determine_output_path,
    extract_pdf_images,
    format_file_size,
    get_supported_files,
    is_image_file,
    is_pdf_file,
    sanitize_filename,
)

logger = logging.getLogger(__name__)

# Shared console instance — CLI sets .quiet on this directly
console = Console()

# OCR prompts for different tasks
OCR_PROMPTS = {
    "convert": """Extract all text from this document and convert it to clean markdown format.

Rules:
- Preserve the document structure (headings, paragraphs, lists, tables)
- Convert tables to markdown table format
- Preserve mathematical equations in LaTeX format where possible
- Include figure/image captions if present
- Do not describe images, just note their presence as [Figure X] or [Image]
- Output ONLY the extracted text in markdown, no commentary""",
    "extract": """Extract all visible text from this document exactly as it appears.
Output only the extracted text, preserving line breaks and spacing.""",
    "describe_figure": """Analyze this figure/chart/diagram in detail:
1. What type of visualization is this? (bar chart, line graph, flowchart, etc.)
2. What are the axes, labels, or key components?
3. What data or information does it convey?
4. What are the main findings or takeaways?

Provide a structured description.""",
    "table": """Extract all tables from this document and convert them to markdown format.
Preserve all data, headers, and structure. Output only the markdown tables.""",
}


@dataclass
class OCRResult:
    """Result from processing a document."""

    file_path: Path
    text: str
    success: bool
    error: str | None = None
    processing_time: float = 0.0
    extracted_images: list[dict[str, Any]] = field(default_factory=list)


class OCRProcessor:
    """OCR processor using Google Gemini API with native PDF support."""

    def __init__(self, config: Config):
        """Initialize the OCR processor."""
        self.config = config
        config.validate_api_key()
        self.client = genai.Client(api_key=config.api_key)
        self.model_name = config.model
        self._lock = threading.Lock()
        logger.info(f"Initialized OCRProcessor with model: {config.model}")

    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        """Check if an error is transient and worth retrying."""
        # Google GenAI SDK typed errors
        for exc_name in ("ResourceExhausted", "InternalServerError", "ServiceUnavailable"):
            if type(error).__name__ == exc_name:
                return True
        # httpx-level HTTP status errors
        if hasattr(error, "response"):
            status = getattr(error.response, "status_code", 0)
            if status in (429, 500, 502, 503, 504):
                return True
        # Network-level transient errors
        if isinstance(error, (TimeoutError, ConnectionError, OSError)):
            return True
        # Check error message for rate-limit indicators
        error_str = str(error).lower()
        return "429" in error_str or "rate limit" in error_str or "quota" in error_str

    def _call_with_retry(self, contents: list[Any], prompt: str) -> str:
        """Call generate_content with exponential backoff on transient errors."""
        max_attempts = self.config.max_retries + 1
        base_delay = self.config.retry_base_delay

        for attempt in range(max_attempts):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[prompt, *contents],
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                    ),
                )
                if response.text:
                    return response.text.strip()
                return ""
            except Exception as e:
                is_last = attempt == max_attempts - 1
                if is_last or not self._is_retryable(e):
                    raise
                delay = base_delay * (2**attempt)
                logger.warning(
                    "Retryable error (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1,
                    max_attempts,
                    e,
                    delay,
                )
                time.sleep(delay)
        raise RuntimeError("Retry loop exited unexpectedly")

    def _upload_file(self, file_path: Path) -> Any:
        """Upload file to Gemini Files API."""
        if self.config.verbose:
            console.print(f"[dim]Uploading {file_path.name}...[/dim]")

        uploaded = self.client.files.upload(file=str(file_path))

        while uploaded.state == "PROCESSING":
            time.sleep(0.5)
            uploaded = self.client.files.get(name=uploaded.name)

        if uploaded.state == "FAILED":
            raise RuntimeError(f"File upload failed: {uploaded.name}")

        if self.config.verbose:
            console.print(f"[dim]Upload complete: {uploaded.name}[/dim]")

        return uploaded

    def _pil_to_part(self, image: Image.Image) -> types.Part:
        """Convert PIL Image to Gemini Part."""
        buffer = io.BytesIO()
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buffer, format="JPEG", quality=95)
        buffer.seek(0)
        return types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/jpeg")

    def process_image(
        self,
        image_path: Path,
        task: str = "convert",
        custom_prompt: str | None = None,
    ) -> OCRResult:
        """Process a single image file."""
        start_time = time.time()
        try:
            self.config.validate_file_size(image_path)
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            prompt = custom_prompt or OCR_PROMPTS.get(task, OCR_PROMPTS["convert"])
            image_part = self._pil_to_part(image)
            text = self._call_with_retry([image_part], prompt)
            return OCRResult(
                file_path=image_path,
                text=text,
                success=True,
                processing_time=time.time() - start_time,
            )
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return OCRResult(
                file_path=image_path,
                text="",
                success=False,
                error=str(e),
                processing_time=time.time() - start_time,
            )

    def process_pdf(
        self,
        pdf_path: Path,
        task: str = "convert",
        custom_prompt: str | None = None,
        show_progress: bool = True,
    ) -> OCRResult:
        """Process a PDF file using native Gemini PDF support."""
        start_time = time.time()
        self.config.validate_file_size(pdf_path)

        try:
            if show_progress and not self.config.quiet:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                    transient=True,
                ) as progress:
                    progress.add_task("Uploading PDF...", total=None)
                    uploaded_file = self._upload_file(pdf_path)
                    progress.update(progress.task_ids[0], description="Processing...")
                    prompt = custom_prompt or OCR_PROMPTS.get(task, OCR_PROMPTS["convert"])
                    text = self._call_with_retry([uploaded_file], prompt)
            else:
                uploaded_file = self._upload_file(pdf_path)
                prompt = custom_prompt or OCR_PROMPTS.get(task, OCR_PROMPTS["convert"])
                text = self._call_with_retry([uploaded_file], prompt)

            extracted_images = []
            if self.config.include_images:
                try:
                    extracted_images = extract_pdf_images(pdf_path)
                except Exception as e:
                    logger.warning(f"Failed to extract embedded images: {e}")

            return OCRResult(
                file_path=pdf_path,
                text=text,
                success=True,
                processing_time=time.time() - start_time,
                extracted_images=extracted_images,
            )
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            return OCRResult(
                file_path=pdf_path,
                text="",
                success=False,
                error=str(e),
                processing_time=time.time() - start_time,
            )

    def process_file(
        self,
        file_path: Path,
        task: str = "convert",
        custom_prompt: str | None = None,
        show_progress: bool = True,
    ) -> OCRResult:
        """Process a single file (image or PDF)."""
        if is_pdf_file(file_path):
            return self.process_pdf(
                file_path, task=task, custom_prompt=custom_prompt, show_progress=show_progress
            )
        elif is_image_file(file_path):
            return self.process_image(file_path, task=task, custom_prompt=custom_prompt)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

    def save_results(self, result: OCRResult, output_dir: Path) -> Path:
        """Save OCR results to per-document folder.

        Output structure:
            output_dir/doc_name/doc_name.md
            output_dir/doc_name/figures/page1_img1.png
        """
        base_name = sanitize_filename(result.file_path.stem)
        doc_dir = output_dir / base_name
        doc_dir.mkdir(parents=True, exist_ok=True)
        markdown_path = doc_dir / f"{base_name}.md"

        # Save original image if configured
        if self.config.save_original_images and is_image_file(result.file_path):
            original_output = doc_dir / f"{base_name}{result.file_path.suffix}"
            shutil.copy2(result.file_path, original_output)

        # Write clean markdown — just the OCR text, no headers
        markdown_path.write_text(
            result.text if result.success else f"*[OCR Failed: {result.error}]*", encoding="utf-8"
        )

        # Save extracted images
        if result.extracted_images and self.config.include_images:
            figures_dir = doc_dir / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)
            for img_info in result.extracted_images:
                img_filename = f"page{img_info['page']}_img{img_info['index']}.{img_info['ext']}"
                img_path = figures_dir / img_filename
                with open(img_path, "wb") as f:
                    f.write(img_info["data"])

        if self.config.verbose:
            console.print(f"[green]Saved:[/green] {markdown_path}")

        return markdown_path

    def process(
        self,
        input_path: Path,
        output_path: Path | None = None,
        task: str = "convert",
        custom_prompt: str | None = None,
        reprocess: bool = False,
    ) -> None:
        """Process input path (file or directory)."""
        if input_path.is_file():
            self._process_single_file(input_path, output_path, task, custom_prompt, reprocess)
        elif input_path.is_dir():
            self._process_directory(input_path, output_path, task, custom_prompt, reprocess)
        else:
            raise ValueError(f"Input path does not exist: {input_path}")

    def _process_single_file(
        self,
        file_path: Path,
        output_path: Path | None,
        task: str,
        custom_prompt: str | None,
        reprocess: bool,
    ) -> None:
        """Process a single file."""
        output_dir = determine_output_path(file_path, output_path)
        meta = MetadataManager(output_dir)

        if meta.is_processed(file_path) and not reprocess:
            console.print(f"[yellow]Already processed:[/yellow] {file_path.name}")
            console.print("[dim]Use --reprocess to force reprocessing[/dim]")
            return

        console.print(f"[blue]Processing:[/blue] {file_path}")
        console.print(f"[blue]Output:[/blue] {output_dir}\n")

        result = self.process_file(file_path, task=task, custom_prompt=custom_prompt)

        if result.success:
            output_file = self.save_results(result, output_dir)
            meta.record(
                file_path,
                processing_time=result.processing_time,
                model=self.model_name,
                output_path=str(output_file.relative_to(output_dir)),
            )
            console.print("\n[green]Success[/green]")
            console.print(f"[dim]Time: {result.processing_time:.2f}s[/dim]")
        else:
            console.print(f"\n[red]Failed to process file: {result.error}[/red]")

    def _process_directory(
        self,
        dir_path: Path,
        output_path: Path | None,
        task: str,
        custom_prompt: str | None,
        reprocess: bool,
    ) -> None:
        """Process all files in a directory."""
        files = get_supported_files(dir_path)
        if not files:
            console.print("[yellow]No supported files found[/yellow]")
            return

        output_dir = determine_output_path(dir_path, output_path)
        meta = MetadataManager(output_dir)

        # Filter files
        files_to_process = []
        for f in files:
            if meta.is_processed(f) and not reprocess:
                if self.config.verbose:
                    console.print(f"[dim]Skipping: {f.name}[/dim]")
            else:
                files_to_process.append(f)

        if not files_to_process:
            console.print("[green]All files already processed[/green]")
            console.print("[dim]Use --reprocess to force reprocessing[/dim]")
            return

        console.print(f"[blue]Processing {len(files_to_process)} file(s)...[/blue]")
        console.print(f"[blue]Output:[/blue] {output_dir}\n")

        start_time = time.time()
        success_count = 0

        if self.config.max_workers > 1:
            success_count = self._process_directory_concurrent(
                files_to_process, output_dir, meta, task, custom_prompt
            )
        else:
            for file_path in files_to_process:
                file_size = format_file_size(file_path.stat().st_size)
                console.print(f"[cyan]{file_path.name}[/cyan] ({file_size})")

                result = self.process_file(file_path, task=task, custom_prompt=custom_prompt)

                if result.success:
                    output_file = self.save_results(result, output_dir)
                    with self._lock:
                        meta.record(
                            file_path,
                            processing_time=result.processing_time,
                            model=self.model_name,
                            output_path=str(output_file.relative_to(output_dir)),
                        )
                    success_count += 1
                    console.print(f"  [green]OK[/green] ({result.processing_time:.1f}s)\n")
                else:
                    console.print(f"  [red]FAILED: {result.error}[/red]\n")

        total_time = time.time() - start_time
        console.print(f"\n[green]Completed:[/green] {success_count}/{len(files_to_process)} files")
        console.print(f"[dim]Total time: {total_time:.2f}s[/dim]")

    def _process_directory_concurrent(
        self,
        files: list[Path],
        output_dir: Path,
        meta: MetadataManager,
        task: str,
        custom_prompt: str | None,
    ) -> int:
        """Process files concurrently with a thread pool."""
        success_count = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        ) as progress:
            progress_task = progress.add_task("Processing...", total=len(files))

            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_path = {
                    executor.submit(self.process_file, f, task, custom_prompt, False): f
                    for f in files
                }

                for future in as_completed(future_to_path):
                    file_path = future_to_path[future]
                    try:
                        result = future.result()
                        if result.success:
                            output_file = self.save_results(result, output_dir)
                            with self._lock:
                                meta.record(
                                    file_path,
                                    processing_time=result.processing_time,
                                    model=self.model_name,
                                    output_path=str(output_file.relative_to(output_dir)),
                                )
                            success_count += 1
                            console.print(
                                f"  [green]OK[/green] {file_path.name} ({result.processing_time:.1f}s)"
                            )
                        else:
                            console.print(f"  [red]FAILED[/red] {file_path.name}: {result.error}")
                    except Exception as e:
                        console.print(f"  [red]ERROR[/red] {file_path.name}: {e}")

                    progress.advance(progress_task)

        return success_count
