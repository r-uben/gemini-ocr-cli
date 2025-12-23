"""Core OCR processing module using Google Gemini."""

import io
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types
from PIL import Image
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from gemini_ocr.config import Config
from gemini_ocr.utils import (
    determine_output_path,
    extract_pdf_images,
    format_file_size,
    get_supported_files,
    is_image_file,
    is_pdf_file,
    load_metadata,
    pdf_to_images,
    sanitize_filename,
    save_metadata,
)

logger = logging.getLogger(__name__)
console = Console()


# OCR prompts for different tasks
OCR_PROMPTS = {
    "convert": """Extract all text from this document image and convert it to clean markdown format.

Rules:
- Preserve the document structure (headings, paragraphs, lists, tables)
- Convert tables to markdown table format
- Preserve mathematical equations in LaTeX format where possible
- Include figure/image captions if present
- Do not describe images, just note their presence as [Figure X] or [Image]
- Output ONLY the extracted text in markdown, no commentary""",
    "extract": """Extract all visible text from this image exactly as it appears.
Output only the extracted text, preserving line breaks and spacing.""",
    "describe_figure": """Analyze this figure/chart/diagram in detail:
1. What type of visualization is this? (bar chart, line graph, flowchart, etc.)
2. What are the axes, labels, or key components?
3. What data or information does it convey?
4. What are the main findings or takeaways?

Provide a structured description.""",
    "table": """Extract the table from this image and convert it to markdown format.
Preserve all data, headers, and structure. Output only the markdown table.""",
}


@dataclass
class PageResult:
    """Result from processing a single page."""

    page_number: int
    text: str
    success: bool
    error: Optional[str] = None
    processing_time: float = 0.0


@dataclass
class OCRResult:
    """Result from processing a document."""

    file_path: Path
    pages: List[PageResult]
    total_pages: int
    successful_pages: int
    failed_pages: int
    processing_time: float
    extracted_images: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.successful_pages > 0

    @property
    def text(self) -> str:
        """Get combined text from all pages."""
        parts = []
        for page in self.pages:
            if page.success and page.text:
                parts.append(f"<!-- Page {page.page_number} -->\n{page.text}")
        return "\n\n".join(parts)


class OCRProcessor:
    """OCR processor using Google Gemini API."""

    def __init__(self, config: Config):
        """Initialize the OCR processor."""
        self.config = config
        config.validate_api_key()

        # Initialize Gemini client with new API
        self.client = genai.Client(api_key=config.api_key)
        self.model_name = config.model

        self.errors: List[Dict] = []
        self.processed_files: List[Dict] = []

        logger.info(f"Initialized OCRProcessor with model: {config.model}")

    def _pil_to_part(self, image: Image.Image) -> types.Part:
        """Convert PIL Image to Gemini Part."""
        buffer = io.BytesIO()
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buffer, format="JPEG", quality=95)
        buffer.seek(0)

        return types.Part.from_bytes(
            data=buffer.getvalue(),
            mime_type="image/jpeg",
        )

    def _process_image_with_gemini(
        self,
        image: Image.Image,
        prompt: str,
    ) -> str:
        """Process a single image with Gemini."""
        try:
            image_part = self._pil_to_part(image)

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, image_part],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=8192,
                ),
            )

            if response.text:
                return response.text.strip()
            return ""

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    def process_image(
        self,
        image_path: Path,
        task: str = "convert",
        custom_prompt: Optional[str] = None,
    ) -> OCRResult:
        """Process a single image file."""
        start_time = time.time()

        try:
            self.config.validate_file_size(image_path)
            image = Image.open(image_path)

            if image.mode != "RGB":
                image = image.convert("RGB")

            prompt = custom_prompt or OCR_PROMPTS.get(task, OCR_PROMPTS["convert"])
            text = self._process_image_with_gemini(image, prompt)

            processing_time = time.time() - start_time

            return OCRResult(
                file_path=image_path,
                pages=[
                    PageResult(
                        page_number=1,
                        text=text,
                        success=True,
                        processing_time=processing_time,
                    )
                ],
                total_pages=1,
                successful_pages=1,
                failed_pages=0,
                processing_time=processing_time,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            logger.error(f"Error processing {image_path}: {error_msg}")

            return OCRResult(
                file_path=image_path,
                pages=[
                    PageResult(
                        page_number=1,
                        text="",
                        success=False,
                        error=error_msg,
                        processing_time=processing_time,
                    )
                ],
                total_pages=1,
                successful_pages=0,
                failed_pages=1,
                processing_time=processing_time,
            )

    def process_pdf(
        self,
        pdf_path: Path,
        task: str = "convert",
        custom_prompt: Optional[str] = None,
        pages: Optional[List[int]] = None,
        show_progress: bool = True,
    ) -> OCRResult:
        """Process a PDF file page by page."""
        start_time = time.time()
        self.config.validate_file_size(pdf_path)

        # Convert PDF pages to images
        if self.config.verbose:
            console.print(f"[dim]Converting PDF to images at {self.config.dpi} DPI...[/dim]")

        page_images = pdf_to_images(pdf_path, dpi=self.config.dpi, pages=pages)
        total_pages = len(page_images)

        if total_pages == 0:
            return OCRResult(
                file_path=pdf_path,
                pages=[],
                total_pages=0,
                successful_pages=0,
                failed_pages=0,
                processing_time=time.time() - start_time,
            )

        prompt = custom_prompt or OCR_PROMPTS.get(task, OCR_PROMPTS["convert"])
        page_results: List[PageResult] = []
        successful = 0
        failed = 0

        # Extract embedded images if configured
        extracted_images = []
        if self.config.include_images:
            try:
                extracted_images = extract_pdf_images(pdf_path)
            except Exception as e:
                logger.warning(f"Failed to extract embedded images: {e}")

        # Process each page
        if show_progress and total_pages > 1:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task_id = progress.add_task(f"Processing {total_pages} pages...", total=total_pages)

                for idx, image in enumerate(page_images):
                    page_num = (pages[idx] if pages else idx) + 1
                    progress.update(task_id, description=f"Page {page_num}/{total_pages}...")

                    page_result = self._process_single_page(image, page_num, prompt)
                    page_results.append(page_result)

                    if page_result.success:
                        successful += 1
                    else:
                        failed += 1

                    progress.update(task_id, advance=1)
        else:
            for idx, image in enumerate(page_images):
                page_num = (pages[idx] if pages else idx) + 1

                if self.config.verbose:
                    console.print(f"[dim]Processing page {page_num}/{total_pages}...[/dim]")

                page_result = self._process_single_page(image, page_num, prompt)
                page_results.append(page_result)

                if page_result.success:
                    successful += 1
                else:
                    failed += 1

        processing_time = time.time() - start_time

        return OCRResult(
            file_path=pdf_path,
            pages=page_results,
            total_pages=total_pages,
            successful_pages=successful,
            failed_pages=failed,
            processing_time=processing_time,
            extracted_images=extracted_images,
        )

    def _process_single_page(
        self,
        image: Image.Image,
        page_number: int,
        prompt: str,
    ) -> PageResult:
        """Process a single page image."""
        page_start = time.time()

        try:
            text = self._process_image_with_gemini(image, prompt)
            return PageResult(
                page_number=page_number,
                text=text,
                success=True,
                processing_time=time.time() - page_start,
            )
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error processing page {page_number}: {error_msg}")
            return PageResult(
                page_number=page_number,
                text="",
                success=False,
                error=error_msg,
                processing_time=time.time() - page_start,
            )

    def describe_figure(self, image_path: Path) -> str:
        """Generate a detailed description of a figure/chart."""
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        return self._process_image_with_gemini(image, OCR_PROMPTS["describe_figure"])

    def process_file(
        self,
        file_path: Path,
        task: str = "convert",
        custom_prompt: Optional[str] = None,
        show_progress: bool = True,
    ) -> OCRResult:
        """Process a single file (image or PDF)."""
        if is_pdf_file(file_path):
            return self.process_pdf(
                file_path,
                task=task,
                custom_prompt=custom_prompt,
                show_progress=show_progress,
            )
        elif is_image_file(file_path):
            return self.process_image(
                file_path,
                task=task,
                custom_prompt=custom_prompt,
            )
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

    def save_results(
        self,
        result: OCRResult,
        output_dir: Path,
    ) -> Path:
        """Save OCR results to files."""
        base_name = sanitize_filename(result.file_path.stem)
        markdown_path = output_dir / f"{base_name}.md"

        # Save original image if configured
        if (
            self.config.save_original_images
            and is_image_file(result.file_path)
        ):
            originals_dir = output_dir / "original_images"
            originals_dir.mkdir(parents=True, exist_ok=True)
            original_output = originals_dir / f"{base_name}{result.file_path.suffix}"
            shutil.copy2(result.file_path, original_output)

        # Build markdown content
        content = []
        content.append(f"# OCR Results\n")
        content.append(f"**Original File:** {result.file_path.name}\n")
        content.append(f"**Full Path:** `{result.file_path}`\n")
        content.append(f"**Processed:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        content.append(f"**Pages:** {result.successful_pages}/{result.total_pages} successful\n")
        content.append(f"**Processing Time:** {result.processing_time:.2f}s\n")
        content.append("\n---\n\n")

        # Add page content
        for page in result.pages:
            if result.total_pages > 1:
                content.append(f"## Page {page.page_number}\n\n")

            if page.success:
                content.append(page.text)
                content.append("\n\n")
            else:
                content.append(f"*[OCR Failed: {page.error}]*\n\n")

        # Save extracted images if any
        if result.extracted_images and self.config.include_images:
            images_dir = output_dir / "extracted_images"
            images_dir.mkdir(parents=True, exist_ok=True)

            content.append("## Extracted Images\n\n")

            for img_info in result.extracted_images:
                img_filename = (
                    f"{base_name}_page{img_info['page']}_img{img_info['index']}.{img_info['ext']}"
                )
                img_path = images_dir / img_filename

                with open(img_path, "wb") as f:
                    f.write(img_info["data"])

                content.append(f"![Page {img_info['page']} Image {img_info['index']}]")
                content.append(f"(./extracted_images/{img_filename})\n\n")

        # Write markdown
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write("".join(content))

        if self.config.verbose:
            console.print(f"[green]Saved:[/green] {markdown_path}")

        return markdown_path

    def process(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        task: str = "convert",
        custom_prompt: Optional[str] = None,
        add_timestamp: bool = False,
        reprocess: bool = False,
    ) -> None:
        """Process input path (file or directory)."""
        if input_path.is_file():
            self._process_single_file(
                input_path,
                output_path,
                task,
                custom_prompt,
                add_timestamp,
                reprocess,
            )
        elif input_path.is_dir():
            self._process_directory(
                input_path,
                output_path,
                task,
                custom_prompt,
                add_timestamp,
                reprocess,
            )
        else:
            raise ValueError(f"Input path does not exist: {input_path}")

    def _process_single_file(
        self,
        file_path: Path,
        output_path: Optional[Path],
        task: str,
        custom_prompt: Optional[str],
        add_timestamp: bool,
        reprocess: bool,
    ) -> None:
        """Process a single file."""
        output_dir = determine_output_path(file_path, output_path, add_timestamp)

        # Check if already processed
        existing = load_metadata(output_dir)
        existing_files = {item["file"] for item in existing["files_processed"]}

        if str(file_path) in existing_files and not reprocess:
            console.print(f"[yellow]Already processed:[/yellow] {file_path.name}")
            console.print("[dim]Use --reprocess to force reprocessing[/dim]")
            return

        console.print(f"[blue]Processing:[/blue] {file_path}")
        console.print(f"[blue]Output:[/blue] {output_dir}\n")

        result = self.process_file(file_path, task=task, custom_prompt=custom_prompt)

        if result.success:
            output_file = self.save_results(result, output_dir)
            self.processed_files.append(
                {
                    "file": str(file_path),
                    "size": file_path.stat().st_size,
                    "output": str(output_file),
                    "pages": result.total_pages,
                    "successful_pages": result.successful_pages,
                }
            )
            save_metadata(output_dir, self.processed_files, result.processing_time, self.errors)
            console.print(f"\n[green]Success:[/green] {result.successful_pages}/{result.total_pages} pages")
            console.print(f"[dim]Time: {result.processing_time:.2f}s[/dim]")
        else:
            self.errors.append({"file": str(file_path), "error": "All pages failed"})
            console.print(f"\n[red]Failed to process file[/red]")

    def _process_directory(
        self,
        dir_path: Path,
        output_path: Optional[Path],
        task: str,
        custom_prompt: Optional[str],
        add_timestamp: bool,
        reprocess: bool,
    ) -> None:
        """Process all files in a directory."""
        files = get_supported_files(dir_path)

        if not files:
            console.print("[yellow]No supported files found[/yellow]")
            return

        output_dir = determine_output_path(dir_path, output_path, add_timestamp)
        existing = load_metadata(output_dir)
        existing_files = {item["file"] for item in existing["files_processed"]}

        # Filter files
        files_to_process = []
        for f in files:
            if str(f) in existing_files and not reprocess:
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

        for file_path in files_to_process:
            file_size = format_file_size(file_path.stat().st_size)
            console.print(f"[cyan]{file_path.name}[/cyan] ({file_size})")

            result = self.process_file(file_path, task=task, custom_prompt=custom_prompt)

            if result.success:
                output_file = self.save_results(result, output_dir)
                self.processed_files.append(
                    {
                        "file": str(file_path),
                        "size": file_path.stat().st_size,
                        "output": str(output_file),
                        "pages": result.total_pages,
                        "successful_pages": result.successful_pages,
                    }
                )
                success_count += 1
                console.print(f"  [green]OK[/green] ({result.successful_pages}/{result.total_pages} pages)\n")
            else:
                self.errors.append({"file": str(file_path), "error": "Processing failed"})
                console.print(f"  [red]FAILED[/red]\n")

        total_time = time.time() - start_time
        save_metadata(output_dir, self.processed_files, total_time, self.errors)

        console.print(f"\n[green]Completed:[/green] {success_count}/{len(files_to_process)} files")
        if self.errors:
            console.print(f"[red]Errors:[/red] {len(self.errors)} file(s)")
        console.print(f"[dim]Total time: {total_time:.2f}s[/dim]")
