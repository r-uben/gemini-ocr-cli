"""Core OCR processing module using Google Gemini with native PDF support."""

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
from rich.progress import Progress, SpinnerColumn, TextColumn

from gemini_ocr.config import Config
from gemini_ocr.retry import retry, is_retryable_error
from gemini_ocr.utils import (
    determine_output_path,
    extract_pdf_images,
    format_file_size,
    get_supported_files,
    is_image_file,
    is_pdf_file,
    load_metadata,
    sanitize_filename,
    save_metadata,
)

logger = logging.getLogger(__name__)
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
    error: Optional[str] = None
    processing_time: float = 0.0
    token_count: Optional[int] = None
    extracted_images: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def total_pages(self) -> int:
        """Estimate page count (for compatibility)."""
        # Rough estimate: ~3000 chars per page
        return max(1, len(self.text) // 3000) if self.text else 0


class OCRProcessor:
    """OCR processor using Google Gemini API with native PDF support."""

    def __init__(self, config: Config):
        """Initialize the OCR processor."""
        self.config = config
        config.validate_api_key()

        # Initialize Gemini client
        self.client = genai.Client(api_key=config.api_key)
        self.model_name = config.model

        self.errors: List[Dict] = []
        self.processed_files: List[Dict] = []

        logger.info(f"Initialized OCRProcessor with model: {config.model}")

    def _upload_file(self, file_path: Path) -> Any:
        """Upload file to Gemini Files API.

        Args:
            file_path: Path to the file to upload

        Returns:
            Uploaded file object from Gemini API
        """
        if self.config.verbose:
            console.print(f"[dim]Uploading {file_path.name}...[/dim]")

        uploaded = self.client.files.upload(file=str(file_path))

        # Wait for file to be processed
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

        return types.Part.from_bytes(
            data=buffer.getvalue(),
            mime_type="image/jpeg",
        )

    @retry(max_attempts=3, backoff_factor=2.0, initial_delay=1.0)
    def _generate_content(
        self,
        contents: List[Any],
        prompt: str,
    ) -> str:
        """Generate content with retry logic.

        Args:
            contents: List of content parts (files, images, text)
            prompt: The prompt to send

        Returns:
            Generated text response
        """
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt, *contents],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=8192,
            ),
        )

        if response.text:
            return response.text.strip()
        return ""

    def process_image(
        self,
        image_path: Path,
        task: str = "convert",
        custom_prompt: Optional[str] = None,
    ) -> OCRResult:
        """Process a single image file.

        Args:
            image_path: Path to the image file
            task: OCR task type
            custom_prompt: Optional custom prompt

        Returns:
            OCRResult with extracted text
        """
        start_time = time.time()

        try:
            self.config.validate_file_size(image_path)
            image = Image.open(image_path)

            if image.mode != "RGB":
                image = image.convert("RGB")

            prompt = custom_prompt or OCR_PROMPTS.get(task, OCR_PROMPTS["convert"])
            image_part = self._pil_to_part(image)

            text = self._generate_content([image_part], prompt)
            processing_time = time.time() - start_time

            return OCRResult(
                file_path=image_path,
                text=text,
                success=True,
                processing_time=processing_time,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            logger.error(f"Error processing {image_path}: {error_msg}")

            return OCRResult(
                file_path=image_path,
                text="",
                success=False,
                error=error_msg,
                processing_time=processing_time,
            )

    def process_pdf(
        self,
        pdf_path: Path,
        task: str = "convert",
        custom_prompt: Optional[str] = None,
        show_progress: bool = True,
    ) -> OCRResult:
        """Process a PDF file using native Gemini PDF support.

        This method uploads the entire PDF to Gemini's Files API and processes
        it in a single API call, which is faster and more accurate than
        converting to images page-by-page.

        Args:
            pdf_path: Path to the PDF file
            task: OCR task type
            custom_prompt: Optional custom prompt
            show_progress: Whether to show progress indicator

        Returns:
            OCRResult with extracted text
        """
        start_time = time.time()
        self.config.validate_file_size(pdf_path)

        try:
            # Upload PDF to Gemini Files API
            if show_progress:
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
                    text = self._generate_content([uploaded_file], prompt)
            else:
                uploaded_file = self._upload_file(pdf_path)
                prompt = custom_prompt or OCR_PROMPTS.get(task, OCR_PROMPTS["convert"])
                text = self._generate_content([uploaded_file], prompt)

            # Extract embedded images if configured
            extracted_images = []
            if self.config.include_images:
                try:
                    extracted_images = extract_pdf_images(pdf_path)
                except Exception as e:
                    logger.warning(f"Failed to extract embedded images: {e}")

            processing_time = time.time() - start_time

            return OCRResult(
                file_path=pdf_path,
                text=text,
                success=True,
                processing_time=processing_time,
                extracted_images=extracted_images,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            logger.error(f"Error processing {pdf_path}: {error_msg}")

            return OCRResult(
                file_path=pdf_path,
                text="",
                success=False,
                error=error_msg,
                processing_time=processing_time,
            )

    def describe_figure(self, image_path: Path) -> str:
        """Generate a detailed description of a figure/chart.

        Args:
            image_path: Path to the image file

        Returns:
            Detailed description of the figure
        """
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        image_part = self._pil_to_part(image)
        return self._generate_content([image_part], OCR_PROMPTS["describe_figure"])

    def process_file(
        self,
        file_path: Path,
        task: str = "convert",
        custom_prompt: Optional[str] = None,
        show_progress: bool = True,
    ) -> OCRResult:
        """Process a single file (image or PDF).

        Args:
            file_path: Path to the file
            task: OCR task type
            custom_prompt: Optional custom prompt
            show_progress: Whether to show progress

        Returns:
            OCRResult with extracted text
        """
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
        """Save OCR results to files.

        Args:
            result: OCRResult to save
            output_dir: Directory to save results

        Returns:
            Path to the saved markdown file
        """
        base_name = sanitize_filename(result.file_path.stem)
        markdown_path = output_dir / f"{base_name}.md"

        # Save original image if configured
        if self.config.save_original_images and is_image_file(result.file_path):
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
        content.append(f"**Processing Time:** {result.processing_time:.2f}s\n")
        content.append("\n---\n\n")

        # Add extracted content
        if result.success:
            content.append(result.text)
            content.append("\n\n")
        else:
            content.append(f"*[OCR Failed: {result.error}]*\n\n")

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
        """Process input path (file or directory).

        Args:
            input_path: Path to file or directory
            output_path: Optional output directory
            task: OCR task type
            custom_prompt: Optional custom prompt
            add_timestamp: Add timestamp to output folder
            reprocess: Reprocess already-processed files
        """
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
                }
            )
            save_metadata(output_dir, self.processed_files, result.processing_time, self.errors)
            console.print(f"\n[green]Success[/green]")
            console.print(f"[dim]Time: {result.processing_time:.2f}s[/dim]")
        else:
            self.errors.append({"file": str(file_path), "error": result.error})
            console.print(f"\n[red]Failed to process file: {result.error}[/red]")

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
                    }
                )
                success_count += 1
                console.print(f"  [green]OK[/green] ({result.processing_time:.1f}s)\n")
            else:
                self.errors.append({"file": str(file_path), "error": result.error})
                console.print(f"  [red]FAILED[/red]\n")

        total_time = time.time() - start_time
        save_metadata(output_dir, self.processed_files, total_time, self.errors)

        console.print(f"\n[green]Completed:[/green] {success_count}/{len(files_to_process)} files")
        if self.errors:
            console.print(f"[red]Errors:[/red] {len(self.errors)} file(s)")
        console.print(f"[dim]Total time: {total_time:.2f}s[/dim]")
