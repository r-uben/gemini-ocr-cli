"""Command-line interface for Gemini OCR."""

import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from gemini_ocr import __version__
from gemini_ocr.config import Config
from gemini_ocr.processor import OCRProcessor
from gemini_ocr.processor import console as proc_console
from gemini_ocr.utils import (
    format_file_size,
    get_pdf_page_count,
    get_supported_files,
    is_pdf_file,
    setup_logging,
)

console = Console()

# Get original working directory if set (for wrapper scripts)
ORIGINAL_CWD = os.environ.get("GEMINI_OCR_CWD", os.getcwd())


def _resolve_path(path: Path) -> Path:
    """Resolve a path relative to the original working directory."""
    if not path.is_absolute():
        return Path(ORIGINAL_CWD) / path
    return path


@click.command()
@click.argument("input_path", type=click.Path(path_type=Path))
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(path_type=Path),
    help="Output directory for results",
)
@click.option(
    "--api-key",
    type=str,
    envvar="GEMINI_API_KEY",
    help="Gemini API key (or set GEMINI_API_KEY env var)",
)
@click.option(
    "--model",
    type=str,
    default="gemini-3.1-flash-lite-preview",
    help="Gemini model to use (default: gemini-3.1-flash-lite-preview)",
)
@click.option(
    "--task",
    type=click.Choice(["convert", "extract", "table", "describe_figure"]),
    default="convert",
    help="OCR task type (default: convert)",
)
@click.option(
    "--prompt",
    type=str,
    help="Custom prompt for OCR processing",
)
@click.option(
    "--include-images/--no-images",
    default=True,
    help="Extract embedded images (default: True)",
)
@click.option(
    "--save-originals/--no-save-originals",
    default=True,
    help="Save original input images alongside results (default: True)",
)
@click.option(
    "--reprocess",
    is_flag=True,
    help="Reprocess files even if already done",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="List files that would be processed without calling API",
)
@click.option(
    "-q",
    "--quiet",
    is_flag=True,
    help="Suppress output except file paths (for scripting)",
)
@click.option(
    "-w",
    "--workers",
    type=click.IntRange(min=1),
    default=1,
    help="Number of concurrent workers (default: 1)",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable verbose output",
)
@click.option(
    "--env-file",
    type=click.Path(exists=True, path_type=Path),
    help="Path to .env file with configuration",
)
@click.option(
    "--info",
    is_flag=True,
    help="Show configuration and system information",
)
@click.version_option(version=__version__, prog_name="gemini-ocr")
def cli(
    input_path: Path,
    output_dir: Path | None,
    api_key: str | None,
    model: str,
    task: str,
    prompt: str | None,
    include_images: bool,
    save_originals: bool,
    reprocess: bool,
    dry_run: bool,
    quiet: bool,
    workers: int,
    verbose: bool,
    env_file: Path | None,
    info: bool,
) -> None:
    """Gemini OCR - Document processing using Google Gemini.

    Process PDF and image files with state-of-the-art OCR.

    \b
    Examples:
        gemini-ocr paper.pdf
        gemini-ocr ./papers/ -o ./results/
        gemini-ocr doc.pdf --model gemini-3.1-pro
        gemini-ocr chart.png --task describe_figure
        gemini-ocr --info
    """
    setup_logging(verbose=verbose)

    # Handle --info flag
    if info:
        _show_info(api_key)
        return

    # Resolve paths
    input_path = _resolve_path(input_path)
    if not input_path.exists():
        console.print(f"[red]Error:[/red] Input path does not exist: {input_path}")
        sys.exit(1)

    if output_dir:
        output_dir = _resolve_path(output_dir)

    # Set quiet mode on processor console too
    if quiet:
        proc_console.quiet = True
        console.quiet = True

    # Handle --dry-run (no API key needed)
    if dry_run:
        _dry_run(input_path)
        return

    try:
        # Load configuration
        if env_file:
            config = Config.from_env(env_file)
        else:
            if api_key:
                os.environ["GEMINI_API_KEY"] = api_key
            config = Config.from_env()

        # Override with CLI options
        config.model = model
        config.include_images = include_images
        config.save_original_images = save_originals
        config.verbose = verbose
        config.quiet = quiet
        config.max_workers = workers

        if not quiet:
            console.print(f"[bold blue]Gemini OCR[/bold blue] [dim]v{__version__}[/dim]")
            console.print(f"[dim]Model: {config.model}[/dim]\n")

        processor = OCRProcessor(config)
        processor.process(
            input_path,
            output_path=output_dir,
            task=task,
            custom_prompt=prompt,
            reprocess=reprocess,
        )

        if not quiet:
            console.print("\n[bold green]Done![/bold green]\n")

    except ValueError as e:
        console.print(f"\n[red]Error:[/red] {e}\n")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]\n")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}\n")
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def _dry_run(input_path: Path) -> None:
    """List files that would be processed without calling the API."""
    if input_path.is_file():
        files = [input_path]
    else:
        files = get_supported_files(input_path)

    if not files:
        console.print("[yellow]No supported files found[/yellow]")
        return

    table = Table(title="Files to process (dry run)")
    table.add_column("File", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("Pages", justify="right")

    total_size = 0
    for f in files:
        size = f.stat().st_size
        total_size += size
        pages = str(get_pdf_page_count(f)) if is_pdf_file(f) else "-"
        table.add_row(f.name, format_file_size(size), pages)

    console.print(table)
    console.print(f"\n[dim]Total: {len(files)} file(s), {format_file_size(total_size)}[/dim]")


def _show_info(api_key: str | None = None) -> None:
    """Show configuration and system information."""
    console.print(f"[bold blue]Gemini OCR[/bold blue] [dim]v{__version__}[/dim]\n")

    sys_table = Table(title="System Information")
    sys_table.add_column("Component", style="cyan")
    sys_table.add_column("Value", style="green")
    sys_table.add_row(
        "Python",
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )
    sys_table.add_row("Platform", sys.platform)
    console.print(sys_table)
    console.print()

    try:
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
        config = Config.from_env()

        config_table = Table(title="Configuration")
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="yellow")
        config_table.add_row("API Key", "Set" if config.api_key else "[red]Not set[/red]")
        config_table.add_row("Model", config.model)
        config_table.add_row("Max File Size", f"{config.max_file_size_mb} MB")
        config_table.add_row("Include Images", str(config.include_images))
        console.print(config_table)
        console.print()

        if config.api_key:
            console.print("[dim]Testing API connection...[/dim]")
            try:
                from google import genai

                client = genai.Client(api_key=config.api_key)
                models = list(client.models.list())
                console.print("[green]API connection successful[/green]")
                console.print(f"[dim]Available models: {len(models)}[/dim]")
            except Exception as e:
                console.print(f"[red]API connection failed:[/red] {e}")
        else:
            console.print("[yellow]Set GEMINI_API_KEY to enable API features[/yellow]")

    except Exception as e:
        console.print(f"[red]Error loading config:[/red] {e}")

    console.print()
    console.print("[bold]Supported Formats:[/bold]")
    console.print("  Images: JPG, PNG, WEBP, GIF, BMP, TIFF")
    console.print("  Documents: PDF")
    console.print()


def main() -> None:
    """Entry point — handles bare invocations and delegates to cli()."""
    argv = sys.argv[1:]

    # If no args at all, show help
    if not argv:
        cli(["--help"])
        return

    cli()


if __name__ == "__main__":
    main()
