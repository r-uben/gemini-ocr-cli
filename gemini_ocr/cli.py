"""Command-line interface for Gemini OCR."""

import os
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from gemini_ocr import __version__
from gemini_ocr.config import Config
from gemini_ocr.processor import OCRProcessor
from gemini_ocr.utils import setup_logging

console = Console()

# Get original working directory if set (for wrapper scripts)
ORIGINAL_CWD = os.environ.get("GEMINI_OCR_CWD", os.getcwd())


def print_banner() -> None:
    """Print CLI banner."""
    console.print(f"[bold blue]Gemini OCR[/bold blue] [dim]v{__version__}[/dim]")
    console.print("[dim]Powered by Google Gemini[/dim]\n")


@click.group()
@click.version_option(version=__version__, prog_name="gemini-ocr")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """Gemini OCR - Document processing using Google Gemini's vision capabilities.

    Process PDF and image files with state-of-the-art OCR, extracting text,
    tables, equations, and figures with high accuracy.

    \b
    Examples:
        gemini-ocr process document.pdf
        gemini-ocr process ./papers/ --recursive
        gemini-ocr describe figure.png
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    setup_logging(verbose=verbose)


@cli.command()
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
    default="gemini-3.0-flash",
    help="Gemini model to use (default: gemini-3.0-flash)",
)
@click.option(
    "--task",
    type=click.Choice(["convert", "extract", "table"]),
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
    help="Save original input images (default: True)",
)
@click.option(
    "--add-timestamp/--no-timestamp",
    default=False,
    help="Add timestamp to output folder (default: False)",
)
@click.option(
    "--reprocess",
    is_flag=True,
    help="Reprocess files even if already done",
)
@click.option(
    "--env-file",
    type=click.Path(exists=True, path_type=Path),
    help="Path to .env file with configuration",
)
@click.pass_context
def process(
    ctx: click.Context,
    input_path: Path,
    output_dir: Optional[Path],
    api_key: Optional[str],
    model: str,
    task: str,
    prompt: Optional[str],
    include_images: bool,
    save_originals: bool,
    add_timestamp: bool,
    reprocess: bool,
    env_file: Optional[Path],
) -> None:
    """Process documents and images with OCR.

    INPUT_PATH can be a single file or directory.

    \b
    Supported formats:
      - Images: JPG, PNG, WEBP, GIF, BMP, TIFF
      - Documents: PDF

    \b
    Examples:
        # Process a single PDF
        gemini-ocr process paper.pdf

        # Process directory with custom output
        gemini-ocr process ./documents -o ./results

        # Use specific model
        gemini-ocr process doc.pdf --model gemini-1.5-pro

        # Custom OCR prompt
        gemini-ocr process form.jpg --prompt "Extract all form fields"
    """
    print_banner()

    try:
        # Resolve paths relative to original CWD
        if not input_path.is_absolute():
            input_path = Path(ORIGINAL_CWD) / input_path

        if not input_path.exists():
            raise ValueError(f"Input path does not exist: {input_path}")

        if output_dir and not output_dir.is_absolute():
            output_dir = Path(ORIGINAL_CWD) / output_dir

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
        config.verbose = ctx.obj["verbose"]

        # Create processor and run
        processor = OCRProcessor(config)
        processor.process(
            input_path,
            output_path=output_dir,
            task=task,
            custom_prompt=prompt,
            add_timestamp=add_timestamp,
            reprocess=reprocess,
        )

        console.print("\n[bold green]Done![/bold green]\n")

    except ValueError as e:
        console.print(f"\n[red]Error:[/red] {e}\n")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]\n")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}\n")
        if ctx.obj["verbose"]:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument("image_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--api-key",
    type=str,
    envvar="GEMINI_API_KEY",
    help="Gemini API key",
)
@click.option(
    "--model",
    type=str,
    default="gemini-2.0-flash",
    help="Gemini model to use",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Output file for description (default: stdout)",
)
@click.pass_context
def describe(
    ctx: click.Context,
    image_path: Path,
    api_key: Optional[str],
    model: str,
    output: Optional[Path],
) -> None:
    """Generate detailed description of a figure/chart/diagram.

    Analyzes the image and provides structured description including:
    - Type of visualization
    - Axes, labels, components
    - Data/information conveyed
    - Key findings

    \b
    Examples:
        gemini-ocr describe chart.png
        gemini-ocr describe diagram.jpg -o description.md
    """
    print_banner()

    try:
        # Resolve path
        if not image_path.is_absolute():
            image_path = Path(ORIGINAL_CWD) / image_path

        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key

        config = Config.from_env()
        config.model = model
        config.verbose = ctx.obj["verbose"]

        processor = OCRProcessor(config)

        console.print(f"[blue]Analyzing:[/blue] {image_path.name}\n")

        description = processor.describe_figure(image_path)

        if output:
            if not output.is_absolute():
                output = Path(ORIGINAL_CWD) / output
            output.write_text(description, encoding="utf-8")
            console.print(f"[green]Saved to:[/green] {output}")
        else:
            console.print("[bold]Description:[/bold]\n")
            console.print(description)

        console.print("\n[bold green]Done![/bold green]\n")

    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}\n")
        if ctx.obj["verbose"]:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
def info() -> None:
    """Show configuration and system information."""
    print_banner()

    # System info
    sys_table = Table(title="System Information")
    sys_table.add_column("Component", style="cyan")
    sys_table.add_column("Value", style="green")

    sys_table.add_row("Python", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    sys_table.add_row("Platform", sys.platform)

    console.print(sys_table)
    console.print()

    # Configuration
    try:
        config = Config.from_env()

        config_table = Table(title="Configuration")
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="yellow")

        config_table.add_row("API Key", "Set" if config.api_key else "[red]Not set[/red]")
        config_table.add_row("Model", config.model)
        config_table.add_row("DPI", str(config.dpi))
        config_table.add_row("Max File Size", f"{config.max_file_size_mb} MB")
        config_table.add_row("Include Images", str(config.include_images))

        console.print(config_table)
        console.print()

        # Test API if key is set
        if config.api_key:
            console.print("[dim]Testing API connection...[/dim]")
            try:
                from google import genai
                client = genai.Client(api_key=config.api_key)
                # Try to list models to verify connection
                models = list(client.models.list())
                console.print(f"[green]API connection successful[/green]")
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
    """Entry point with shorthand support.

    Allows `gemini-ocr file.pdf` as shorthand for `gemini-ocr process file.pdf`
    """
    argv = sys.argv[1:]

    if argv:
        known_commands = {"process", "describe", "info"}

        # Find first non-option argument
        first_arg_idx = None
        for idx, arg in enumerate(argv):
            if not arg.startswith("-"):
                first_arg_idx = idx
                break

        # If it's a file path, insert "process" command
        if first_arg_idx is not None:
            candidate = argv[first_arg_idx]
            if candidate not in known_commands:
                # Check if it looks like a path
                potential_path = Path(ORIGINAL_CWD) / candidate if not Path(candidate).is_absolute() else Path(candidate)
                if potential_path.exists():
                    argv = argv[:first_arg_idx] + ["process"] + argv[first_arg_idx:]
                    sys.argv = [sys.argv[0], *argv]

    cli(obj={})


if __name__ == "__main__":
    main()
