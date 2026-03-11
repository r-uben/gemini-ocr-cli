# Gemini OCR CLI

[![CI](https://github.com/r-uben/gemini-ocr-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/r-uben/gemini-ocr-cli/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/gemini-ocr-cli.svg)](https://badge.fury.io/py/gemini-ocr-cli)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A command-line tool for OCR processing using Google Gemini's vision capabilities. Process PDFs and images to extract text, tables, equations, and figures.

## Installation

Requires Python 3.11+ and a [Google Gemini API key](https://aistudio.google.com/apikey).

```bash
pip install gemini-ocr-cli
```

Or from source:

```bash
git clone https://github.com/r-uben/gemini-ocr-cli.git
cd gemini-ocr-cli
uv sync
```

## Quick start

```bash
# Set your API key
export GEMINI_API_KEY="your_key_here"

# Process a single file
gemini-ocr document.pdf

# Process a directory
gemini-ocr ./documents -o ./results

# Preview what would be processed (no API calls)
gemini-ocr ./documents --dry-run

# Process 4 files concurrently
gemini-ocr ./documents -w 4
```

## Options

```
Usage: gemini-ocr [OPTIONS] INPUT_PATH

Options:
  -o, --output-dir PATH           Output directory (default: <input_dir>/gemini_ocr_output/)
  --api-key TEXT                  Gemini API key (or set GEMINI_API_KEY env var)
  --model TEXT                    Model to use (default: gemini-3.1-flash-lite-preview)
  --task [convert|extract|table|describe_figure]
                                  OCR task type (default: convert)
  --prompt TEXT                   Custom prompt for OCR processing

  --include-images/--no-images    Extract embedded images (default: True)
  --save-originals/--no-save-originals  Copy original images to output (default: True)

  -w, --workers N                 Concurrent workers for batch processing (default: 1)
  --reprocess                     Reprocess already-processed files
  --dry-run                       List files without calling the API
  -q, --quiet                     Suppress all output except errors
  -v, --verbose                   Enable verbose/debug output
  --info                          Show configuration and system info
  --env-file PATH                 Path to .env file
  --version                       Show version
  --help                          Show this message
```

## Output structure

```
gemini_ocr_output/
├── document_name/
│   ├── document_name.md        # OCR markdown (clean text only)
│   └── figures/                # extracted embedded images
│       ├── page1_img1.png
│       └── page2_img1.png
├── another_document/
│   └── ...
└── metadata.json               # processing stats, checksums, file list
```

## API key resolution

**Priority order:**
1. `--api-key` CLI argument
2. `GEMINI_API_KEY` environment variable
3. `GOOGLE_API_KEY` environment variable (fallback)
4. `.env` file in current directory

## Configuration

All CLI options can also be set via environment variables or a `.env` file:

| CLI flag | Environment variable | Default |
|----------|---------------------|---------|
| `--api-key` | `GEMINI_API_KEY` | (required) |
| `--model` | `GEMINI_MODEL` | `gemini-3.1-flash-lite-preview` |
| `--include-images` | `GEMINI_INCLUDE_IMAGES` | `true` |
| `--save-originals` | `GEMINI_SAVE_ORIGINAL_IMAGES` | `true` |
| `--workers` | `GEMINI_MAX_WORKERS` | `1` |
| `--verbose` | `GEMINI_VERBOSE` | `false` |
| | `GEMINI_MAX_FILE_SIZE_MB` | `50` |
| | `GEMINI_MAX_RETRIES` | `3` |
| | `GEMINI_RETRY_BASE_DELAY` | `1.0` |

CLI flags override environment variables when explicitly passed.

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run mypy gemini_ocr/ --ignore-missing-imports
```

## Limitations

- Maximum file size: 50 MB (configurable via `GEMINI_MAX_FILE_SIZE_MB`)
- Supported formats: PDF, JPG, JPEG, PNG, WEBP, GIF, BMP, TIFF

## License

MIT License - see [LICENSE](LICENSE) for details.
