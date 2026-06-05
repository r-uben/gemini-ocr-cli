# Gemini OCR CLI

[![CI](https://github.com/r-uben/gemini-ocr-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/r-uben/gemini-ocr-cli/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/gemini-ocr-cli.svg)](https://badge.fury.io/py/gemini-ocr-cli)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A command-line tool for OCR processing using Google Gemini's vision capabilities. Process PDFs and images to extract text, tables, equations, and figures.

## Choosing an OCR tool

This is one of five OCR CLI tools with a shared design: clean Markdown output, batch processing, and figure extraction. Pick based on your constraints:

| Tool | Engine | Runs | Cost | Best for |
|------|--------|------|------|----------|
| [deepseek-ocr-cli](https://github.com/r-uben/deepseek-ocr-cli) | DeepSeek vision | Local (Ollama / vLLM) | Free | General-purpose local OCR with multi-backend flexibility |
| **gemini-ocr-cli** (this repo) | Google Gemini | Cloud API | Free tier / Pay-per-use | Fast cloud OCR with concurrent processing |
| [marker-ocr-cli](https://github.com/r-uben/marker-ocr-cli) | Marker (Surya + Texify) | Local | Free | Academic papers with equations, tables, complex layouts |
| [mistral-ocr-cli](https://github.com/r-uben/mistral-ocr-cli) | Mistral OCR API | Cloud API | ~$1/1k pages | Structured extraction (tables, headers, footers) |
| [nougat-ocr-cli](https://github.com/r-uben/nougat-ocr-cli) | Meta Nougat | Local (GPU) | Free | Academic papers, GPU-accelerated batch processing |

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
  -o, --output-dir PATH           Output directory (default: <input_parent>/ocr/)
  --api-key TEXT                  Gemini API key (or set GEMINI_API_KEY env var)
  --model TEXT                    Model to use (default: gemini-3-flash-preview)
  --task [convert|extract|table|describe_figure]
                                  OCR task type (default: convert)
  --prompt TEXT                   Custom prompt for OCR processing

  --per-page                      Force page-by-page OCR (one call/page).
  --whole-pdf                     Force whole-PDF OCR (one call/document), NO
                                  truncation fallback.
                                  (default: auto = whole-PDF with auto-fallback)
  --include-images/--no-images    Extract embedded figures (default: True)

  -w, --workers N                 Concurrent workers for batch processing (default: 1)
  --reprocess                     Reprocess already-processed files
  --dry-run                       List files without calling the API
  -q, --quiet                     Print only output .md paths (for scripting)
  -v, --verbose                   Enable verbose/debug output
  --info                          Show configuration and system info
  --env-file PATH                 Path to .env file
  --version                       Show version
  --help                          Show this message
```

## Output structure

`gemini-ocr` follows the shared OCR-CLI output contract. The default output
root is an `ocr/` folder next to the input (`-o` overrides it). Each source
document gets one aggregated folder that **mirrors the input subtree, keyed by
the input-relative path** (not the basename), so two `intro.pdf` files in
different directories never collide:

```
<input_parent>/ocr/
├── a/
│   └── intro/
│       ├── intro.md            # all pages, separated by "## Page N" headers
│       ├── figures/            # figure_<N>_page<P>.png (normalised to PNG)
│       │   └── figure_1_page2.png
│       └── metadata.json       # per-document provenance sidecar
├── b/
│   └── intro/                  # same basename, different subtree — no collision
│       └── intro.md
└── metadata.json               # rolled-up index keyed by input-relative path
```

The markdown body is clean (no YAML frontmatter); all provenance lives in the
JSON sidecars. Each `metadata.json` entry records
`{status, checksum, model, backend, processing_time, timestamp, output_path,
pages, mode, fell_back_from_whole_pdf}`. Failures are recorded with
`status: "failed"`, and the process exits non-zero if **any** file or page fails
(uniform across single-file and batch).

## PDF processing modes

PDFs can be processed in three modes; the output **structure is identical**
across all three (one `<stem>.md` with `## Page N` headers, dual-level metadata,
figures). The mode actually used is recorded per document in metadata (`mode`),
together with whether an auto run fell back (`fell_back_from_whole_pdf`):

- **`auto` (default).** Uploads the whole PDF once and OCRs it in a **single**
  call (the model is asked to begin each page with a `## Page N` header, which is
  split back into pages). If that response looks **truncated**, the document is
  automatically re-processed page-by-page and that result is used instead, with a
  one-line note printed and the fallback recorded in metadata
  (`mode: "per_page"`, `fell_back_from_whole_pdf: true`). Truncation is detected
  from real signals, with no hardcoded page-count threshold: (a) a length-limited
  finish reason (`MAX_TOKENS` / `LENGTH`), or (b) fewer `## Page N` markers
  recovered than the PDF's actual page count (PyMuPDF). This is the
  cost/quality-preferred default for academic PDFs: one call per document, full
  document context, but no silently truncated tails on long papers.
- **`--per-page`.** Forces page-by-page OCR: each page is rendered to an image and
  OCR'd in its own call (N calls per document). Honest per-page failure tracking
  (one page can fail without sinking the document) at higher cost.
- **`--whole-pdf`.** Forces a single whole-PDF call with **no** truncation
  fallback. Cheapest, but a long document may be silently truncated; use only
  when you explicitly accept that risk.

```bash
gemini-ocr paper.pdf              # auto: whole-PDF, auto-fallback on truncation
gemini-ocr paper.pdf --per-page   # force page-by-page (N calls, per-page failures)
gemini-ocr paper.pdf --whole-pdf  # force single call, no fallback (truncation risk)
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
| `--model` | `GEMINI_MODEL` | `gemini-3-flash-preview` |
| `--per-page` / `--whole-pdf` | `GEMINI_PDF_MODE` (`auto`\|`per_page`\|`whole_pdf`) | `auto` |
| `--include-images` | `GEMINI_INCLUDE_IMAGES` | `true` |
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
