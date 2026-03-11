# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-03-11

### Changed

- **BREAKING**: Flat CLI — single `gemini-ocr <input>` command replaces `gemini-ocr process/describe/info` subcommands
- **BREAKING**: Per-document output folders (`output/doc_name/doc_name.md` + `output/doc_name/figures/`)
- **BREAKING**: Clean markdown output — no headers or metadata in `.md` files, just the OCR text
- Default model updated to `gemini-3.1-flash-lite-preview`
- Max file size default raised to 50 MB
- Python requirement raised to >=3.11
- Retry logic now only retries transient errors (429, 5xx, timeouts)

### Added

- `--dry-run` flag — list files without calling the API (no API key required)
- `--quiet` / `-q` flag — suppress output for scripting
- `--workers` / `-w` flag — concurrent file processing with ThreadPoolExecutor
- `--task describe_figure` — replaces the old `describe` command
- `--info` flag — replaces the old `info` subcommand
- `metadata.py` module with SHA256 checksums for change detection and atomic writes
- `.github/workflows/ci.yml` — test matrix (Python 3.11/3.12/3.13)
- `.pre-commit-config.yaml` — ruff lint + format hooks
- Output directory exclusion — `get_supported_files` skips `gemini_ocr_output/`
- `get_pdf_page_count()` utility for dry-run page display

### Removed

- `describe` command (use `--task describe_figure` instead)
- `info` subcommand (use `--info` flag instead)
- `--dpi` flag and `dpi` config field (unused since v0.2.0)
- `--add-timestamp` flag
- `pdf_to_images()` utility (unused since native PDF upload)
- `retry.py` module (retry logic moved inline to processor)
- `is_retryable_error()` function (never called)
- `max_output_tokens=8192` cap (was silently truncating long documents)
- `token_count` and `total_pages` fields from `OCRResult`

## [0.2.0] - 2024-12-23

### Changed

- **BREAKING**: Replaced page-by-page PDF processing with native Gemini Files API upload
- Updated default model from `gemini-2.0-flash-exp` to `gemini-3.0-flash`
- Simplified `OCRResult` dataclass (removed per-page tracking)

### Added

- Retry logic with exponential backoff for API rate limits
- Comprehensive test suite (105 unit tests, integration tests)

### Removed

- `--dpi` CLI flag (no longer applicable with native PDF upload)
- `GEMINI_DPI` environment variable

### Fixed

- API key resolution now correctly prioritizes `GEMINI_API_KEY` over `GOOGLE_API_KEY`

## [0.1.0] - 2024-12-22

### Added

- Initial release
- PDF and image OCR using Google Gemini vision models
- CLI commands: `process`, `describe`, `info`
- Batch processing with progress tracking
- Incremental processing (skip already-processed files)
- Markdown output format
- Figure/chart description generation
- Support for multiple image formats (JPG, PNG, WEBP, GIF, BMP, TIFF)
