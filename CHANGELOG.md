# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Bumped the shared `ocr-output-contract` pin to `v0.1.3` and re-locked `uv.lock`
  so a frozen install/CI exercises the same contract the engine is reviewed
  against.

### Fixed

- **AUTO completeness oracle** (round-3 HIGH): the default `auto` path no longer
  silently caches an incomplete native OCR as `completed`. Two data-loss modes
  that escaped the tail-aware `is_truncated` now force the per-page fallback:
  (1) a `STOP`-finish response cut mid-last-page, detected via a missing
  end-of-document sentinel (the native prompt asks the model to emit a terminal
  `<!-- OCR-END -->` line, which is stripped from the saved body); and (2) a
  multi-page PDF whose response carries zero `## Page N` markers (markerless
  collapse to one blob).
- **Failure-record checksums** now use the contract's `failure_checksum`, so an
  unreadable-input `status=failed` record carries the canonical schema-valid
  `UNREADABLE_CHECKSUM` sentinel (`sha256:0…0`) instead of the non-conforming
  `"sha256:unavailable"`. A later readable run gets a real digest and reprocesses.
- **SYS-02**: an unreadable/race-deleted input no longer aborts the whole batch.
  The directory pre-filter and single-file paths now checksum via the contract's
  `safe_checksum`; a `None` result records a durable `status=failed` for that
  file and the batch CONTINUES. The serial loop's `stat()` size print is also
  moved inside per-file isolation.
- **Idempotency fingerprint** now folds in the resolved output-affecting flags
  (`pdf_mode`, `include_images`) via `run_fingerprint(extra=...)`, so a cross-mode
  re-run (e.g. `--whole-pdf` then default `auto`, or `--no-images` then
  `--include-images`) reprocesses instead of silently reusing the cached result.
- **Auto-fallback** now uses the v0.1.2 tail-aware `is_truncated`: the recovered
  physical `## Page N` marker numbers are passed in, so a legitimately-blank
  interior page no longer false-triggers a full per-page re-OCR while a genuinely
  dropped tail still does.
- **Whole-PDF page identity**: `assemble_pages` is now given the recovered marker
  numbers, so a model-skipped page is labeled by its source number instead of
  being silently renumbered to 1..N under `--whole-pdf`.
- **Files-API upload** now has a bounded poll deadline; an upload stuck in
  `PROCESSING` raises (deleting the orphan) instead of hanging a worker forever,
  so the per-file failure path records `status=failed`.

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
