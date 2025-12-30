# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2024-12-23

### Changed

- **BREAKING**: Replaced page-by-page PDF processing with native Gemini Files API upload
  - PDFs are now uploaded directly to Gemini (single API call per document)
  - Significantly faster processing for multi-page documents
  - Better quality: native PDF parsing preserves text, tables, and layout
- Updated default model from `gemini-2.0-flash-exp` to `gemini-3.0-flash`
- Simplified `OCRResult` dataclass (removed per-page tracking)

### Added

- Retry logic with exponential backoff for API rate limits
- Comprehensive test suite (105 unit tests, integration tests)
- `token_count` field in `OCRResult` for usage tracking

### Removed

- `--dpi` CLI flag (no longer applicable with native PDF upload)
- `GEMINI_DPI` environment variable
- Unused utility functions: `image_to_base64`, `pil_image_to_base64`, `save_base64_image`
- Module-level `settings` singleton from config

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
