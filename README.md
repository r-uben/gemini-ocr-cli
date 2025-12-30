# Gemini OCR CLI

Command-line tool for OCR processing using Google Gemini's vision capabilities. Extract text, tables, equations, and figures from PDFs and images with high accuracy.

## Features

- **Native PDF upload**: Direct PDF processing via Gemini Files API (fast, single API call)
- **Multi-format support**: PDF and images (JPG, PNG, WEBP, GIF, BMP, TIFF)
- **High-quality OCR**: Leverages Gemini's advanced vision models
- **Structure preservation**: Maintains headings, tables, lists, equations
- **Figure analysis**: Generate detailed descriptions of charts and diagrams
- **Batch processing**: Process entire directories with progress tracking
- **Incremental processing**: Skip already-processed files
- **Automatic retry**: Exponential backoff for API rate limits
- **Markdown output**: Clean, structured output format

## Installation

### From PyPI (recommended)

```bash
pip install gemini-ocr-cli
```

### Using pipx

```bash
pipx install gemini-ocr-cli
```

### From source

```bash
git clone https://github.com/r-uben/gemini-ocr-cli.git
cd gemini-ocr-cli
uv pip install -e .
```

## Quick Start

### API Key Resolution

The CLI automatically picks up your API key from environment variables (no configuration needed if already set):

**Priority order:**
1. `--api-key` CLI argument (highest priority)
2. `GEMINI_API_KEY` environment variable
3. `GOOGLE_API_KEY` environment variable (fallback)
4. `.env` file in current directory

```bash
# Option 1: Set environment variable (recommended)
export GEMINI_API_KEY="your-api-key"

# Option 2: Use existing GOOGLE_API_KEY (auto-detected)
export GOOGLE_API_KEY="your-api-key"

# Option 3: Create a .env file
echo "GEMINI_API_KEY=your-api-key" > .env

# Option 4: Pass directly (not recommended for security)
gemini-ocr paper.pdf --api-key "your-api-key"
```

### Process documents

```bash
# Single file
gemini-ocr paper.pdf

# Directory
gemini-ocr ./documents/ -o ./results/

# With custom model
gemini-ocr paper.pdf --model gemini-1.5-pro
```

### Describe figures

```bash
# Analyze a chart/diagram
gemini-ocr describe chart.png

# Save to file
gemini-ocr describe figure.jpg -o description.md
```

## CLI Reference

### `gemini-ocr process`

Process documents and images with OCR.

```
Usage: gemini-ocr process [OPTIONS] INPUT_PATH

Options:
  -o, --output-dir PATH           Output directory for results
  --api-key TEXT                  Gemini API key
  --model TEXT                    Model to use (default: gemini-3.0-flash)
  --task [convert|extract|table]  OCR task type (default: convert)
  --prompt TEXT                   Custom prompt for OCR
  --include-images/--no-images    Extract embedded images (default: True)
  --save-originals/--no-save-originals
                                  Save original input images (default: True)
  --add-timestamp/--no-timestamp  Add timestamp to output folder
  --reprocess                     Reprocess existing files
  --env-file PATH                 Path to .env file
  -v, --verbose                   Enable verbose output
```

### `gemini-ocr describe`

Generate detailed descriptions of figures, charts, and diagrams.

```
Usage: gemini-ocr describe [OPTIONS] IMAGE_PATH

Options:
  --api-key TEXT    Gemini API key
  --model TEXT      Model to use
  -o, --output PATH Output file (default: stdout)
```

### `gemini-ocr info`

Show configuration and system information.

## Output Format

Results are saved as Markdown files with:
- File metadata (original path, processing time)
- Extracted text (full document)
- Embedded image references (if enabled)
- `metadata.json` tracking all processed files

## Models

| Model | Speed | Quality | Cost | Recommended For |
|-------|-------|---------|------|-----------------|
| `gemini-3.0-flash` | Fast | Good | Low | Default, most documents |
| `gemini-1.5-flash` | Fast | Good | Low | Simple documents |
| `gemini-1.5-pro` | Slower | Best | Higher | Complex layouts, equations |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | Required |
| `GOOGLE_API_KEY` | Fallback API key | - |
| `GEMINI_MODEL` | Default model | `gemini-3.0-flash` |

## License

MIT
