"""Conformance / golden tests for the canonical OCR output contract.

These tests pin the ratified output shape so the 6 sibling CLIs that copy this
reference implementation inherit an unambiguous, regression-guarded contract.
Spec: ``docs/plans/00-output-contract/DECISION.md``.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz
import pytest

from gemini_ocr.output_contract import (
    DEFAULT_OUTPUT_DIRNAME,
    DocMetadata,
    RootIndex,
    RunOutcome,
    Status,
    assemble_pages,
    doc_dir_for,
    figure_filename,
    markdown_path_for,
    relative_key,
    resolve_output_root,
    sha256_checksum,
    utc_timestamp,
    write_doc_metadata,
)
from gemini_ocr.processor import OCRProcessor

# ---------------------------------------------------------------------------
# Pure contract helpers
# ---------------------------------------------------------------------------


class TestOutputRootResolution:
    def test_default_root_is_ocr_next_to_file(self, tmp_path):
        f = tmp_path / "doc.pdf"
        f.touch()
        assert resolve_output_root(f, None) == tmp_path / "ocr"

    def test_default_root_for_directory(self, tmp_path):
        d = tmp_path / "papers"
        d.mkdir()
        assert resolve_output_root(d, None) == d / "ocr"

    def test_output_dir_override_verbatim(self, tmp_path):
        f = tmp_path / "doc.pdf"
        f.touch()
        custom = tmp_path / "somewhere" / "else"
        assert resolve_output_root(f, custom) == custom

    def test_default_dirname_is_one_word(self):
        assert DEFAULT_OUTPUT_DIRNAME == "ocr"


class TestRelativeKeying:
    def test_key_is_relative_path_not_basename(self, tmp_path):
        root = tmp_path / "in"
        (root / "a").mkdir(parents=True)
        (root / "b").mkdir(parents=True)
        fa = root / "a" / "intro.pdf"
        fb = root / "b" / "intro.pdf"
        fa.touch()
        fb.touch()
        ka = relative_key(fa, root)
        kb = relative_key(fb, root)
        # Same basename, distinct keys — this is the collision fix.
        assert ka == "a/intro.pdf"
        assert kb == "b/intro.pdf"
        assert ka != kb

    def test_single_file_key_is_basename(self, tmp_path):
        f = tmp_path / "report.pdf"
        f.touch()
        assert relative_key(f, tmp_path) == "report.pdf"

    def test_doc_dir_mirrors_subtree(self, tmp_path):
        out = tmp_path / "ocr"
        doc_dir = doc_dir_for(out, "a/intro.pdf")
        assert doc_dir == out / "a" / "intro"
        md = markdown_path_for(doc_dir, "a/intro.pdf")
        assert md == out / "a" / "intro" / "intro.md"


class TestPageAssembly:
    def test_page_headers_present_and_numbered(self):
        body = assemble_pages(["first", "second", "third"])
        assert "## Page 1" in body
        assert "## Page 2" in body
        assert "## Page 3" in body
        # Order preserved
        assert body.index("## Page 1") < body.index("## Page 2") < body.index("## Page 3")

    def test_no_yaml_frontmatter(self):
        body = assemble_pages(["hello"])
        assert not body.lstrip().startswith("---")
        assert "source:" not in body
        assert "processed:" not in body

    def test_empty_pages(self):
        assert assemble_pages([]) == ""


class TestFigureNaming:
    def test_figure_filename_canonical(self):
        assert figure_filename(1, 3) == "figure_1_page3.png"
        assert figure_filename(2, 1) == "figure_2_page1.png"


class TestChecksumAndTimestamp:
    def test_checksum_format(self, tmp_path):
        f = tmp_path / "x.bin"
        f.write_bytes(b"abc")
        cs = sha256_checksum(f)
        assert cs.startswith("sha256:")
        assert len(cs) == len("sha256:") + 64

    def test_timestamp_is_utc_iso(self):
        ts = utc_timestamp()
        assert ts.endswith("+00:00")


class TestRunOutcome:
    def test_exit_zero_on_all_completed(self):
        o = RunOutcome()
        o.add(Status.COMPLETED)
        o.add(Status.COMPLETED)
        assert o.exit_code == 0

    def test_exit_nonzero_on_any_failure(self):
        o = RunOutcome()
        o.add(Status.COMPLETED)
        o.add(Status.FAILED, detail="b/x.pdf")
        assert o.exit_code == 1

    def test_partial_counts_as_failure_for_exit(self):
        o = RunOutcome()
        o.add(Status.PARTIAL, detail="doc.pdf")
        assert o.exit_code == 1


class TestMetadataWriters:
    def test_doc_metadata_roundtrip(self, tmp_path):
        doc_dir = tmp_path / "ocr" / "doc"
        meta = DocMetadata(
            status=Status.COMPLETED,
            checksum="sha256:deadbeef",
            model="gemini-3-flash-preview",
            backend="gemini-api",
            processing_time=1.234,
            timestamp=utc_timestamp(),
            output_path="doc/doc.md",
            pages=2,
        )
        path = write_doc_metadata(doc_dir, "doc.pdf", meta)
        data = json.loads(path.read_text())
        assert data["key"] == "doc.pdf"
        assert data["status"] == "completed"
        assert data["backend"] == "gemini-api"
        assert data["pages"] == 2
        assert data["processing_time"] == 1.23

    def test_root_index_keyed_by_relpath(self, tmp_path):
        out = tmp_path / "ocr"
        out.mkdir()
        index = RootIndex(out)
        meta = DocMetadata(
            status=Status.COMPLETED,
            checksum="sha256:abc",
            model="m",
            backend="gemini-api",
            processing_time=0.5,
            timestamp=utc_timestamp(),
            output_path="a/intro/intro.md",
            pages=1,
        )
        index.record("a/intro.pdf", meta)
        data = json.loads((out / "metadata.json").read_text())
        assert data["version"] == "1"
        assert "a/intro.pdf" in data["files"]
        assert data["files"]["a/intro.pdf"]["status"] == "completed"

    def test_root_index_tolerant_load(self, tmp_path):
        out = tmp_path / "ocr"
        out.mkdir()
        (out / "metadata.json").write_text("not valid json")
        index = RootIndex(out)
        assert index.files == {}


# ---------------------------------------------------------------------------
# End-to-end conformance through the processor (mocked Gemini)
# ---------------------------------------------------------------------------


def _make_multipage_pdf(path: Path, n_pages: int = 3) -> Path:
    doc = fitz.open()
    for i in range(n_pages):
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 72), f"Page {i + 1} heading", fontsize=20)
        page.insert_text((72, 120), f"Body text for page {i + 1}.", fontsize=12)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def processor(mock_config, mock_genai_client):
    # These conformance tests pin per-page failure semantics + tree shape, so they
    # run in explicit per-page mode (tree/metadata shape is mode-agnostic).
    mock_config.pdf_mode = "per_page"
    with patch("gemini_ocr.processor.genai") as mock_genai:
        mock_genai.Client.return_value = mock_genai_client
        proc = OCRProcessor(mock_config)
        proc.client = mock_genai_client
        return proc


@pytest.fixture
def native_processor(mock_config, mock_genai_client):
    # Auto-mode processor for native/whole-PDF + auto-fallback conformance.
    mock_config.pdf_mode = "auto"
    with patch("gemini_ocr.processor.genai") as mock_genai:
        mock_genai.Client.return_value = mock_genai_client
        proc = OCRProcessor(mock_config)
        proc.client = mock_genai_client
        return proc


def _mock_page_response(text: str, finish_reason: str = "STOP") -> MagicMock:
    """Build a GenerateContentResponse mock whose parts yield `text`."""
    part = MagicMock()
    part.text = text
    part.thought = False
    content = MagicMock()
    content.parts = [part]
    candidate = MagicMock()
    candidate.content = content
    candidate.finish_reason = finish_reason
    resp = MagicMock()
    resp.candidates = [candidate]
    return resp


class TestProcessorConformance:
    def test_multipage_output_tree_shape(self, processor, tmp_path):
        pdf = _make_multipage_pdf(tmp_path / "sample.pdf", n_pages=3)
        out = tmp_path / "out"

        # Each Gemini call returns page-specific text.
        counter = {"n": 0}

        def gen(*args, **kwargs):
            counter["n"] += 1
            return _mock_page_response(f"OCR text page {counter['n']}")

        processor.client.models.generate_content.side_effect = gen

        outcome = processor.process(pdf, output_path=out)

        # Exit policy: all pages succeeded -> 0.
        assert outcome.exit_code == 0

        # Tree shape: <out>/<stem>/<stem>.md (single-file => key is basename).
        md = out / "sample" / "sample.md"
        assert md.exists(), f"expected {md}"

        body = md.read_text()
        # ## Page N headers present for all 3 pages.
        assert "## Page 1" in body
        assert "## Page 2" in body
        assert "## Page 3" in body
        # No YAML frontmatter.
        assert not body.lstrip().startswith("---")
        # Genuine per-page text made it in.
        assert "OCR text page 1" in body

    def test_both_metadata_levels_written(self, processor, tmp_path):
        pdf = _make_multipage_pdf(tmp_path / "sample.pdf", n_pages=2)
        out = tmp_path / "out"
        processor.client.models.generate_content.side_effect = lambda *a, **k: _mock_page_response(
            "text"
        )

        processor.process(pdf, output_path=out)

        # Per-document sidecar.
        doc_meta_path = out / "sample" / "metadata.json"
        assert doc_meta_path.exists()
        doc_meta = json.loads(doc_meta_path.read_text())
        assert doc_meta["key"] == "sample.pdf"
        assert doc_meta["status"] == "completed"
        assert doc_meta["backend"] == "gemini-api"
        assert doc_meta["model"] == "gemini-3-flash-preview"
        assert doc_meta["pages"] == 2
        assert doc_meta["checksum"].startswith("sha256:")
        assert doc_meta["timestamp"].endswith("+00:00")
        assert doc_meta["output_path"] == "sample/sample.md"

        # Root index, keyed by input-relative path.
        root_meta_path = out / "metadata.json"
        assert root_meta_path.exists()
        root_meta = json.loads(root_meta_path.read_text())
        assert root_meta["version"] == "1"
        assert "sample.pdf" in root_meta["files"]
        entry = root_meta["files"]["sample.pdf"]
        assert entry["status"] == "completed"
        assert isinstance(entry["pages"], int)
        assert isinstance(entry["processing_time"], (int, float))

    def test_nested_batch_keys_by_relative_path(self, processor, tmp_path):
        root = tmp_path / "in"
        (root / "a").mkdir(parents=True)
        (root / "b").mkdir(parents=True)
        _make_multipage_pdf(root / "a" / "intro.pdf", n_pages=1)
        _make_multipage_pdf(root / "b" / "intro.pdf", n_pages=1)
        out = tmp_path / "out"
        processor.client.models.generate_content.side_effect = lambda *a, **k: _mock_page_response(
            "text"
        )

        processor.process(root, output_path=out)

        # Both same-basename docs survive in distinct mirrored subtrees.
        assert (out / "a" / "intro" / "intro.md").exists()
        assert (out / "b" / "intro" / "intro.md").exists()
        root_meta = json.loads((out / "metadata.json").read_text())
        assert "a/intro.pdf" in root_meta["files"]
        assert "b/intro.pdf" in root_meta["files"]

    def test_page_failure_yields_failed_status_and_nonzero_exit(self, processor, tmp_path):
        pdf = _make_multipage_pdf(tmp_path / "sample.pdf", n_pages=3)
        out = tmp_path / "out"

        # Page 2 raises a non-retryable error; pages 1 and 3 succeed.
        counter = {"n": 0}

        def gen(*args, **kwargs):
            counter["n"] += 1
            if counter["n"] == 2:
                raise ValueError("injected page failure")
            return _mock_page_response(f"text {counter['n']}")

        processor.client.models.generate_content.side_effect = gen

        outcome = processor.process(pdf, output_path=out)

        # A per-page failure must propagate to a nonzero exit.
        assert outcome.exit_code != 0

        # Output is still written; status recorded as partial/failed (not completed).
        doc_meta = json.loads((out / "sample" / "metadata.json").read_text())
        assert doc_meta["status"] in ("partial", "failed")
        assert "error" in doc_meta

        root_meta = json.loads((out / "metadata.json").read_text())
        assert root_meta["files"]["sample.pdf"]["status"] in ("partial", "failed")

    def test_total_failure_records_failed_status(self, processor, tmp_path):
        pdf = _make_multipage_pdf(tmp_path / "sample.pdf", n_pages=2)
        out = tmp_path / "out"
        processor.client.models.generate_content.side_effect = ValueError("all pages fail")

        outcome = processor.process(pdf, output_path=out)
        assert outcome.exit_code != 0

        doc_meta = json.loads((out / "sample" / "metadata.json").read_text())
        # Every page failed and no usable text -> failed.
        assert doc_meta["status"] == "failed"


class TestNativeAndFallbackConformance:
    """Conformance for native whole-PDF and auto-fallback paths.

    Asserts the output STRUCTURE is identical regardless of the path taken, and
    that the chosen mode + fallback flag land in BOTH metadata levels.
    """

    def test_native_whole_pdf_tree_and_metadata(self, native_processor, tmp_path):
        pdf = _make_multipage_pdf(tmp_path / "sample.pdf", n_pages=3)
        out = tmp_path / "out"
        # One complete whole-PDF call: 3 page markers, STOP finish.
        native_processor.client.models.generate_content.return_value = _mock_page_response(
            "## Page 1\n\nP1 text\n\n## Page 2\n\nP2 text\n\n## Page 3\n\nP3 text"
        )

        outcome = native_processor.process(pdf, output_path=out)
        assert outcome.exit_code == 0
        # Exactly one whole-PDF call (no fallback).
        assert native_processor.client.models.generate_content.call_count == 1

        md = out / "sample" / "sample.md"
        assert md.exists()
        body = md.read_text()
        assert "## Page 1" in body and "## Page 2" in body and "## Page 3" in body
        assert not body.lstrip().startswith("---")
        assert "P1 text" in body and "P3 text" in body

        doc_meta = json.loads((out / "sample" / "metadata.json").read_text())
        assert doc_meta["status"] == "completed"
        assert doc_meta["pages"] == 3
        assert doc_meta["mode"] == "whole_pdf"
        assert doc_meta["fell_back_from_whole_pdf"] is False

        root_meta = json.loads((out / "metadata.json").read_text())
        entry = root_meta["files"]["sample.pdf"]
        assert entry["mode"] == "whole_pdf"
        assert entry["fell_back_from_whole_pdf"] is False

    def test_auto_fallback_records_mode_and_flag(self, native_processor, tmp_path):
        pdf = _make_multipage_pdf(tmp_path / "sample.pdf", n_pages=3)
        out = tmp_path / "out"

        first = {"done": False}

        def gen(*args, **kwargs):
            if not first["done"]:
                first["done"] = True
                # Whole-PDF call truncated by token limit (tail dropped).
                return _mock_page_response(
                    "## Page 1\n\nP1\n\n## Page 2\n\nP2 (cut off",
                    finish_reason="MAX_TOKENS",
                )
            # Per-page fallback calls succeed.
            return _mock_page_response("recovered page text")

        native_processor.client.models.generate_content.side_effect = gen

        outcome = native_processor.process(pdf, output_path=out)
        assert outcome.exit_code == 0
        # 1 whole-PDF call + 3 per-page calls.
        assert native_processor.client.models.generate_content.call_count == 4

        # Structure identical to the whole-PDF path: ## Page N, no frontmatter.
        md = out / "sample" / "sample.md"
        body = md.read_text()
        assert "## Page 1" in body and "## Page 2" in body and "## Page 3" in body
        assert not body.lstrip().startswith("---")

        doc_meta = json.loads((out / "sample" / "metadata.json").read_text())
        assert doc_meta["status"] == "completed"
        assert doc_meta["pages"] == 3
        assert doc_meta["mode"] == "per_page"
        assert doc_meta["fell_back_from_whole_pdf"] is True

        root_meta = json.loads((out / "metadata.json").read_text())
        entry = root_meta["files"]["sample.pdf"]
        assert entry["mode"] == "per_page"
        assert entry["fell_back_from_whole_pdf"] is True
