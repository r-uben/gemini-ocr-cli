"""Engine-level conformance tests: gemini's REAL output vs the shared contract.

The contract *primitives* (path/key computation, page assembly, metadata writers,
``is_truncated``/``split_native_pages``, the exit-code policy) are unit-tested
inside the ``ocr-output-contract`` package itself, so they are NOT re-tested here.

What stays here is the engine-side proof: run gemini's actual processor (with a
mocked Gemini client) over real multi-page PDFs and assert the produced output
tree conforms to the family-wide contract via the package's reusable
:func:`ocr_output_contract.conformance.assert_conforms` harness, plus
gemini-specific provenance (the ``mode`` / ``fell_back_from_whole_pdf`` fields)
that the engine — not the contract — decides.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz
import pytest
from ocr_output_contract.conformance import ExpectedDoc, assert_conforms

from gemini_ocr.processor import OCRProcessor

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
    """Per-page engine output conforms to the contract (via assert_conforms)."""

    def test_multipage_output_conforms(self, processor, tmp_path):
        pdf = _make_multipage_pdf(tmp_path / "sample.pdf", n_pages=3)
        out = tmp_path / "out"

        counter = {"n": 0}

        def gen(*args, **kwargs):
            counter["n"] += 1
            return _mock_page_response(f"OCR text page {counter['n']}")

        processor.client.models.generate_content.side_effect = gen

        outcome = processor.process(pdf, output_path=out)
        assert outcome.exit_code == 0

        # The package's harness asserts every canonical invariant against gemini's
        # REAL produced tree: layout, ## Page N markers, no frontmatter, dual-level
        # metadata keyed by input-relative path, and exit-policy wiring.
        assert_conforms(
            out,
            [ExpectedDoc(rel_key="sample.pdf", pages=3, status="completed")],
            require_failures_nonzero_exit=outcome.exit_code != 0,
        )
        # Genuine per-page text made it in.
        assert "OCR text page 1" in (out / "sample" / "sample.md").read_text()

    def test_metadata_provenance_fields(self, processor, tmp_path):
        pdf = _make_multipage_pdf(tmp_path / "sample.pdf", n_pages=2)
        out = tmp_path / "out"
        processor.client.models.generate_content.side_effect = lambda *a, **k: _mock_page_response(
            "text"
        )

        processor.process(pdf, output_path=out)
        assert_conforms(out, [ExpectedDoc(rel_key="sample.pdf", pages=2, status="completed")])

        # Engine-supplied provenance the contract harness does not pin (backend/model).
        doc_meta = json.loads((out / "sample" / "metadata.json").read_text())
        assert doc_meta["backend"] == "gemini-api"
        assert doc_meta["model"] == "gemini-3-flash-preview"
        assert doc_meta["output_path"] == "sample/sample.md"

    def test_nested_batch_conforms(self, processor, tmp_path):
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

        # Both same-basename docs survive in distinct mirrored subtrees, conforming.
        assert_conforms(
            out,
            [
                ExpectedDoc(rel_key="a/intro.pdf", pages=1, status="completed"),
                ExpectedDoc(rel_key="b/intro.pdf", pages=1, status="completed"),
            ],
        )

    def test_page_failure_conforms_with_partial_status(self, processor, tmp_path):
        pdf = _make_multipage_pdf(tmp_path / "sample.pdf", n_pages=3)
        out = tmp_path / "out"

        # Page 2 raises a non-retryable error; pages 1 and 3 succeed -> partial.
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

        # Output still written, recorded partial; harness verifies exit-policy wiring.
        assert_conforms(
            out,
            [ExpectedDoc(rel_key="sample.pdf", pages=3, status="partial")],
            require_failures_nonzero_exit=True,
        )
        doc_meta = json.loads((out / "sample" / "metadata.json").read_text())
        assert "error" in doc_meta

    def test_total_failure_conforms_with_failed_status(self, processor, tmp_path):
        pdf = _make_multipage_pdf(tmp_path / "sample.pdf", n_pages=2)
        out = tmp_path / "out"
        processor.client.models.generate_content.side_effect = ValueError("all pages fail")

        outcome = processor.process(pdf, output_path=out)
        assert outcome.exit_code != 0

        # Every page failed and no usable text -> failed (no markdown required).
        assert_conforms(
            out,
            [ExpectedDoc(rel_key="sample.pdf", status="failed")],
            require_failures_nonzero_exit=True,
        )


class TestNativeAndFallbackConformance:
    """Native whole-PDF and auto-fallback both produce conforming output.

    Asserts the output STRUCTURE conforms regardless of the path taken, and that
    the gemini-specific mode + fallback flag land in BOTH metadata levels.
    """

    def test_native_whole_pdf_conforms(self, native_processor, tmp_path):
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

        assert_conforms(out, [ExpectedDoc(rel_key="sample.pdf", pages=3, status="completed")])
        body = (out / "sample" / "sample.md").read_text()
        assert "P1 text" in body and "P3 text" in body

        doc_meta = json.loads((out / "sample" / "metadata.json").read_text())
        assert doc_meta["mode"] == "whole_pdf"
        assert doc_meta["fell_back_from_whole_pdf"] is False
        root_entry = json.loads((out / "metadata.json").read_text())["files"]["sample.pdf"]
        assert root_entry["mode"] == "whole_pdf"
        assert root_entry["fell_back_from_whole_pdf"] is False

    def test_auto_fallback_conforms_and_records_flag(self, native_processor, tmp_path):
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

        # Structure conforms identically to the whole-PDF path.
        assert_conforms(out, [ExpectedDoc(rel_key="sample.pdf", pages=3, status="completed")])

        doc_meta = json.loads((out / "sample" / "metadata.json").read_text())
        assert doc_meta["mode"] == "per_page"
        assert doc_meta["fell_back_from_whole_pdf"] is True
        root_entry = json.loads((out / "metadata.json").read_text())["files"]["sample.pdf"]
        assert root_entry["mode"] == "per_page"
        assert root_entry["fell_back_from_whole_pdf"] is True
