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


class TestBatchFailureResilience:
    """SYS-02: a failing file in a batch is recorded status=failed AND the rest
    of the batch still processes (one bad file never aborts the run). Covers the
    serial and concurrent paths and a pre-OCRResult (oversized) failure.
    """

    def test_oversized_file_does_not_abort_serial_batch(self, processor, tmp_path):
        # A 3-file batch: the middle file is oversized (pre-OCRResult failure).
        # It must be recorded status=failed and the other two must complete.
        root = tmp_path / "in"
        root.mkdir()
        _make_multipage_pdf(root / "a.pdf", n_pages=1)
        big = root / "big.pdf"
        _make_multipage_pdf(big, n_pages=1)
        # Pad big.pdf well past the size cap so validate_file_size fails.
        big.write_bytes(big.read_bytes() + b"\x00" * 2_000_000)
        _make_multipage_pdf(root / "c.pdf", n_pages=1)

        processor.config.max_file_size_mb = 1.0  # ~1MB cap; big.pdf exceeds it
        processor.client.models.generate_content.side_effect = lambda *a, **k: _mock_page_response(
            "ok"
        )

        out = tmp_path / "out"
        outcome = processor.process(root, output_path=out)

        # The whole batch was attempted: two completed, one failed (nonzero exit).
        assert outcome.exit_code != 0
        assert outcome.completed == 2
        assert outcome.failed == 1

        # Durable status=failed metadata exists for the oversized file AND
        # completed metadata for the others — all conform.
        assert_conforms(
            out,
            [
                ExpectedDoc(rel_key="a.pdf", pages=1, status="completed"),
                ExpectedDoc(rel_key="big.pdf", status="failed"),
                ExpectedDoc(rel_key="c.pdf", pages=1, status="completed"),
            ],
            require_failures_nonzero_exit=True,
        )
        # The failed file's error is captured in its durable metadata.
        big_meta = json.loads((out / "big" / "metadata.json").read_text())
        assert big_meta["status"] == "failed"
        assert "exceeds maximum" in big_meta["error"]
        # And the root index records it keyed by input-relative path.
        root_idx = json.loads((out / "metadata.json").read_text())["files"]
        assert root_idx["big.pdf"]["status"] == "failed"

    def test_unsupported_type_in_concurrent_batch_persists_failed(
        self, mock_config, mock_genai_client, tmp_path
    ):
        # The concurrent catch-all must _persist a FAILED record (not just count
        # it), so a worker that raises before building an OCRResult still leaves
        # durable per-doc + root metadata.
        mock_config.pdf_mode = "per_page"
        mock_config.max_workers = 2
        with patch("gemini_ocr.processor.genai") as mock_genai:
            mock_genai.Client.return_value = mock_genai_client
            proc = OCRProcessor(mock_config)
            proc.client = mock_genai_client

        root = tmp_path / "in"
        root.mkdir()
        _make_multipage_pdf(root / "good.pdf", n_pages=1)
        # process_file raises ValueError('Unsupported file type') for .xyz, which
        # the concurrent future surfaces as a pre-OCRResult exception.
        (root / "bad.xyz").write_bytes(b"not ocr-able")

        proc.client.models.generate_content.side_effect = lambda *a, **k: _mock_page_response("ok")

        out = tmp_path / "out"
        # bad.xyz is not a supported suffix, so discovery skips it; force it into
        # the batch by also dropping a supported-but-unprocessable file. Instead,
        # assert via a direct failing future: monkeypatch process_file to raise
        # for one rel_key.
        real_process_file = proc.process_file

        def flaky(file_path, *a, **k):
            if file_path.name == "good.pdf":
                return real_process_file(file_path, *a, **k)
            raise RuntimeError("worker blew up")

        # Add a second supported file that the flaky wrapper will fail.
        _make_multipage_pdf(root / "boom.pdf", n_pages=1)
        with patch.object(proc, "process_file", side_effect=flaky):
            outcome = proc.process(root, output_path=out)

        assert outcome.exit_code != 0
        assert outcome.failed == 1
        # Durable FAILED metadata for the worker-exception file.
        assert_conforms(
            out,
            [
                ExpectedDoc(rel_key="good.pdf", pages=1, status="completed"),
                ExpectedDoc(rel_key="boom.pdf", status="failed"),
            ],
            require_failures_nonzero_exit=True,
        )
        boom_meta = json.loads((out / "boom" / "metadata.json").read_text())
        assert boom_meta["status"] == "failed"
        assert "worker blew up" in boom_meta["error"]


class TestRunFingerprintInvalidation:
    """A re-run under a different model/mode reprocesses instead of reusing the
    cached output (run_fingerprint stamped in metadata + checked by is_completed).
    """

    def test_model_change_forces_reprocess(self, mock_config, mock_genai_client, tmp_path):
        mock_config.pdf_mode = "per_page"
        pdf = _make_multipage_pdf(tmp_path / "sample.pdf", n_pages=1)
        out = tmp_path / "out"

        def make_proc(model):
            mock_config.model = model
            with patch("gemini_ocr.processor.genai") as mg:
                mg.Client.return_value = mock_genai_client
                p = OCRProcessor(mock_config)
                p.client = mock_genai_client
                return p

        mock_genai_client.models.generate_content.side_effect = lambda *a, **k: _mock_page_response(
            "text"
        )

        p1 = make_proc("gemini-3-flash-preview")
        p1.process(pdf, output_path=out)
        calls_after_first = mock_genai_client.models.generate_content.call_count
        assert calls_after_first >= 1

        # Same model -> cache hit, no new OCR call.
        p_same = make_proc("gemini-3-flash-preview")
        p_same.process(pdf, output_path=out)
        assert mock_genai_client.models.generate_content.call_count == calls_after_first

        # Different model -> fingerprint mismatch -> reprocess (new OCR call).
        p2 = make_proc("gemini-3-pro")
        p2.process(pdf, output_path=out)
        assert mock_genai_client.models.generate_content.call_count > calls_after_first
        meta = json.loads((out / "sample" / "metadata.json").read_text())
        assert meta["model"] == "gemini-3-pro"
        assert meta["fingerprint"].startswith("fp:")


class TestImageDocConformance:
    """A single image input conforms under v0.1.1 stem+ext keying (<stem>_<ext>)."""

    def test_image_input_conforms(self, processor, tmp_path):
        from PIL import Image as _Image

        img = tmp_path / "scan.png"
        _Image.new("RGB", (50, 50), "white").save(img)
        out = tmp_path / "out"
        processor.client.models.generate_content.side_effect = lambda *a, **k: _mock_page_response(
            "image text"
        )

        outcome = processor.process(img, output_path=out)
        assert outcome.exit_code == 0
        # Harness resolves the doc dir via the contract's doc_dir_for, which maps
        # scan.png -> scan_png/scan.md and verifies inline image links resolve.
        assert_conforms(out, [ExpectedDoc(rel_key="scan.png", pages=1, status="completed")])
        assert (out / "scan_png" / "scan.md").exists()


class TestNativeAndFallbackConformance:
    """Native whole-PDF and auto-fallback both produce conforming output.

    Asserts the output STRUCTURE conforms regardless of the path taken, and that
    the gemini-specific mode + fallback flag land in BOTH metadata levels.
    """

    def test_native_whole_pdf_conforms(self, native_processor, tmp_path):
        pdf = _make_multipage_pdf(tmp_path / "sample.pdf", n_pages=3)
        out = tmp_path / "out"
        # One complete whole-PDF call: 3 page markers, STOP finish, terminal
        # completeness sentinel (so the AUTO oracle accepts it without fallback).
        native_processor.client.models.generate_content.return_value = _mock_page_response(
            "## Page 1\n\nP1 text\n\n## Page 2\n\nP2 text\n\n## Page 3\n\nP3 text"
            "\n\n<!-- OCR-END -->"
        )

        outcome = native_processor.process(pdf, output_path=out)
        assert outcome.exit_code == 0
        # Exactly one whole-PDF call (no fallback).
        assert native_processor.client.models.generate_content.call_count == 1

        assert_conforms(out, [ExpectedDoc(rel_key="sample.pdf", pages=3, status="completed")])
        body = (out / "sample" / "sample.md").read_text()
        assert "P1 text" in body and "P3 text" in body
        # The completeness sentinel never leaks into the saved body.
        assert "OCR-END" not in body

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


class TestUnreadableInputBatchResilience:
    """SYS-02 (round-2 HIGH): an unreadable/race-deleted input must be recorded
    status=failed and the rest of the batch must still process. The pre-filter
    checksum used to run OUTSIDE per-file isolation (bare ``sha256_checksum``), so
    a chmod-000 file mid-batch aborted the WHOLE run (zero output, zero failed
    records). The v0.1.2 ``safe_checksum`` wrap fixes it.
    """

    def test_chmod000_file_mid_batch_does_not_abort_others(self, processor, tmp_path):
        import os
        import stat

        root = tmp_path / "in"
        root.mkdir()
        _make_multipage_pdf(root / "a.pdf", n_pages=1)
        bad = root / "bad.pdf"
        _make_multipage_pdf(bad, n_pages=1)
        _make_multipage_pdf(root / "c.pdf", n_pages=1)

        processor.client.models.generate_content.side_effect = lambda *a, **k: _mock_page_response(
            "ok"
        )

        # Make the middle file unreadable AFTER discovery would see it.
        os.chmod(bad, 0)
        try:
            out = tmp_path / "out"
            outcome = processor.process(root, output_path=out)

            # The whole batch was attempted: the two good files completed, the
            # unreadable file is recorded failed, and the exit code is nonzero.
            assert outcome.exit_code != 0
            assert outcome.completed == 2
            assert outcome.failed == 1

            # A durable status=failed record exists for the unreadable file.
            bad_meta = json.loads((out / "bad" / "metadata.json").read_text())
            assert bad_meta["status"] == "failed"
            assert "unreadable" in bad_meta["error"].lower()

            # The good files conform and were NOT skipped.
            assert_conforms(
                out,
                [
                    ExpectedDoc(rel_key="a.pdf", pages=1, status="completed"),
                    ExpectedDoc(rel_key="c.pdf", pages=1, status="completed"),
                ],
            )
            root_idx = json.loads((out / "metadata.json").read_text())["files"]
            assert root_idx["bad.pdf"]["status"] == "failed"
        finally:
            os.chmod(bad, stat.S_IRUSR | stat.S_IWUSR)

    def test_single_unreadable_file_records_failed_not_abort(self, processor, tmp_path):
        import os
        import stat

        bad = tmp_path / "lonely.pdf"
        _make_multipage_pdf(bad, n_pages=1)
        os.chmod(bad, 0)
        try:
            out = tmp_path / "out"
            outcome = processor.process(bad, output_path=out)
            assert outcome.exit_code != 0
            assert outcome.failed == 1
            meta = json.loads((out / "lonely" / "metadata.json").read_text())
            assert meta["status"] == "failed"
            assert "unreadable" in meta["error"].lower()
            # v0.1.3: an unreadable-input failure record must carry a schema-valid
            # ``sha256:`` checksum (the UNREADABLE_CHECKSUM sentinel), NOT
            # None/""/the old "sha256:unavailable". Both metadata levels agree.
            from ocr_output_contract import UNREADABLE_CHECKSUM

            assert meta["checksum"] == UNREADABLE_CHECKSUM
            assert meta["checksum"].startswith("sha256:")
            root_entry = json.loads((out / "metadata.json").read_text())["files"]["lonely.pdf"]
            assert root_entry["checksum"] == UNREADABLE_CHECKSUM
            # The failure record must satisfy the contract's own harness.
            assert_conforms(out, [ExpectedDoc(rel_key="lonely.pdf", status="failed")])
        finally:
            os.chmod(bad, stat.S_IRUSR | stat.S_IWUSR)


class TestBlankInteriorPageNoNeedlessFallback:
    """round-2 MEDIUM: a legitimately-blank interior page (a marker the model
    omits) must NOT false-trigger a full per-page re-OCR. The v0.1.2 tail-aware
    ``is_truncated`` only fires when the END of the document is missing; gemini
    feeds it the recovered ``## Page N`` marker numbers so this works.
    """

    def test_blank_interior_page_keeps_whole_pdf_no_fallback(self, native_processor, tmp_path):
        pdf = _make_multipage_pdf(tmp_path / "sample.pdf", n_pages=3)
        out = tmp_path / "out"
        # Model omits the BLANK page 2 but stamps physical markers 1 and 3, stops
        # cleanly (STOP), AND emits the terminal sentinel. max(recovered={1,3})==3
        # ==actual -> not tail-truncated; sentinel present -> completeness OK; so
        # NO needless per-page fallback for a legitimately blank interior page.
        native_processor.client.models.generate_content.return_value = _mock_page_response(
            "## Page 1\n\nP1 text\n\n## Page 3\n\nP3 text\n\n<!-- OCR-END -->",
            finish_reason="STOP",
        )

        outcome = native_processor.process(pdf, output_path=out)
        assert outcome.exit_code == 0
        # Exactly one whole-PDF call: NO needless per-page fallback.
        assert native_processor.client.models.generate_content.call_count == 1

        doc_meta = json.loads((out / "sample" / "metadata.json").read_text())
        assert doc_meta["mode"] == "whole_pdf"
        assert doc_meta["fell_back_from_whole_pdf"] is False

        # Source page identity preserved: the body keeps ## Page 3, not a
        # silently-renumbered ## Page 2 (assemble_pages got the marker numbers).
        body = (out / "sample" / "sample.md").read_text()
        assert "## Page 1" in body
        assert "## Page 3" in body
        assert "## Page 2" not in body

    def test_dropped_tail_still_triggers_fallback(self, native_processor, tmp_path):
        # Contrast: a genuinely dropped TAIL (markers 1,2 of a 3-page doc) must
        # still fall back, so the tail check does not over-suppress.
        pdf = _make_multipage_pdf(tmp_path / "sample.pdf", n_pages=3)
        out = tmp_path / "out"

        first = {"done": False}

        def gen(*args, **kwargs):
            if not first["done"]:
                first["done"] = True
                # max(recovered={1,2})==2 < actual=3 -> truncated tail.
                return _mock_page_response(
                    "## Page 1\n\nP1\n\n## Page 2\n\nP2", finish_reason="STOP"
                )
            return _mock_page_response("recovered page text")

        native_processor.client.models.generate_content.side_effect = gen
        outcome = native_processor.process(pdf, output_path=out)
        assert outcome.exit_code == 0
        # 1 whole-PDF call + 3 per-page fallback calls.
        assert native_processor.client.models.generate_content.call_count == 4
        doc_meta = json.loads((out / "sample" / "metadata.json").read_text())
        assert doc_meta["mode"] == "per_page"
        assert doc_meta["fell_back_from_whole_pdf"] is True


class TestCrossModeFingerprintReprocess:
    """round-2 HIGH: the idempotency fingerprint must fold in the resolved
    output-affecting flags (``pdf_mode``, ``include_images``) so a cross-mode
    re-run reprocesses instead of silently reusing the cached result.
    """

    def _make_proc(self, mock_config, mock_genai_client, *, pdf_mode, include_images):
        mock_config.pdf_mode = pdf_mode
        mock_config.include_images = include_images
        with patch("gemini_ocr.processor.genai") as mg:
            mg.Client.return_value = mock_genai_client
            p = OCRProcessor(mock_config)
            p.client = mock_genai_client
            return p

    def test_whole_pdf_then_auto_reprocesses(self, mock_config, mock_genai_client, tmp_path):
        pdf = _make_multipage_pdf(tmp_path / "sample.pdf", n_pages=2)
        out = tmp_path / "out"
        # Complete whole-PDF response so the first (whole_pdf) run is COMPLETED
        # and would otherwise be a cache hit on the second run.
        mock_genai_client.models.generate_content.side_effect = lambda *a, **k: _mock_page_response(
            "## Page 1\n\nA\n\n## Page 2\n\nB", finish_reason="STOP"
        )

        p1 = self._make_proc(
            mock_config, mock_genai_client, pdf_mode="whole_pdf", include_images=True
        )
        p1.process(pdf, output_path=out)
        calls_after_first = mock_genai_client.models.generate_content.call_count
        assert calls_after_first >= 1

        # Re-run in the SAME mode -> cache hit, no new OCR call.
        p_same = self._make_proc(
            mock_config, mock_genai_client, pdf_mode="whole_pdf", include_images=True
        )
        p_same.process(pdf, output_path=out)
        assert mock_genai_client.models.generate_content.call_count == calls_after_first

        # Re-run in a DIFFERENT mode (auto) -> fingerprint mismatch -> reprocess,
        # so auto gets a chance to evaluate the response (and fall back if needed).
        p2 = self._make_proc(mock_config, mock_genai_client, pdf_mode="auto", include_images=True)
        p2.process(pdf, output_path=out)
        assert mock_genai_client.models.generate_content.call_count > calls_after_first
        meta = json.loads((out / "sample" / "metadata.json").read_text())
        assert meta["fingerprint"].startswith("fp:")

    def test_include_images_toggle_reprocesses(self, mock_config, mock_genai_client, tmp_path):
        pdf = _make_multipage_pdf(tmp_path / "sample.pdf", n_pages=1)
        out = tmp_path / "out"
        mock_genai_client.models.generate_content.side_effect = lambda *a, **k: _mock_page_response(
            "## Page 1\n\nA", finish_reason="STOP"
        )

        p1 = self._make_proc(
            mock_config, mock_genai_client, pdf_mode="whole_pdf", include_images=False
        )
        p1.process(pdf, output_path=out)
        calls_after_first = mock_genai_client.models.generate_content.call_count

        # Flip --include-images on: must reprocess (else figures stay missing).
        p2 = self._make_proc(
            mock_config, mock_genai_client, pdf_mode="whole_pdf", include_images=True
        )
        p2.process(pdf, output_path=out)
        assert mock_genai_client.models.generate_content.call_count > calls_after_first
