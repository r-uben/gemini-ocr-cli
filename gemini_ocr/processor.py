"""Core OCR processing module using Google Gemini.

PDFs are processed in one of three modes, all routed through the shared
``ocr-output-contract`` package (the family-wide output contract):

* **auto** (DEFAULT) — the whole PDF is uploaded once via the Gemini Files API
  and OCR'd in a *single* call. The prompt asks the model to begin each page with
  a ``## Page N`` header; the returned markdown is split on those markers to
  recover per-page text. If that whole-PDF response looks truncated (a
  length-limited finish reason, or fewer recovered pages than the PDF actually
  has -- see :func:`is_truncated`), the document is automatically re-processed
  page-by-page and that result is used instead, recorded non-silently in
  metadata. This is the cost/quality-preferred default for academic PDFs.
* **whole_pdf** (``--whole-pdf``) — a single whole-PDF call with NO fallback; the
  caller accepts that a long document may be truncated.
* **per_page** (``--per-page``) — each page is rendered to an image and OCR'd in
  its own call, then joined under ``## Page N`` headers. Honest per-page failure
  tracking at the cost of N calls per document.

All three produce byte-identical *structure*: one ``<stem>.md`` with ``## Page
N`` headers, no frontmatter, dual-level metadata, and the uniform exit policy.
This module owns *how OCR happens*; the ``ocr-output-contract`` package owns
*where bytes go*.
"""

import io
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from google import genai
from google.genai import types
from ocr_output_contract import (
    PAGE_MARKER_RE,
    TRUNCATION_FINISH_REASONS,
    DocMetadata,
    RootIndex,
    RunOutcome,
    Status,
    assemble_pages,
    doc_dir_for,
    failure_checksum,
    figure_filename,
    figure_markdown_link,
    figures_dir_for,
    is_truncated,
    markdown_path_for,
    relative_key,
    resolve_output_root,
    run_fingerprint,
    safe_checksum,
    sha256_checksum,
    split_native_pages,
    utc_timestamp,
    write_doc_metadata,
)
from PIL import Image
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

from gemini_ocr.config import Config
from gemini_ocr.utils import (
    extract_pdf_images,
    format_file_size,
    get_pdf_page_count,
    get_supported_files,
    is_image_file,
    is_pdf_file,
)

logger = logging.getLogger(__name__)

#: Backend identifier recorded in metadata. Gemini is a cloud API.
BACKEND = "gemini-api"

# Shared console instance — CLI sets .quiet on this directly
console = Console()

# OCR prompts for the per-page / single-image path. The contract package supplies
# the ``## Page N`` headers when assembling per-page results, so these prompts ask
# only for the page body.
OCR_PROMPTS = {
    "convert": """Convert this document into well-structured markdown.

- Maintain headings, paragraphs, lists, and tables (use markdown table format).
- Represent equations in LaTeX syntax.
- Preserve figure captions as [Figure N: <caption>]. Do not describe figure contents.
- Output only the resulting markdown, no commentary.""",
    "extract": """Extract all visible text from this document exactly as it appears.
Output only the extracted text, preserving line breaks and spacing.""",
    "describe_figure": """Analyze this figure/chart/diagram in detail:
1. What type of visualization is this? (bar chart, line graph, flowchart, etc.)
2. What are the axes, labels, or key components?
3. What data or information does it convey?
4. What are the main findings or takeaways?

Provide a structured description.""",
    "table": """Extract all tables from this document and convert them to markdown format.
Preserve all data, headers, and structure. Output only the markdown tables.""",
}

#: End-of-document sentinel the model is asked to emit as the FINAL line of a
#: native whole-PDF response, AFTER the last page's body. It is the completeness
#: signal for the AUTO default: a response that finishes with ``STOP`` but whose
#: body was cut mid-last-page will NOT carry this trailing marker, so its ABSENCE
#: is treated as truncation and triggers the per-page fallback (see the
#: completeness oracle in :meth:`_process_pdf_native`). It is an HTML comment so,
#: on the rare path where it survives into saved output, it renders as nothing; we
#: strip it from the body before splitting regardless (see
#: :func:`_strip_end_sentinel`).
_OCR_END_SENTINEL = "<!-- OCR-END -->"

#: Matches the end-of-document sentinel anywhere it appears, tolerant of
#: surrounding whitespace. Used both to detect completeness and to strip the
#: sentinel from the body so it never leaks into the saved markdown.
_OCR_END_SENTINEL_RE = re.compile(r"[ \t]*<!--\s*OCR-END\s*-->[ \t]*", re.IGNORECASE)

#: Instruction appended to native whole-PDF prompts so the single returned blob
#: carries explicit per-page boundaries we can split on. The model is asked to
#: begin EACH source page with a ``## Page N`` header (1-indexed, matching the
#: PDF's own page order), which we then parse back into per-page sections, AND to
#: emit a terminal completeness sentinel after the last page. The sentinel's
#: absence signals a cut-off (STOP-with-truncated-tail) response and drives the
#: AUTO fallback; we strip it from the saved body.
_NATIVE_PAGE_MARKER_INSTRUCTION = (
    "\n\nThis is a multi-page PDF. Process EVERY page, in order. Begin the "
    "transcription of each source page with a level-2 markdown header of the "
    'exact form "## Page N" on its own line, where N is the 1-indexed page '
    "number (## Page 1 for the first page, ## Page 2 for the second, and so on). "
    "Do not skip, merge, or renumber pages. After you have transcribed the LAST "
    f'page in full, emit the exact line "{_OCR_END_SENTINEL}" on its own line as '
    "the very last line of your output, to signal the document is complete. "
    "Output only the resulting markdown."
)

# OCR prompts for the native whole-PDF path. Identical to the per-page prompts but
# with the page-marker instruction appended so a single call yields page markers.
OCR_PROMPTS_NATIVE = {
    task: prompt + _NATIVE_PAGE_MARKER_INSTRUCTION for task, prompt in OCR_PROMPTS.items()
}

#: DPI used to rasterize PDF pages before sending them to Gemini (per-page mode).
PDF_RENDER_DPI = 200

#: Wall-clock deadline (seconds) for the Files-API upload to leave the PROCESSING
#: state. Native whole-PDF is the DEFAULT path, so an upload stuck in PROCESSING
#: would otherwise hang a worker (or the whole single-file run) forever with no
#: durable status=failed record. On timeout we raise so the per-file isolation
#: records status=failed and the batch continues (canon SYS-02).
UPLOAD_POLL_TIMEOUT_S = 300.0
#: Interval (seconds) between Files-API state polls.
UPLOAD_POLL_INTERVAL_S = 0.5

# --- PDF processing modes ---------------------------------------------------
#: Mode actually USED for a document (recorded in metadata). Distinct from the
#: config's requested mode, which may be ``auto``. The page-marker convention,
#: truncation-reason set, and the ``split_native_pages``/``is_truncated`` helpers
#: now live in the shared ``ocr-output-contract`` package (imported above).
MODE_WHOLE_PDF = "whole_pdf"
MODE_PER_PAGE = "per_page"


def _finish_reason_is_length_limited(finish_reason: Any) -> bool:
    """True if a finish_reason is a length/token-limit token (for messaging only).

    Mirrors the contract's internal normalization (``_normalize_finish_reason``
    is not part of the package's public API). Used to phrase the fallback notice;
    the actual fallback decision is made by the package's :func:`is_truncated`.
    """
    if finish_reason is None:
        return False
    name = getattr(finish_reason, "name", None)
    raw = name if isinstance(name, str) else str(finish_reason)
    token = re.sub(r"[^A-Za-z0-9]", "", raw).upper()
    return token in TRUNCATION_FINISH_REASONS


def _recover_page_numbers(text: str) -> list[int]:
    """Extract the physical page numbers from a native ``## Page N`` blob.

    Returns the integers captured by the contract's ``PAGE_MARKER_RE``, in the
    order the markers appear and aligned 1:1 with :func:`split_native_pages`
    output. These are fed to the contract's tail-aware :func:`is_truncated`
    (so a blank interior page is not a false truncation while a dropped tail
    still is) and to :func:`assemble_pages` (so a model-skipped page is not
    silently renumbered under ``--whole-pdf``).

    Falls back to ``[]`` when the model emitted no markers; callers treat that
    (one undivided page) by NOT passing page numbers, so the single page is
    labeled ``## Page 1`` as before.
    """
    return [int(m.group(1)) for m in PAGE_MARKER_RE.finditer(text or "")]


def _has_end_sentinel(text: str) -> bool:
    """True if the native response carries the end-of-document sentinel."""
    return _OCR_END_SENTINEL_RE.search(text or "") is not None


def _strip_end_sentinel(text: str) -> str:
    """Remove the end-of-document sentinel (and its line) from a native blob.

    Called BEFORE splitting/saving so the completeness marker never leaks into
    the persisted markdown body. Tolerant of surrounding whitespace and casing.
    """
    return _OCR_END_SENTINEL_RE.sub("", text or "")


@dataclass
class OCRResult:
    """Result from processing a document (one or more pages).

    ``pages`` holds the per-page markdown text in order. ``success`` is True only
    when every page succeeded; ``page_errors`` maps a 1-indexed page number to
    its error string for any page that failed (a partial document has both some
    text and some entries here).
    """

    file_path: Path
    pages: list[str]
    success: bool
    error: str | None = None
    processing_time: float = 0.0
    page_errors: dict[int, str] = field(default_factory=dict)
    extracted_images: list[dict[str, Any]] = field(default_factory=list)
    #: PDF mode actually used (``whole_pdf`` / ``per_page``); ``None`` for images.
    mode: str | None = None
    #: True when ``auto`` mode tried whole-PDF, detected truncation, and redid the
    #: document page-by-page. Recorded in metadata so the fallback is non-silent.
    fell_back_from_whole_pdf: bool = False
    #: Physical PDF page numbers recovered from the native ``## Page N`` markers,
    #: aligned 1:1 with ``pages``. Set only on the whole-PDF path; ``None`` for the
    #: per-page path (which already labels pages 1..N by construction). Passed to
    #: ``assemble_pages(page_numbers=...)`` so a model-skipped page is NOT silently
    #: renumbered under ``--whole-pdf``.
    recovered_page_numbers: list[int] | None = None

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def status(self) -> Status:
        """Map the result to a contract status enum.

        ``completed`` = every attempted page succeeded; ``partial`` = some pages
        succeeded and some failed; ``failed`` = no page produced usable text
        (every page failed, or the document could not be opened at all). Note
        that failed page slots carry a placeholder marker, so we count genuine
        successes via ``page_count - len(page_errors)`` rather than by scanning
        the (placeholder-filled) page text.
        """
        if self.success and not self.page_errors:
            return Status.COMPLETED
        succeeded = self.page_count - len(self.page_errors)
        if succeeded > 0:
            return Status.PARTIAL
        return Status.FAILED

    @property
    def text(self) -> str:
        """Backwards-compatible flat text view (pages joined with blank lines)."""
        return "\n\n".join(p for p in self.pages if p)


class OCRProcessor:
    """OCR processor using Google Gemini API, processing PDFs page by page."""

    def __init__(self, config: Config):
        """Initialize the OCR processor."""
        self.config = config
        config.validate_api_key()
        self.client = genai.Client(api_key=config.api_key)
        self.model_name = config.model
        self._lock = threading.Lock()
        #: Run-config fingerprint (model/backend/task/prompt) for the current
        #: ``process()`` call. Stamped into every DocMetadata so re-runs under a
        #: different model/mode/prompt invalidate the cache (see RootIndex.is_completed).
        self._run_fingerprint: str | None = None
        logger.info(f"Initialized OCRProcessor with model: {config.model}")

    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        """Check if an error is transient and worth retrying."""
        # Google GenAI SDK typed errors
        for exc_name in ("ResourceExhausted", "InternalServerError", "ServiceUnavailable"):
            if type(error).__name__ == exc_name:
                return True
        # httpx-level HTTP status errors
        if hasattr(error, "response"):
            status = getattr(error.response, "status_code", 0)
            if status in (429, 500, 502, 503, 504):
                return True
        # Network-level transient errors
        if isinstance(error, (TimeoutError, ConnectionError, OSError)):
            return True
        # Check error message for rate-limit indicators
        error_str = str(error).lower()
        return "429" in error_str or "rate limit" in error_str or "quota" in error_str

    # Gemini 3.x Flash models use thinking architecture and need explicit config
    # to avoid empty responses (thinking stalls at low temperature).
    # Does NOT match: gemini-2.x (different thinking API), gemini-3-pro (not Flash)
    _GEMINI_3_FLASH_RE = re.compile(r"gemini-3(?:\.\d+)?-flash")

    def _build_generation_config(self) -> types.GenerateContentConfig:
        """Build GenerateContentConfig, adding thinking config for Gemini 3 Flash models."""
        kwargs: dict[str, Any] = {"temperature": 0.1}

        if self._GEMINI_3_FLASH_RE.search(self.model_name):
            kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level="MINIMAL",
            )

        return types.GenerateContentConfig(**kwargs)

    @staticmethod
    def _extract_text(response: Any) -> tuple[str, Any]:
        """Extract ``(text, finish_reason)`` from a GenerateContentResponse.

        Walks ``candidates[0].content.parts`` explicitly: the ``.text`` shortcut
        returns None when parts include thought summaries, non-text parts, or when
        finish_reason != STOP, which is common with Gemini 3.x thinking models.

        The ``finish_reason`` is returned alongside the text so callers can detect
        length-limited truncation (used by native-mode auto-fallback). Raises
        ``RuntimeError`` on an empty extraction.
        """
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            feedback = getattr(response, "prompt_feedback", None)
            raise RuntimeError(f"Empty response: no candidates (prompt_feedback={feedback})")

        candidate = candidates[0]
        finish = getattr(candidate, "finish_reason", None)
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        text = "".join(
            p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False)
        ).strip()

        if not text:
            safety = getattr(candidate, "safety_ratings", None)
            part_types = [type(p).__name__ for p in parts]
            raise RuntimeError(
                f"Empty response: finish_reason={finish}, "
                f"len(parts)={len(parts)}, part_types={part_types}, "
                f"safety_ratings={safety}"
            )
        return text, finish

    def _call_with_retry_detailed(self, contents: list[Any], prompt: str) -> tuple[str, Any]:
        """Call generate_content with backoff; return ``(text, finish_reason)``."""
        max_attempts = self.config.max_retries + 1
        base_delay = self.config.retry_base_delay
        config = self._build_generation_config()

        for attempt in range(max_attempts):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[prompt, *contents],
                    config=config,
                )
                return self._extract_text(response)
            except Exception as e:
                is_last = attempt == max_attempts - 1
                if is_last or not self._is_retryable(e):
                    raise
                delay = base_delay * (2**attempt)
                logger.warning(
                    "Retryable error (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1,
                    max_attempts,
                    e,
                    delay,
                )
                time.sleep(delay)
        raise RuntimeError("Retry loop exited unexpectedly")

    def _call_with_retry(self, contents: list[Any], prompt: str) -> str:
        """Call generate_content with backoff; return just the extracted text."""
        text, _finish = self._call_with_retry_detailed(contents, prompt)
        return text

    def _upload_file(self, file_path: Path) -> Any:
        """Upload a file to the Gemini Files API (native whole-PDF mode)."""
        if self.config.verbose:
            console.print(f"[dim]Uploading {file_path.name}...[/dim]")

        uploaded = self.client.files.upload(file=str(file_path))

        # Bound the PROCESSING poll: an upload stuck in PROCESSING must not hang a
        # worker / single-file run forever on the now-default native path. On
        # timeout, delete the orphan remote object and raise so the per-file
        # failure path records status=failed and the batch continues (SYS-02).
        deadline = time.monotonic() + UPLOAD_POLL_TIMEOUT_S
        while uploaded.state == "PROCESSING":
            if time.monotonic() >= deadline:
                try:
                    self.client.files.delete(name=uploaded.name)
                except Exception as del_err:
                    logger.debug(f"Failed to delete stuck-upload remote file: {del_err}")
                raise RuntimeError(
                    f"File upload stuck in PROCESSING after {UPLOAD_POLL_TIMEOUT_S:.0f}s: "
                    f"{file_path.name}"
                )
            time.sleep(UPLOAD_POLL_INTERVAL_S)
            uploaded = self.client.files.get(name=uploaded.name)

        if uploaded.state == "FAILED":
            # The upload created a remote object even though processing failed.
            # Delete it here (the caller's finally never sees this orphan because
            # _upload_file did not return it) so it does not linger 48h.
            try:
                self.client.files.delete(name=uploaded.name)
            except Exception as del_err:
                logger.debug(f"Failed to delete failed-upload remote file: {del_err}")
            raise RuntimeError(f"File upload failed: {uploaded.name}")

        if self.config.verbose:
            console.print(f"[dim]Upload complete: {uploaded.name}[/dim]")

        return uploaded

    def _pil_to_part(self, image: Image.Image) -> types.Part:
        """Convert PIL Image to Gemini Part."""
        buffer = io.BytesIO()
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buffer, format="JPEG", quality=95)
        buffer.seek(0)
        return types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/jpeg")

    def _ocr_image(
        self,
        image: Image.Image,
        task: str,
        custom_prompt: str | None,
    ) -> str:
        """Run OCR on a single in-memory image and return its markdown text."""
        prompt = custom_prompt or OCR_PROMPTS.get(task, OCR_PROMPTS["convert"])
        image_part = self._pil_to_part(image)
        return self._call_with_retry([image_part], prompt)

    def process_image(
        self,
        image_path: Path,
        task: str = "convert",
        custom_prompt: str | None = None,
    ) -> OCRResult:
        """Process a single image file as a one-page document."""
        start_time = time.time()
        try:
            self.config.validate_file_size(image_path)
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            text = self._ocr_image(image, task, custom_prompt)
            return OCRResult(
                file_path=image_path,
                pages=[text],
                success=True,
                processing_time=time.time() - start_time,
            )
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return OCRResult(
                file_path=image_path,
                pages=[],
                success=False,
                error=str(e),
                processing_time=time.time() - start_time,
                page_errors={1: str(e)},
            )

    def process_pdf(
        self,
        pdf_path: Path,
        task: str = "convert",
        custom_prompt: str | None = None,
        show_progress: bool = True,
        mode: str | None = None,
        per_page: bool | None = None,
    ) -> OCRResult:
        """Process a PDF in one of three modes (auto / whole_pdf / per_page).

        ``mode`` selects the path; when ``None`` it falls back to
        ``self.config.pdf_mode`` (default ``"auto"``). For backwards/test
        convenience ``per_page=True`` forces ``per_page`` and ``per_page=False``
        forces ``whole_pdf`` (overriding ``mode``).

        * ``auto`` — whole-PDF (one call); if the response is truncated (see
          :func:`is_truncated`) the document is automatically re-processed
          page-by-page and that result is used, with the fallback recorded in
          metadata (non-silent).
        * ``whole_pdf`` — whole-PDF, no fallback (caller accepts truncation risk).
        * ``per_page`` — page-by-page always.
        """
        if per_page is True:
            effective = MODE_PER_PAGE
        elif per_page is False:
            effective = MODE_WHOLE_PDF
        else:
            effective = mode if mode is not None else self.config.pdf_mode

        if effective == MODE_PER_PAGE:
            return self._process_pdf_per_page(
                pdf_path, task=task, custom_prompt=custom_prompt, show_progress=show_progress
            )
        # whole_pdf and auto both start with a single whole-PDF call.
        allow_fallback = effective != MODE_WHOLE_PDF
        return self._process_pdf_native(
            pdf_path,
            task=task,
            custom_prompt=custom_prompt,
            show_progress=show_progress,
            allow_fallback=allow_fallback,
        )

    def _process_pdf_native(
        self,
        pdf_path: Path,
        task: str = "convert",
        custom_prompt: str | None = None,
        show_progress: bool = True,
        allow_fallback: bool = False,
    ) -> OCRResult:
        """Process a PDF natively: one upload + one Gemini call for the document.

        The whole PDF is uploaded via the Files API and OCR'd in a single call.
        The prompt asks the model to begin each page with a ``## Page N`` header;
        the returned markdown is split on those markers (see
        :func:`split_native_pages`) to recover per-page text for the contract.

        When ``allow_fallback`` is True (the ``auto`` default) and the response is
        truncated (length-limited finish reason, or fewer recovered pages than the
        PDF actually has), the document is automatically re-processed page-by-page
        and that result is returned instead, flagged ``fell_back_from_whole_pdf``.

        Failure policy (canon SYS-02): if the call errors or returns empty text,
        the result is ``FAILED`` (no pages, ``status=failed`` recorded). When the
        single-call response carries no ``## Page N`` markers the whole blob is
        kept as one page so content is never silently dropped.
        """
        start_time = time.time()

        # Native prompt: a single whole-PDF call MUST yield ``## Page N`` markers
        # so the response can be split back into per-page sections. A custom
        # prompt is honored but the page-marker instruction is still appended
        # (otherwise the model returns one undivided blob and a multi-page PDF is
        # silently recorded as ONE page); the task default is already augmented.
        if custom_prompt is not None:
            prompt = custom_prompt + _NATIVE_PAGE_MARKER_INSTRUCTION
        else:
            prompt = OCR_PROMPTS_NATIVE.get(task, OCR_PROMPTS_NATIVE["convert"])

        uploaded_file = None
        try:
            # Size validation lives INSIDE the try so an oversized PDF returns a
            # FAILED OCRResult (recorded status=failed) instead of raising out of
            # the batch loop and aborting the run (canon SYS-02: durable failure).
            self.config.validate_file_size(pdf_path)

            # Actual page count is a truncation signal (recovered < real pages).
            actual_pages = 0
            try:
                actual_pages = get_pdf_page_count(pdf_path)
            except Exception as e:
                logger.debug(f"Could not read page count for {pdf_path.name}: {e}")

            if show_progress and not self.config.quiet:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                    transient=True,
                ) as progress:
                    progress.add_task("Uploading PDF...", total=None)
                    uploaded_file = self._upload_file(pdf_path)
                    progress.update(progress.task_ids[0], description="Processing...")
                    text, finish_reason = self._call_with_retry_detailed([uploaded_file], prompt)
            else:
                uploaded_file = self._upload_file(pdf_path)
                text, finish_reason = self._call_with_retry_detailed([uploaded_file], prompt)

            # Completeness sentinel: the model is asked to emit a terminal
            # ``<!-- OCR-END -->`` line after the LAST page. Capture its presence
            # from the RAW response, then strip it so it never leaks into the
            # saved markdown body (split/assemble operate on sentinel-free text).
            has_sentinel = _has_end_sentinel(text)
            text = _strip_end_sentinel(text)

            pages = split_native_pages(text)
            if not pages:
                # Empty document output is a failure per the contract.
                return OCRResult(
                    file_path=pdf_path,
                    pages=[],
                    success=False,
                    error="Empty response from Gemini (no text returned)",
                    processing_time=time.time() - start_time,
                    mode=MODE_WHOLE_PDF,
                )

            # Recover the PHYSICAL page numbers the model stamped in its markers
            # (aligned 1:1 with ``pages``). These drive the tail-aware truncation
            # check below and preserve source page identity on the whole-PDF path.
            recovered = _recover_page_numbers(text)
            # Only use recovered numbers when they align 1:1 with the recovered
            # pages (markers present and counts match). A marker-less blob yields
            # [], and the rare split/marker count mismatch falls back to 1..N
            # labeling rather than risking a misaligned page_numbers list.
            recovered_page_numbers = recovered if len(recovered) == len(pages) else None

            # AUTO completeness oracle (the residual HIGH this PR closes). The
            # contract's tail-aware ``is_truncated`` catches the length-limited
            # finish reason and the dropped-tail-WITH-marker-shortfall case, but
            # silently accepts two genuine data-loss modes the contract docstring
            # explicitly defers to an upstream sentinel:
            #   1. STOP-with-cut-last-page: the model finishes (STOP) but the body
            #      was cut mid-last-page; ``max(recovered) == actual_pages`` so the
            #      page signal stays silent. The model never reaches the terminal
            #      sentinel, so its ABSENCE flags this as truncation.
            #   2. Markerless multi-page collapse: a multi-page PDF (actual_pages>1)
            #      whose response carries ZERO ``## Page N`` markers collapses to a
            #      single ``## Page 1`` blob. Recovered is empty, the page signal is
            #      disabled, and ``is_truncated`` returns False -- so we must treat
            #      a markerless multi-page response as truncation directly.
            markerless_multipage = actual_pages > 1 and len(recovered) == 0
            sentinel_missing = not has_sentinel
            completeness_truncated = sentinel_missing or markerless_multipage

            # Auto-fallback: a truncated whole-PDF response means content (usually
            # the TAIL) was dropped. ``is_truncated`` catches length-limited /
            # dropped-tail; the completeness oracle above catches STOP-with-cut-
            # tail and markerless collapse. Either fires the per-page re-OCR.
            contract_truncated = is_truncated(
                finish_reason,
                len(pages),
                actual_pages,
                recovered_page_numbers=recovered_page_numbers,
            )
            if allow_fallback and (contract_truncated or completeness_truncated):
                if _finish_reason_is_length_limited(finish_reason):
                    reason = "length-limited finish reason"
                elif markerless_multipage:
                    reason = f"no per-page markers for a {actual_pages}-page PDF"
                elif sentinel_missing:
                    reason = "missing end-of-document sentinel (tail likely cut)"
                else:
                    reason = f"recovered {len(pages)} of {actual_pages} page(s)"
                if not self.config.quiet:
                    console.print(
                        f"[yellow]Whole-PDF output truncated ({reason}); "
                        f"falling back to per-page for {pdf_path.name}[/yellow]"
                    )
                logger.info("Auto-fallback to per-page for %s (%s)", pdf_path.name, reason)
                fallback = self._process_pdf_per_page(
                    pdf_path, task=task, custom_prompt=custom_prompt, show_progress=show_progress
                )
                fallback.fell_back_from_whole_pdf = True
                fallback.processing_time = time.time() - start_time
                return fallback

            extracted_images: list[dict[str, Any]] = []
            if self.config.include_images:
                try:
                    extracted_images = extract_pdf_images(pdf_path)
                except Exception as e:
                    logger.warning(f"Failed to extract embedded images: {e}")

            return OCRResult(
                file_path=pdf_path,
                pages=pages,
                success=True,
                processing_time=time.time() - start_time,
                extracted_images=extracted_images,
                mode=MODE_WHOLE_PDF,
                recovered_page_numbers=recovered_page_numbers,
            )
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            return OCRResult(
                file_path=pdf_path,
                pages=[],
                success=False,
                error=str(e),
                processing_time=time.time() - start_time,
                mode=MODE_WHOLE_PDF,
            )
        finally:
            # Clean up the uploaded file from the Files API (48h retention).
            if uploaded_file is not None:
                try:
                    self.client.files.delete(name=uploaded_file.name)
                except Exception as del_err:
                    logger.debug(f"Failed to delete uploaded file: {del_err}")

    def _process_pdf_per_page(
        self,
        pdf_path: Path,
        task: str = "convert",
        custom_prompt: str | None = None,
        show_progress: bool = True,
    ) -> OCRResult:
        """Process a PDF page by page, rendering each page and OCR'ing it.

        Each page is an independent Gemini call so page boundaries and per-page
        failures are real. A page that fails is recorded in ``page_errors`` and
        emits an explicit failure marker in its slot, so the page count and
        ``## Page N`` numbering stay aligned with the source PDF.
        """
        start_time = time.time()

        try:
            # Size validation is INSIDE the try so an oversized PDF returns a
            # FAILED OCRResult (recorded status=failed) rather than raising out of
            # the batch loop and aborting the run (canon SYS-02: durable failure).
            self.config.validate_file_size(pdf_path)
            doc = fitz.open(pdf_path)
        except Exception as e:
            logger.error(f"Error opening {pdf_path}: {e}")
            return OCRResult(
                file_path=pdf_path,
                pages=[],
                success=False,
                error=str(e),
                processing_time=time.time() - start_time,
                mode=MODE_PER_PAGE,
            )

        try:
            num_pages = len(doc)
            pages: list[str] = []
            page_errors: dict[int, str] = {}
            zoom = PDF_RENDER_DPI / 72.0
            matrix = fitz.Matrix(zoom, zoom)

            progress_ctx = (
                Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    console=console,
                    transient=True,
                )
                if (show_progress and not self.config.quiet)
                else None
            )

            def run() -> None:
                task_id = None
                if progress_ctx is not None:
                    task_id = progress_ctx.add_task(f"OCR {pdf_path.name}", total=num_pages)
                for page_idx in range(num_pages):
                    page_no = page_idx + 1
                    try:
                        pix = doc[page_idx].get_pixmap(matrix=matrix)
                        image = Image.open(io.BytesIO(pix.tobytes("png")))
                        text = self._ocr_image(image, task, custom_prompt)
                        pages.append(text)
                    except Exception as page_err:
                        logger.error(f"Page {page_no} of {pdf_path.name} failed: {page_err}")
                        page_errors[page_no] = str(page_err)
                        pages.append(f"*[OCR failed for page {page_no}]*")
                    if progress_ctx is not None and task_id is not None:
                        progress_ctx.advance(task_id)

            if progress_ctx is not None:
                with progress_ctx:
                    run()
            else:
                run()

            extracted_images: list[dict[str, Any]] = []
            if self.config.include_images:
                try:
                    extracted_images = extract_pdf_images(pdf_path)
                except Exception as e:
                    logger.warning(f"Failed to extract embedded images: {e}")

            success = not page_errors
            error = None if success else f"{len(page_errors)} of {num_pages} page(s) failed"
            return OCRResult(
                file_path=pdf_path,
                pages=pages,
                success=success,
                error=error,
                processing_time=time.time() - start_time,
                page_errors=page_errors,
                extracted_images=extracted_images,
                mode=MODE_PER_PAGE,
            )
        finally:
            doc.close()

    def process_file(
        self,
        file_path: Path,
        task: str = "convert",
        custom_prompt: str | None = None,
        show_progress: bool = True,
        per_page: bool | None = None,
    ) -> OCRResult:
        """Process a single file (image or PDF).

        ``per_page`` selects the PDF processing mode (``None`` = config default).
        It is ignored for single images (always a one-call, one-page document).
        """
        if is_pdf_file(file_path):
            return self.process_pdf(
                file_path,
                task=task,
                custom_prompt=custom_prompt,
                show_progress=show_progress,
                per_page=per_page,
            )
        elif is_image_file(file_path):
            return self.process_image(file_path, task=task, custom_prompt=custom_prompt)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

    # ------------------------------------------------------------------
    # Output writing (all routed through the ocr-output-contract package)
    # ------------------------------------------------------------------

    def save_results(
        self,
        result: OCRResult,
        output_root: Path,
        rel_key: str,
    ) -> Path:
        """Write the aggregated markdown + figures for one document.

        The layout is determined entirely by the contract package:
        ``<output_root>/<rel/dir>/<stem>/<stem>.md`` plus a ``figures/`` folder.
        Figures are normalised to PNG and named ``figure_<N>_page<P>.png`` with
        resolving links appended to the markdown.
        """
        doc_dir = doc_dir_for(output_root, rel_key)
        doc_dir.mkdir(parents=True, exist_ok=True)
        markdown_path = markdown_path_for(doc_dir, rel_key)

        # Build the clean markdown body: pages under ## Page N, no frontmatter.
        # On the whole-PDF path, pass the recovered PHYSICAL page numbers so a
        # model-skipped page (e.g. markers 1,3 with 2 omitted) is labeled by its
        # source number instead of being silently renumbered to 1..N. The per-page
        # path leaves recovered_page_numbers=None and is labeled 1..N as before.
        if result.pages:
            page_numbers = result.recovered_page_numbers
            if page_numbers is not None and len(page_numbers) == len(result.pages):
                body = assemble_pages(result.pages, page_numbers=page_numbers)
            else:
                body = assemble_pages(result.pages)
        else:
            body = "*[OCR Failed]*\n"

        # Save & link extracted figures (normalised to PNG).
        figure_links = self._save_figures(result, doc_dir)
        if figure_links:
            body = body.rstrip("\n") + "\n\n## Figures\n\n" + "\n\n".join(figure_links) + "\n"

        markdown_path.write_text(body, encoding="utf-8")

        if self.config.verbose:
            console.print(f"[green]Saved:[/green] {markdown_path}")

        return markdown_path

    def _save_figures(self, result: OCRResult, doc_dir: Path) -> list[str]:
        """Persist extracted images as PNG and return resolving markdown links."""
        if not (result.extracted_images and self.config.include_images):
            return []

        figures_dir = figures_dir_for(doc_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        links: list[str] = []
        figure_counter = 0
        for img_info in result.extracted_images:
            figure_counter += 1
            page_no = int(img_info.get("page", 1))
            filename = figure_filename(figure_counter, page_no)
            img_path = figures_dir / filename
            try:
                image = Image.open(io.BytesIO(img_info["data"]))
                if image.mode not in ("RGB", "RGBA"):
                    image = image.convert("RGB")
                image.save(img_path, format="PNG")
                links.append(figure_markdown_link(figure_counter, page_no))
            except Exception as e:
                logger.warning(f"Failed to save figure {figure_counter} (page {page_no}): {e}")
                figure_counter -= 1
        return links

    def _build_doc_metadata(
        self,
        result: OCRResult,
        file_path: Path,
        markdown_path: Path,
        output_root: Path,
    ) -> DocMetadata:
        """Assemble the per-document metadata record from a result."""
        status = result.status
        error = None
        if status is not Status.COMPLETED:
            if result.page_errors:
                error = "; ".join(
                    f"page {n}: {msg}" for n, msg in sorted(result.page_errors.items())
                )
            elif result.error:
                error = result.error
        if status is Status.COMPLETED:
            # A completed doc was definitely readable; record its real digest.
            checksum = sha256_checksum(file_path)
        else:
            # Failure/partial record: the input may be unreadable (the failure
            # WAS an I/O error). Use the contract's failure_checksum so the entry
            # still carries a schema-valid ``sha256:`` checksum -- the real digest
            # if readable, else the canonical UNREADABLE_CHECKSUM sentinel --
            # instead of the old non-conforming "sha256:unavailable" / None / "".
            checksum = failure_checksum(file_path)
        return DocMetadata(
            status=status,
            checksum=checksum,
            model=self.model_name,
            backend=BACKEND,
            processing_time=result.processing_time,
            timestamp=utc_timestamp(),
            output_path=str(markdown_path.relative_to(output_root)),
            pages=result.page_count,
            error=error,
            fingerprint=self._run_fingerprint,
            mode=result.mode,
            fell_back_from_whole_pdf=result.fell_back_from_whole_pdf,
        )

    def _persist(
        self,
        result: OCRResult,
        file_path: Path,
        output_root: Path,
        rel_key: str,
        index: RootIndex,
    ) -> tuple[DocMetadata, Path]:
        """Write markdown, figures, and BOTH metadata levels for one document.

        Always writes output (markdown + per-doc + root metadata) regardless of
        success, so failures are recorded with ``status=failed`` per the canon.
        Returns ``(metadata, markdown_path)`` (caller maps status to outcome).
        """
        markdown_path = self.save_results(result, output_root, rel_key)
        meta = self._build_doc_metadata(result, file_path, markdown_path, output_root)
        doc_dir = doc_dir_for(output_root, rel_key)
        write_doc_metadata(doc_dir, rel_key, meta)
        with self._lock:
            index.record(rel_key, meta)
        return meta, markdown_path

    def _persist_failure(
        self,
        error: str,
        file_path: Path,
        output_root: Path,
        rel_key: str,
        index: RootIndex,
    ) -> DocMetadata:
        """Persist durable ``status=failed`` metadata for a pre-OCRResult error.

        Covers failures that occur BEFORE an :class:`OCRResult` could be built
        (an oversized file slipping through, an unsupported type, or any
        exception raised inside ``process_file`` / a worker future). Without this
        the canon's SYS-02 promise ("every attempted file leaves a status=failed
        record so 'which of my 500 PDFs failed?' is answerable") would be broken
        for that whole class of failures. Mirrors a FAILED OCRResult through the
        normal :meth:`_persist` path so both metadata levels stay in sync.

        A checksum is recorded when the file is still readable; if even that
        fails the doc is still recorded (sentinel checksum) so it is never lost.
        """
        synthesized = OCRResult(
            file_path=file_path,
            pages=[],
            success=False,
            error=error,
        )
        return self._persist(synthesized, file_path, output_root, rel_key, index)[0]

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def process(
        self,
        input_path: Path,
        output_path: Path | None = None,
        task: str = "convert",
        custom_prompt: str | None = None,
        reprocess: bool = False,
    ) -> RunOutcome:
        """Process input path (file or directory). Returns a RunOutcome.

        The returned :class:`RunOutcome` carries the uniform exit policy: nonzero
        if any file or page failed, across both single-file and batch runs.
        """
        # Stamp the run-config fingerprint so a re-run under a different
        # model/backend/task/prompt OR a different output-affecting flag
        # invalidates the cache and reprocesses, instead of silently reusing
        # output keyed only on the input checksum. ``extra`` carries the RESOLVED
        # effective flags that change what output an input produces:
        #   * ``pdf_mode`` — a completed/truncated ``--whole-pdf`` run must NOT be
        #     skipped on a later default ``auto`` run (auto would then never get a
        #     chance to fall back to per-page).
        #   * ``include_images`` — a ``--no-images`` run must NOT be skipped on a
        #     later ``--include-images`` run (figures would stay permanently
        #     missing). Single images are unaffected by pdf_mode but the flag set
        #     is uniform per run, so both are always folded in.
        self._run_fingerprint = run_fingerprint(
            self.model_name,
            BACKEND,
            task=task,
            prompt=custom_prompt,
            extra={
                "pdf_mode": self.config.pdf_mode,
                "include_images": self.config.include_images,
            },
        )
        if input_path.is_file():
            return self._process_single_file(
                input_path, output_path, task, custom_prompt, reprocess
            )
        elif input_path.is_dir():
            return self._process_directory(input_path, output_path, task, custom_prompt, reprocess)
        else:
            raise ValueError(f"Input path does not exist: {input_path}")

    def _process_single_file(
        self,
        file_path: Path,
        output_path: Path | None,
        task: str,
        custom_prompt: str | None,
        reprocess: bool,
    ) -> RunOutcome:
        """Process a single file. Scan root is the file's parent (rel key = name)."""
        outcome = RunOutcome()
        output_root = resolve_output_root(file_path, output_path)
        output_root.mkdir(parents=True, exist_ok=True)
        scan_root = file_path.parent
        rel_key = relative_key(file_path, scan_root)
        index = RootIndex(output_root)

        # Checksum via safe_checksum: an unreadable/race-deleted input must be
        # recorded status=failed (durable), not raise OSError out of the run
        # (canon SYS-02). None => skip the idempotency check, fall through, and
        # let the per-file failure path persist a failed record below.
        checksum = safe_checksum(file_path)
        if checksum is None:
            error = f"Input file is unreadable: {file_path}"
            logger.error(error)
            self._persist_failure(error, file_path, output_root, rel_key, index)
            outcome.add(Status.FAILED, detail=rel_key)
            console.print(f"\n[red]Failed:[/red] {error}")
            return outcome

        if not reprocess and index.is_completed(
            rel_key, checksum, fingerprint=self._run_fingerprint
        ):
            console.print(f"[yellow]Already processed:[/yellow] {file_path.name}")
            console.print("[dim]Use --reprocess to force reprocessing[/dim]")
            # Emit the cached output path so -q still prints it (scripting contract).
            markdown_path = markdown_path_for(doc_dir_for(output_root, rel_key), rel_key)
            outcome.add(Status.COMPLETED, output_path=str(markdown_path))
            return outcome

        console.print(f"[blue]Processing:[/blue] {file_path}")
        console.print(f"[blue]Output:[/blue] {output_root}\n")

        try:
            result = self.process_file(file_path, task=task, custom_prompt=custom_prompt)
            meta, markdown_path = self._persist(result, file_path, output_root, rel_key, index)
            outcome.add(
                meta.status,
                detail=None if meta.status is Status.COMPLETED else rel_key,
                output_path=str(markdown_path),
            )
            if meta.status is Status.COMPLETED:
                console.print("\n[green]Success[/green]")
                console.print(f"[dim]Time: {result.processing_time:.2f}s[/dim]")
            else:
                console.print(f"\n[red]Failed ({meta.status.value}):[/red] {meta.error}")
        except Exception as e:
            # Pre-OCRResult exception (or a _persist failure): record durable
            # status=failed metadata rather than letting it abort the run.
            logger.error(f"Error processing {rel_key}: {e}")
            self._persist_failure(str(e), file_path, output_root, rel_key, index)
            outcome.add(Status.FAILED, detail=rel_key)
            console.print(f"\n[red]Failed:[/red] {e}")
        return outcome

    def _process_directory(
        self,
        dir_path: Path,
        output_path: Path | None,
        task: str,
        custom_prompt: str | None,
        reprocess: bool,
    ) -> RunOutcome:
        """Process all files in a directory, keyed on input-relative paths."""
        outcome = RunOutcome()

        # Resolve the output root FIRST so discovery can exclude it (a default
        # root sits inside the scanned tree as <input>/ocr/; without this the
        # engine would re-ingest its own .md/figure outputs on a re-run).
        output_root = resolve_output_root(dir_path, output_path)
        output_root.mkdir(parents=True, exist_ok=True)

        files = get_supported_files(dir_path, output_root)
        if not files:
            console.print("[yellow]No supported files found[/yellow]")
            return outcome

        index = RootIndex(output_root)

        # Filter already-processed files (keyed by input-relative path).
        files_to_process: list[tuple[Path, str]] = []
        for f in files:
            rel_key = relative_key(f, dir_path)
            # safe_checksum tolerates an input that became unreadable between
            # discovery and this pre-filter (permission denied, race-deleted,
            # broken symlink). On None we record a durable status=failed for THIS
            # file and CONTINUE the batch, instead of letting OSError abort the
            # whole run with zero output (canon SYS-02: one bad file never aborts
            # the batch). The checksum was previously computed OUTSIDE the
            # per-file try, so an unreadable mid-batch file killed everything.
            checksum = safe_checksum(f)
            if checksum is None:
                error = f"Input file is unreadable: {f}"
                logger.error(error)
                self._persist_failure(error, f, output_root, rel_key, index)
                outcome.add(Status.FAILED, detail=rel_key)
                console.print(f"  [red]ERROR: {error}[/red]\n")
                continue
            if not reprocess and index.is_completed(
                rel_key, checksum, fingerprint=self._run_fingerprint
            ):
                if self.config.verbose:
                    console.print(f"[dim]Skipping: {rel_key}[/dim]")
                # Emit the cached path so -q still lists it (scripting contract).
                markdown_path = markdown_path_for(doc_dir_for(output_root, rel_key), rel_key)
                outcome.add(Status.COMPLETED, output_path=str(markdown_path))
            else:
                files_to_process.append((f, rel_key))

        if not files_to_process:
            # Nothing left to OCR. Distinguish "all cached" from "the only files
            # were unreadable and already recorded status=failed in the pre-filter".
            if outcome.has_failures:
                console.print("[red]No processable files (unreadable inputs recorded failed)[/red]")
            else:
                console.print("[green]All files already processed[/green]")
                console.print("[dim]Use --reprocess to force reprocessing[/dim]")
            return outcome

        console.print(f"[blue]Processing {len(files_to_process)} file(s)...[/blue]")
        console.print(f"[blue]Output:[/blue] {output_root}\n")

        start_time = time.time()

        if self.config.max_workers > 1:
            self._process_directory_concurrent(
                files_to_process, output_root, index, task, custom_prompt, outcome
            )
        else:
            for file_path, rel_key in files_to_process:
                # Each file is isolated: a pre-OCRResult exception (oversized file,
                # unsupported type, _persist error, OR a now-race-deleted input
                # whose stat() raises) is recorded as status=failed and the loop
                # CONTINUES so one bad file never aborts the batch (canon SYS-02).
                # The stat()/size print lives INSIDE the try for that reason.
                try:
                    file_size = format_file_size(file_path.stat().st_size)
                    console.print(f"[cyan]{rel_key}[/cyan] ({file_size})")
                    result = self.process_file(
                        file_path, task=task, custom_prompt=custom_prompt, show_progress=False
                    )
                    meta, _ = self._persist(result, file_path, output_root, rel_key, index)
                    outcome.add(
                        meta.status,
                        detail=None if meta.status is Status.COMPLETED else rel_key,
                        output_path=str(
                            markdown_path_for(doc_dir_for(output_root, rel_key), rel_key)
                        ),
                    )
                    if meta.status is Status.COMPLETED:
                        console.print(f"  [green]OK[/green] ({result.processing_time:.1f}s)\n")
                    else:
                        console.print(f"  [red]{meta.status.value.upper()}: {meta.error}[/red]\n")
                except Exception as e:
                    logger.error(f"Error processing {rel_key}: {e}")
                    self._persist_failure(str(e), file_path, output_root, rel_key, index)
                    outcome.add(Status.FAILED, detail=rel_key)
                    console.print(f"  [red]ERROR: {e}[/red]\n")

        total_time = time.time() - start_time
        console.print(
            f"\n[green]Completed:[/green] {outcome.completed}/"
            f"{outcome.completed + outcome.failed + outcome.partial} files"
        )
        if outcome.has_failures:
            console.print(
                f"[red]Failures:[/red] {outcome.failed} failed, {outcome.partial} partial"
            )
        console.print(f"[dim]Total time: {total_time:.2f}s[/dim]")
        return outcome

    def _process_directory_concurrent(
        self,
        files: list[tuple[Path, str]],
        output_root: Path,
        index: RootIndex,
        task: str,
        custom_prompt: str | None,
        outcome: RunOutcome,
    ) -> None:
        """Process files concurrently with a thread pool."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        ) as progress:
            progress_task = progress.add_task("Processing...", total=len(files))

            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_meta = {
                    executor.submit(self.process_file, f, task, custom_prompt, False): (f, rel_key)
                    for f, rel_key in files
                }

                for future in as_completed(future_to_meta):
                    file_path, rel_key = future_to_meta[future]
                    try:
                        result = future.result()
                        meta, markdown_path = self._persist(
                            result, file_path, output_root, rel_key, index
                        )
                        outcome.add(
                            meta.status,
                            detail=None if meta.status is Status.COMPLETED else rel_key,
                            output_path=str(markdown_path),
                        )
                        if meta.status is Status.COMPLETED:
                            console.print(
                                f"  [green]OK[/green] {rel_key} ({result.processing_time:.1f}s)"
                            )
                        else:
                            console.print(
                                f"  [red]{meta.status.value.upper()}[/red] {rel_key}: {meta.error}"
                            )
                    except Exception as e:
                        # Pre-OCRResult failure (the worker raised, or _persist
                        # itself failed): persist durable status=failed metadata so
                        # the doc is not silently dropped (canon SYS-02), matching
                        # the serial path's behavior. _persist_failure is best-effort.
                        console.print(f"  [red]ERROR[/red] {rel_key}: {e}")
                        try:
                            self._persist_failure(str(e), file_path, output_root, rel_key, index)
                        except Exception as persist_err:
                            logger.error(
                                f"Failed to persist failure metadata for {rel_key}: {persist_err}"
                            )
                        outcome.add(Status.FAILED, detail=rel_key)

                    progress.advance(progress_task)
