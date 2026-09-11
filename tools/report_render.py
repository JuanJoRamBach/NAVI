"""
tools/report_render.py

Renders RESEARCH_EXECUTE_PLAN.md's JSON-schema output into the real
finished deliverable — the docx template (tools/templates/
research_report.docx, see tools/build_report_templates.py) holds every
design decision; this module only fills in data, never makes a layout
choice itself.

`render_research_report_docx` -> bytes is the DOCX deliverable directly.
`convert_docx_to_pdf` hands those same bytes to a self-hosted Gotenberg
instance (see docker/gotenberg — GOTENBERG_URL env var) for a faithful
PDF rendering of the identical document, rather than maintaining a
second, separately-designed PDF template that could drift out of visual
sync with the DOCX one.
"""

import os
from io import BytesIO

import requests
from docxtpl import DocxTemplate, RichText

TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "templates", "research_report.docx")

GOTENBERG_URL = os.environ.get("GOTENBERG_URL")  # e.g. "http://localhost:3000" — unset means the PDF path is simply unavailable, not a crash
GOTENBERG_TIMEOUT_S = 60


class ReportRenderError(Exception):
    pass


_CONFIDENCE_COLORS = {
    "high": "2FA377",
    "medium": "B8861F",
    "low": "C74B3D",
}


def _confidence_richtext(confidence: str) -> RichText:
    """Colors each finding's confidence label the same way NAVI's own UI
    already color-codes status (navi-pwa/src/index.css's --status-success/
    -warning/-danger, darkened for print — see build_report_templates.py's
    own comment on why) rather than leaving it as plain, unscannable text."""
    label = (confidence or "").strip().lower()
    color = _CONFIDENCE_COLORS.get(label, "555555")
    rt = RichText()
    rt.add(f"Confidence: {label.upper() or 'UNKNOWN'}", color=color, bold=True)
    return rt


def render_research_report_docx(report: dict) -> bytes:
    """`report` is exactly RESEARCH_EXECUTE_PLAN.md's JSON schema (title,
    date, objective, executive_summary, methodology, key_findings,
    analysis, recommendations, gaps, sources). Raises ReportRenderError
    on a malformed report rather than silently producing a half-filled
    document."""
    try:
        doc = DocxTemplate(TEMPLATE_PATH)
        context = dict(report)
        context["key_findings"] = [
            {**f, "confidence_rt": _confidence_richtext(f.get("confidence", ""))}
            for f in report.get("key_findings", [])
        ]
        doc.render(context)
        buf = BytesIO()
        doc.save(buf)
        return buf.getvalue()
    except Exception as e:
        raise ReportRenderError(f"Failed to render research report: {e}") from e


def convert_docx_to_pdf(docx_bytes: bytes) -> bytes:
    """Converts via a self-hosted Gotenberg instance's LibreOffice route
    — same rendering engine real Word/LibreOffice would use, so the PDF
    is a faithful copy of the DOCX, not a second hand-built approximation.
    Raises ReportRenderError if GOTENBERG_URL isn't configured or the
    service call fails — callers decide whether that's fatal or whether
    to fall back to offering the DOCX alone."""
    if not GOTENBERG_URL:
        raise ReportRenderError("GOTENBERG_URL is not configured — PDF conversion is unavailable")
    try:
        response = requests.post(
            f"{GOTENBERG_URL.rstrip('/')}/forms/libreoffice/convert",
            files={"files": ("report.docx", docx_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
            timeout=GOTENBERG_TIMEOUT_S,
        )
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        raise ReportRenderError(f"Gotenberg conversion failed: {e}") from e
