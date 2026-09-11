"""
tools/build_report_templates.py

Generates the .docx TEMPLATE files docxtpl renders research deliverables
against (tools/templates/research_report.docx). This is the "design
system" — run ONCE (or whenever the design changes), not at request
time. The design lives entirely in this generated Word file (fonts,
colors, margins, table style, page numbers via python-docx's real
Heading/Title styles) so rendering a report is just filling in data —
no layout decision is ever made by an LLM at generation time, which is
the whole point (see IDEAS.md's "Document generation & viewing stack"
section for why).

Typography/color choices, not arbitrary:
- Body font Calibri — Word's own modern default, and LibreOffice ships
  a metric-compatible substitute (Carlito) so a Gotenberg/LibreOffice
  conversion of this template doesn't reflow. Matches the real
  consulting-report guidance checked before building this: one or two
  fonts max, neutral sans-serif, no more.
- Heading color reuses NAVI's own product accent (`--accent: #5d8bff`,
  navi-pwa/src/index.css) rather than inventing a new one — the one
  place this deliverately ties back to NAVI's identity. Body stays
  plain black; restraint over decoration, per the same research.
- Confidence color-coding (high/medium/low on each finding) reuses
  NAVI's own existing status colors (--status-success/-warning/-danger
  in the same file) rather than a fresh palette a template like this
  would otherwise invent from nothing.

Run: `python -m tools.build_report_templates`
"""

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

TEMPLATES_DIR = Path(__file__).parent / "templates"

# Reused verbatim from navi-pwa/src/index.css's :root token block —
# keep these two files in sync by hand if NAVI's palette ever changes.
ACCENT = RGBColor(0x5D, 0x8B, 0xFF)
TEXT_PRIMARY = RGBColor(0x1A, 0x1A, 0x1A)
STATUS_SUCCESS = RGBColor(0x2F, 0xA3, 0x77)  # darker than the UI's #4ad9a0 — needs to read on white paper, not a dark UI surface
STATUS_WARNING = RGBColor(0xB8, 0x86, 0x1F)  # darker than the UI's #f2bd57, same reason
STATUS_DANGER = RGBColor(0xC7, 0x4B, 0x3D)   # darker than the UI's #ef7a6b, same reason


def _add_page_number_field(paragraph) -> None:
    """python-docx has no high-level API for Word field codes — this is
    the standard low-level recipe (fldChar begin/instrText "PAGE"/fldChar
    end) for inserting a live page-number field, verified by re-opening
    the generated file afterward rather than assumed correct."""
    run = paragraph.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = "PAGE"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr_text)
    run._r.append(fld_end)


def _set_default_styles(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal.font.color.rgb = TEXT_PRIMARY
    normal.paragraph_format.space_after = Pt(8)
    normal.paragraph_format.line_spacing = 1.15

    title = doc.styles["Title"]
    title.font.name = "Calibri"
    title.font.size = Pt(26)
    title.font.bold = True
    title.font.color.rgb = TEXT_PRIMARY

    heading_sizes = {"Heading 1": 16, "Heading 2": 13, "Heading 3": 11.5}
    for name, size in heading_sizes.items():
        style = doc.styles[name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = ACCENT
        style.paragraph_format.space_before = Pt(18)
        style.paragraph_format.space_after = Pt(6)


def _set_margins(doc: Document) -> None:
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)


def _add_footer_page_number(doc: Document) -> None:
    footer = doc.sections[0].footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.style = doc.styles["Normal"]
    for run in p.runs:
        run.text = ""
    prefix = p.add_run("Page ")
    prefix.font.size = Pt(9)
    prefix.font.color.rgb = RGBColor(0x77, 0x77, 0x77)
    _add_page_number_field(p)


def build_research_report_template() -> Path:
    doc = Document()
    _set_margins(doc)
    _set_default_styles(doc)
    _add_footer_page_number(doc)

    doc.add_paragraph("{{ title }}", style="Title")
    meta = doc.add_paragraph()
    meta.paragraph_format.space_after = Pt(24)
    meta_run = meta.add_run("Prepared {{ date }}")
    meta_run.font.size = Pt(10)
    meta_run.font.color.rgb = RGBColor(0x77, 0x77, 0x77)

    doc.add_heading("Executive Summary", level=1)
    doc.add_paragraph("{{ executive_summary }}")

    doc.add_heading("Objective", level=1)
    doc.add_paragraph("{{ objective }}")

    doc.add_heading("Methodology", level=1)
    doc.add_paragraph("{{ methodology.approach }}")
    doc.add_heading("Sub-questions pursued", level=2)
    doc.add_paragraph("{% for q in methodology.sub_questions %}")
    doc.add_paragraph("{{ q }}", style="List Bullet")
    doc.add_paragraph("{% endfor %}")
    doc.add_paragraph("Sources consulted: {{ methodology.sources_consulted }}")
    doc.add_heading("Limitations", level=2)
    doc.add_paragraph("{% for l in methodology.limitations %}")
    doc.add_paragraph("{{ l }}", style="List Bullet")
    doc.add_paragraph("{% endfor %}")

    doc.add_heading("Key Findings", level=1)
    doc.add_paragraph("{% for f in key_findings %}")
    doc.add_heading("{{ f.sub_question }}", level=3)
    finding_p = doc.add_paragraph()
    finding_p.add_run("{{ f.finding }}").bold = True
    doc.add_paragraph("{{ f.supporting_evidence }}")
    conf_p = doc.add_paragraph()
    conf_p.paragraph_format.space_after = Pt(14)
    conf_run = conf_p.add_run("{{r f.confidence_rt }}")
    conf_run.italic = True
    evidence_run = conf_p.add_run("  —  {{ f.confidence_rationale }}  (Sources: {{ f.source_ids|join(', ') }})")
    evidence_run.italic = True
    evidence_run.font.size = Pt(9.5)
    evidence_run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    doc.add_paragraph("{% endfor %}")

    doc.add_heading("Analysis", level=1)
    doc.add_paragraph("{{ analysis }}")

    doc.add_paragraph("{% if recommendations %}")
    doc.add_heading("Recommendations", level=1)
    doc.add_paragraph("{% for r in recommendations %}")
    rec_p = doc.add_paragraph(style="List Bullet")
    rec_p.add_run("{{ r.recommendation }}").bold = True
    rec_p.add_run(" (priority: {{ r.priority }}) — {{ r.rationale }}")
    doc.add_paragraph("{% endfor %}")
    doc.add_paragraph("{% endif %}")

    doc.add_paragraph("{% if gaps %}")
    doc.add_heading("Gaps", level=1)
    doc.add_paragraph("{% for g in gaps %}")
    doc.add_paragraph("{{ g }}", style="List Bullet")
    doc.add_paragraph("{% endfor %}")
    doc.add_paragraph("{% endif %}")

    doc.add_heading("Sources", level=1)
    # 4 real rows, not 2 — docxtpl's {%tr %} tag deletes the ENTIRE
    # <w:tr> row it's found in (verified against the installed docxtpl
    # 0.20.2's actual regex in template.py's patch_xml, not assumed from
    # memory — a first attempt putting both {%tr for %} and
    # {%tr endfor %} in the same row as the real data cells silently
    # dropped the whole row, data cells included). The for/endfor tags
    # need their OWN dedicated rows so only THEIR <w:tr> wrapper gets
    # stripped, leaving the real templated row in between as the thing
    # Jinja's `for` loop actually repeats.
    table = doc.add_table(rows=4, cols=4)
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    header_cells = table.rows[0].cells
    for cell, text in zip(header_cells, ["ID", "Title", "Publisher", "Accessed"]):
        cell.paragraphs[0].add_run(text).bold = True

    table.rows[1].cells[0].paragraphs[0].add_run("{%tr for s in sources %}")

    data_cells = table.rows[2].cells
    data_cells[0].paragraphs[0].add_run("{{ s.id }}")
    # The URL is a real hyperlink target in real usage but python-docx
    # can't easily template a dynamic hyperlink's address — plain text
    # under the title is the honest compromise rather than a fake-looking
    # non-clickable "link" style run.
    data_cells[1].paragraphs[0].add_run("{{ s.title }}")
    url_p = data_cells[1].add_paragraph()
    url_run = url_p.add_run("{{ s.url }}")
    url_run.font.size = Pt(8.5)
    url_run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    data_cells[2].paragraphs[0].add_run("{{ s.publisher }}")
    data_cells[3].paragraphs[0].add_run("{{ s.accessed_date }}")

    table.rows[3].cells[0].paragraphs[0].add_run("{%tr endfor %}")

    for row in (table.rows[0], table.rows[2]):
        for cell in row.cells:
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(9.5)

    TEMPLATES_DIR.mkdir(exist_ok=True)
    out_path = TEMPLATES_DIR / "research_report.docx"
    doc.save(out_path)
    return out_path


if __name__ == "__main__":
    path = build_research_report_template()
    print(f"Wrote {path}")
