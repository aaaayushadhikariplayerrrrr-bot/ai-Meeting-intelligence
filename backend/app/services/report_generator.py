"""PDF Meeting Report Generator for MEMORA.

Generates comprehensive, publication-quality executive reports in PDF format:
1. MEMORA branding, report title, metadata, and confidentiality notes.
2. Meeting title, actual/scheduled date, project, category, and source.
3. Registered participants and speaker attributions.
4. Executive summary and key takeaways (AI generated).
5. Important discussion points and topic timeline.
6. Explicit decisions, status, rationale, and evidence grounding.
7. Action items, assigned owners, deadlines, and lifecycle status.
8. Unresolved questions and identified risks.
9. Transcript highlights with exact timestamps and privacy notice.
"""

import io
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from xml.sax.saxutils import escape as xml_escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger(__name__)

# Primary Brand Colors (matching MEMORA dark/violet theme adapted for white paper)
PRIMARY_COLOR = colors.HexColor("#4338ca")      # Deep Indigo
SECONDARY_COLOR = colors.HexColor("#6366f1")    # Iris / Violet
TEXT_DARK = colors.HexColor("#0f172a")          # Slate 900
TEXT_MUTED = colors.HexColor("#475569")         # Slate 600
TEXT_LIGHT = colors.HexColor("#94a3b8")         # Slate 400
BG_LIGHT = colors.HexColor("#f8fafc")           # Slate 50
BG_HEADER = colors.HexColor("#f1f5f9")          # Slate 100
BORDER_COLOR = colors.HexColor("#e2e8f0")       # Slate 200
SUCCESS_COLOR = colors.HexColor("#059669")      # Emerald
WARNING_COLOR = colors.HexColor("#d97706")      # Amber
DANGER_COLOR = colors.HexColor("#dc2626")       # Red


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas for total page count and professional header/footer."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._saved_page_states: List[Dict[str, Any]] = []

    def showPage(self) -> None:
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages: int) -> None:
        self.saveState()
        # Top Running Header (Pages 2+)
        if self._pageNumber > 1:
            self.setFont("Helvetica", 8)
            self.setFillColor(TEXT_MUTED)
            self.drawString(36, 756, "MEMORA · Meeting Intelligence & Organizational Memory Report")
            self.drawRightString(612 - 36, 756, f"Meeting Report")
            self.setStrokeColor(BORDER_COLOR)
            self.setLineWidth(0.5)
            self.line(36, 750, 612 - 36, 750)

        # Bottom Running Footer (All Pages)
        self.setFont("Helvetica", 8)
        self.setFillColor(TEXT_MUTED)
        self.drawString(36, 28, "MEMORA Organizational Memory Platform · Confidential & Proprietary")
        self.drawRightString(612 - 36, 28, f"Page {self._pageNumber} of {total_pages}")
        self.setStrokeColor(BORDER_COLOR)
        self.setLineWidth(0.5)
        self.line(36, 38, 612 - 36, 38)
        self.restoreState()


def safe_text(value: Any, default: str = "—") -> str:
    """Escapes XML entities and provides default fallback for empty values."""
    if value is None or str(value).strip() == "":
        return default
    return xml_escape(str(value).strip())


def format_seconds(seconds: Optional[float]) -> str:
    """Converts seconds into MM:SS or HH:MM:SS format."""
    if seconds is None:
        return "—"
    try:
        total_sec = int(round(seconds))
        hrs = total_sec // 3600
        mins = (total_sec % 3600) // 60
        secs = total_sec % 60
        if hrs > 0:
            return f"{hrs:02d}:{mins:02d}:{secs:02d}"
        return f"{mins:02d}:{secs:02d}"
    except Exception:
        return "—"


def format_date_str(dt_val: Any) -> str:
    """Formats datetime or ISO string to readable date/time."""
    if not dt_val:
        return "—"
    if isinstance(dt_val, datetime):
        return dt_val.strftime("%b %d, %Y %I:%M %p UTC")
    try:
        parsed = datetime.fromisoformat(str(dt_val).replace("Z", "+00:00"))
        return parsed.strftime("%b %d, %Y %I:%M %p UTC")
    except Exception:
        return str(dt_val)[:19]


def generate_meeting_pdf_report(
    meeting_data: Dict[str, Any],
    intelligence_data: Optional[Dict[str, Any]] = None,
    participants_data: Optional[List[Dict[str, Any]]] = None,
    evidence_snippets: Optional[List[Dict[str, Any]]] = None,
) -> bytes:
    """Generates a professional meeting report PDF and returns its bytes.

    Args:
        meeting_data: Dict with meeting title, date, category, project, status, etc.
        intelligence_data: Dict with summary, topics, decisions, action_items, risks, questions.
        participants_data: List of participant dicts (display_name, email, role, is_speaker).
        evidence_snippets: List of key transcript evidence snippets with timestamps.

    Returns:
        bytes: Generated PDF content.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=44,
        bottomMargin=44,
    )

    # Styles
    base_styles = getSampleStyleSheet()

    title_brand_style = ParagraphStyle(
        "TitleBrand",
        parent=base_styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=PRIMARY_COLOR,
    )

    doc_subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=TEXT_MUTED,
    )

    section_heading_style = ParagraphStyle(
        "SectionHeading",
        parent=base_styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=PRIMARY_COLOR,
        spaceBefore=10,
        spaceAfter=4,
    )

    body_style = ParagraphStyle(
        "DocBody",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=TEXT_DARK,
    )

    body_bold_style = ParagraphStyle(
        "DocBodyBold",
        parent=body_style,
        fontName="Helvetica-Bold",
    )

    meta_label_style = ParagraphStyle(
        "MetaLabel",
        parent=base_styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=TEXT_MUTED,
    )

    meta_val_style = ParagraphStyle(
        "MetaVal",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=TEXT_DARK,
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=base_styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=TEXT_DARK,
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=TEXT_DARK,
    )

    badge_active_style = ParagraphStyle(
        "BadgeActive",
        parent=table_cell_style,
        fontName="Helvetica-Bold",
        textColor=SUCCESS_COLOR,
    )

    badge_warn_style = ParagraphStyle(
        "BadgeWarn",
        parent=table_cell_style,
        fontName="Helvetica-Bold",
        textColor=WARNING_COLOR,
    )

    notice_style = ParagraphStyle(
        "NoticeStyle",
        parent=base_styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=10,
        textColor=TEXT_MUTED,
    )

    story: List[Any] = []
    content_width = 540  # 612 - 36 - 36

    # -------------------------------------------------------------
    # 1. Header Banner
    # -------------------------------------------------------------
    story.append(Paragraph("MEMORA", title_brand_style))
    story.append(Paragraph("Executive Meeting Intelligence & Organizational Memory Report", doc_subtitle_style))
    story.append(Spacer(1, 6))
    story.append(
        HRFlowable(
            width="100%",
            thickness=1.5,
            color=PRIMARY_COLOR,
            spaceBefore=2,
            spaceAfter=10,
        )
    )

    # -------------------------------------------------------------
    # 2. Meeting Metadata Table (Overview Card)
    # -------------------------------------------------------------
    actual_date = (
        meeting_data.get("actual_start_time")
        or meeting_data.get("scheduled_start_time")
        or meeting_data.get("created_at")
    )
    meeting_date_str = format_date_str(actual_date)
    project_name = meeting_data.get("project_name") or "Unassigned"
    category_name = meeting_data.get("category") or "General"
    status_str = meeting_data.get("status") or "PROCESSED"
    source_type = meeting_data.get("input_source_type") or "UPLOAD"

    meta_table_data = [
        [
            Paragraph("Meeting Title:", meta_label_style),
            Paragraph(safe_text(meeting_data.get("title")), meta_val_style),
            Paragraph("Date / Time:", meta_label_style),
            Paragraph(meeting_date_str, meta_val_style),
        ],
        [
            Paragraph("Project:", meta_label_style),
            Paragraph(safe_text(project_name), meta_val_style),
            Paragraph("Category:", meta_label_style),
            Paragraph(safe_text(category_name), meta_val_style),
        ],
        [
            Paragraph("Source:", meta_label_style),
            Paragraph(safe_text(source_type), meta_val_style),
            Paragraph("Status:", meta_label_style),
            Paragraph(safe_text(status_str), meta_val_style),
        ],
    ]

    meta_table = Table(
        meta_table_data,
        colWidths=[80, 190, 80, 190],
    )
    meta_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), BG_LIGHT),
            ("BOX", (0, 0), (-1, -1), 1, BORDER_COLOR),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ])
    )
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # Description if present
    if meeting_data.get("description"):
        story.append(Paragraph(f"<b>Overview:</b> {safe_text(meeting_data['description'])}", body_style))
        story.append(Spacer(1, 8))

    # -------------------------------------------------------------
    # 3. Participants Section
    # -------------------------------------------------------------
    story.append(Paragraph("Participants", section_heading_style))
    participants = participants_data or []
    if participants:
        part_rows = [
            [
                Paragraph("Participant Name", table_header_style),
                Paragraph("Email / Identifier", table_header_style),
                Paragraph("Role", table_header_style),
                Paragraph("Speaker", table_header_style),
            ]
        ]
        for p in participants:
            speaker_badge = "Yes" if p.get("is_speaker") else "Attendee"
            part_rows.append([
                Paragraph(safe_text(p.get("display_name")), table_cell_style),
                Paragraph(safe_text(p.get("email")), table_cell_style),
                Paragraph(safe_text(p.get("role")), table_cell_style),
                Paragraph(speaker_badge, table_cell_style),
            ])
        part_table = Table(part_rows, colWidths=[160, 200, 100, 80])
        part_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), BG_HEADER),
                ("BOX", (0, 0), (-1, -1), 0.8, BORDER_COLOR),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ])
        )
        story.append(part_table)
    else:
        story.append(Paragraph("<i>No registered participants recorded for this meeting.</i>", body_style))
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------
    # 4. Executive Summary Section
    # -------------------------------------------------------------
    story.append(Paragraph("Executive Summary", section_heading_style))
    story.append(Paragraph("Synthesized with privacy-safe redactions applied.", notice_style))
    story.append(Spacer(1, 4))

    intel = intelligence_data or {}
    summary_obj = intel.get("summary") or {}
    summary_text = (
        summary_obj.get("summary_text")
        or summary_obj.get("executive_summary")
        or summary_obj.get("detailed_summary")
    ) if isinstance(summary_obj, dict) else None

    if summary_text:
        # Wrap paragraphs
        for para in summary_text.split("\n\n"):
            if para.strip():
                story.append(Paragraph(safe_text(para), body_style))
                story.append(Spacer(1, 4))

        # Key Takeaways
        key_takeaways = summary_obj.get("key_takeaways")
        if key_takeaways:
            if isinstance(key_takeaways, str):
                try:
                    key_takeaways = json.loads(key_takeaways)
                except Exception:
                    key_takeaways = [key_takeaways]
            if isinstance(key_takeaways, list) and key_takeaways:
                story.append(Spacer(1, 4))
                story.append(Paragraph("<b>Key Takeaways:</b>", body_bold_style))
                for item in key_takeaways:
                    story.append(Paragraph(f"• {safe_text(item)}", body_style))
                    story.append(Spacer(1, 2))

        # Action Items Overview
        action_overview = summary_obj.get("action_items_overview")
        if action_overview:
            story.append(Spacer(1, 4))
            story.append(Paragraph(f"<b>Actions Overview:</b> {safe_text(action_overview)}", body_style))
    else:
        story.append(Paragraph("<i>Executive summary has not yet been extracted for this meeting.</i>", body_style))
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------
    # 5. Key Discussion Topics & Agenda
    # -------------------------------------------------------------
    topics = intel.get("topics") or []
    if topics:
        story.append(KeepTogether([
            Paragraph("Important Discussion Points & Topics", section_heading_style),
            Spacer(1, 4),
        ]))
        topic_rows = [
            [
                Paragraph("Time", table_header_style),
                Paragraph("Topic & Summary", table_header_style),
            ]
        ]
        for t in topics:
            time_range = f"{format_seconds(t.get('start_time'))} – {format_seconds(t.get('end_time'))}"
            topic_desc = f"<b>{safe_text(t.get('title'))}</b>"
            if t.get("description"):
                topic_desc += f"<br/>{safe_text(t.get('description'))}"
            topic_rows.append([
                Paragraph(time_range, table_cell_style),
                Paragraph(topic_desc, table_cell_style),
            ])
        top_table = Table(topic_rows, colWidths=[90, 450])
        top_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), BG_HEADER),
                ("BOX", (0, 0), (-1, -1), 0.8, BORDER_COLOR),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ])
        )
        story.append(top_table)
        story.append(Spacer(1, 10))

    # -------------------------------------------------------------
    # 6. Decisions Made (Evidence Grounded)
    # -------------------------------------------------------------
    decisions = intel.get("decisions") or []
    if decisions:
        story.append(KeepTogether([
            Paragraph("Decisions Made", section_heading_style),
            Spacer(1, 4),
        ]))
        dec_rows = [
            [
                Paragraph("Decision", table_header_style),
                Paragraph("Status", table_header_style),
                Paragraph("Rationale & Evidence", table_header_style),
                Paragraph("Timestamp", table_header_style),
            ]
        ]
        for d in decisions:
            status = d.get("status") or "ACTIVE"
            status_style = badge_active_style if status == "ACTIVE" else badge_warn_style
            ev_ref = f"Segment ID: {safe_text(d.get('evidence_segment_id'))}" if d.get("evidence_segment_id") else "Direct discussion"
            rationale_full = safe_text(d.get("rationale") or "—")
            dec_rows.append([
                Paragraph(f"<b>{safe_text(d.get('decision_text'))}</b>", table_cell_style),
                Paragraph(status, status_style),
                Paragraph(f"{rationale_full}<br/><font color='#64748b' size='7'>{ev_ref}</font>", table_cell_style),
                Paragraph(format_seconds(d.get("timestamp_seconds")), table_cell_style),
            ])
        dec_table = Table(dec_rows, colWidths=[180, 75, 215, 70])
        dec_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), BG_HEADER),
                ("BOX", (0, 0), (-1, -1), 0.8, BORDER_COLOR),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ])
        )
        story.append(dec_table)
        story.append(Spacer(1, 10))

    # -------------------------------------------------------------
    # 7. Action Items & Commitments
    # -------------------------------------------------------------
    action_items = intel.get("action_items") or []
    if action_items:
        story.append(KeepTogether([
            Paragraph("Action Items & Responsibilities", section_heading_style),
            Spacer(1, 4),
        ]))
        act_rows = [
            [
                Paragraph("Action Item", table_header_style),
                Paragraph("Assignee", table_header_style),
                Paragraph("Status", table_header_style),
                Paragraph("Deadline", table_header_style),
                Paragraph("Ref", table_header_style),
            ]
        ]
        for a in action_items:
            deadline_str = format_date_str(a.get("deadline"))
            status = a.get("status") or "PENDING"
            act_rows.append([
                Paragraph(safe_text(a.get("description")), table_cell_style),
                Paragraph(f"<b>{safe_text(a.get('owner_name') or 'Unassigned')}</b>", table_cell_style),
                Paragraph(status, badge_active_style if status == "COMPLETED" else badge_warn_style),
                Paragraph(deadline_str, table_cell_style),
                Paragraph(format_seconds(a.get("timestamp_seconds")), table_cell_style),
            ])
        act_table = Table(act_rows, colWidths=[200, 100, 80, 100, 60])
        act_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), BG_HEADER),
                ("BOX", (0, 0), (-1, -1), 0.8, BORDER_COLOR),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ])
        )
        story.append(act_table)
        story.append(Spacer(1, 10))

    # -------------------------------------------------------------
    # 8. Unresolved Questions & Risks
    # -------------------------------------------------------------
    questions = intel.get("questions") or []
    risks = intel.get("risks") or []
    if questions or risks:
        story.append(KeepTogether([
            Paragraph("Unresolved Questions & Identified Risks", section_heading_style),
            Spacer(1, 4),
        ]))
        if questions:
            q_rows = [
                [
                    Paragraph("Unresolved Question", table_header_style),
                    Paragraph("Raised By", table_header_style),
                    Paragraph("Status", table_header_style),
                ]
            ]
            for q in questions:
                q_rows.append([
                    Paragraph(safe_text(q.get("question_text")), table_cell_style),
                    Paragraph(safe_text(q.get("raised_by") or "—"), table_cell_style),
                    Paragraph(safe_text(q.get("status") or "OPEN"), badge_warn_style),
                ])
            q_table = Table(q_rows, colWidths=[330, 130, 80])
            q_table.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), BG_HEADER),
                    ("BOX", (0, 0), (-1, -1), 0.8, BORDER_COLOR),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ])
            )
            story.append(q_table)
            story.append(Spacer(1, 6))

        if risks:
            r_rows = [
                [
                    Paragraph("Risk / Blocker", table_header_style),
                    Paragraph("Severity", table_header_style),
                ]
            ]
            for r in risks:
                sev = r.get("severity") or "MEDIUM"
                r_rows.append([
                    Paragraph(safe_text(r.get("description")), table_cell_style),
                    Paragraph(sev, badge_warn_style if sev in ("HIGH", "CRITICAL") else table_cell_style),
                ])
            r_table = Table(r_rows, colWidths=[420, 120])
            r_table.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), BG_HEADER),
                    ("BOX", (0, 0), (-1, -1), 0.8, BORDER_COLOR),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ])
            )
            story.append(r_table)
            story.append(Spacer(1, 10))

    # -------------------------------------------------------------
    # 9. Important Evidence & Transcript Highlights
    # -------------------------------------------------------------
    evidence = evidence_snippets or []
    if evidence:
        story.append(KeepTogether([
            Paragraph("Verified Evidence Highlights", section_heading_style),
            Paragraph("Original transcript excerpts grounded with exact audio timestamps.", notice_style),
            Spacer(1, 4),
        ]))
        ev_rows = [
            [
                Paragraph("Timestamp", table_header_style),
                Paragraph("Speaker", table_header_style),
                Paragraph("Verified Passage Excerpt", table_header_style),
            ]
        ]
        for snip in evidence[:8]:  # Top 8 highlights
            ts_str = f"{format_seconds(snip.get('start_time'))} – {format_seconds(snip.get('end_time'))}"
            ev_rows.append([
                Paragraph(ts_str, table_cell_style),
                Paragraph(safe_text(snip.get("speaker") or "Speaker"), table_cell_style),
                Paragraph(f"<i>\"{safe_text(snip.get('text'))}\"</i>", table_cell_style),
            ])
        ev_table = Table(ev_rows, colWidths=[90, 80, 370])
        ev_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), BG_HEADER),
                ("BOX", (0, 0), (-1, -1), 0.8, BORDER_COLOR),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ])
        )
        story.append(ev_table)
        story.append(Spacer(1, 10))

    # Build PDF with NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    return buffer.getvalue()
