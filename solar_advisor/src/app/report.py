"""Monthly Energy Report PDF generator using ReportLab."""

from __future__ import annotations

from io import BytesIO

import pandas as pd
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.legends import Legend
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _monthly_bar_drawing(monthly_df: pd.DataFrame) -> Drawing:
    """Build a crisp vector bar chart of monthly generation vs consumption."""
    drawing = Drawing(460, 190)
    if monthly_df is None or monthly_df.empty:
        return drawing

    subset = monthly_df.tail(12)
    months = [str(m) for m in subset["month"].tolist()]
    gen_vals = [float(v) for v in subset["gen_kwh"].tolist()]
    cons_vals = [float(v) for v in subset["cons_kwh"].tolist()]

    chart = VerticalBarChart()
    chart.x = 45
    chart.y = 35
    chart.height = 125
    chart.width = 395
    chart.data = [gen_vals, cons_vals]
    chart.categoryAxis.categoryNames = months
    chart.categoryAxis.labels.boxAnchor = "ne"
    chart.categoryAxis.labels.angle = 30
    chart.categoryAxis.labels.fontSize = 7
    chart.valueAxis.valueMin = 0
    max_val = max(gen_vals + cons_vals + [10.0])
    chart.valueAxis.valueMax = round(max_val * 1.15, 0)
    chart.valueAxis.labels.fontSize = 7
    chart.bars[0].fillColor = colors.HexColor("#F59E0B")
    chart.bars[1].fillColor = colors.HexColor("#2563EB")

    legend = Legend()
    legend.x = 260
    legend.y = 175
    legend.fontSize = 8
    legend.colorNamePairs = [
        (colors.HexColor("#F59E0B"), "Solar Generation (kWh)"),
        (colors.HexColor("#2563EB"), "Consumption (kWh)"),
    ]

    drawing.add(chart)
    drawing.add(legend)
    return drawing


def build_monthly_pdf_report(
    home_label: str,
    month: str,
    kpis: dict,
    monthly_df: pd.DataFrame,
    anomalies: list[dict],
    recommendations: list[dict],
    ai_summary: str,
    import_tariff: float,
    export_tariff: float,
) -> bytes:
    """Build and return a PDF report as bytes using ReportLab."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title=f"SolarSense Monthly Energy Report - {home_label}",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "SolarTitle",
        parent=styles["Title"],
        fontSize=18,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=4,
    )
    h2_style = ParagraphStyle(
        "SolarH2",
        parent=styles["Heading2"],
        fontSize=12,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=10,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "SolarBody",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=13.5,
    )

    # Replace Unicode rupee symbol with ASCII-safe "INR " for built-in PDF Helvetica font
    clean_summary = (ai_summary or "No summary generated.").replace("₹", "INR ")

    elements = [
        Paragraph("SolarSense - Monthly Energy Advisory Report", title_style),
        Paragraph(
            f"<b>Household:</b> {home_label} &nbsp;|&nbsp; <b>Report Month:</b> {month} "
            f"&nbsp;|&nbsp; <b>Period:</b> {kpis.get('period', 'N/A')}<br/>"
            f"<b>Tariffs:</b> Import INR {import_tariff:.2f}/kWh, Export INR {export_tariff:.2f}/kWh",
            body_style,
        ),
        Spacer(1, 8),
        Paragraph("1. Key Performance Indicators", h2_style),
    ]

    kpi_rows = [
        ["Metric", "Value", "Metric", "Value"],
        [
            "Solar Generation",
            f"{kpis.get('gen_kwh', 0.0):,.2f} kWh",
            "Total Consumption",
            f"{kpis.get('cons_kwh', 0.0):,.2f} kWh",
        ],
        [
            "Self-Consumption",
            f"{kpis.get('self_consumption_pct', 0.0):.2f} %",
            "Self-Sufficiency",
            f"{kpis.get('self_sufficiency_pct', 0.0):.2f} %",
        ],
        [
            "Grid Import",
            f"{kpis.get('import_kwh', 0.0):,.2f} kWh",
            "Solar Export",
            f"{kpis.get('export_kwh', 0.0):,.2f} kWh",
        ],
        [
            "Baseload / Peak",
            f"{kpis.get('baseload_kw', 0.0):.2f} / {kpis.get('peak_kw', 0.0):.2f} kW",
            "Estimated Net Bill",
            f"INR {kpis.get('net_bill_inr', 0.0):,.2f}",
        ],
    ]
    kpi_table = Table(kpi_rows, colWidths=[4.2 * cm, 4.3 * cm, 4.2 * cm, 4.3 * cm])
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F8FAFC")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.extend([kpi_table, Spacer(1, 8)])

    elements.append(Paragraph("2. Generation vs Consumption Trend", h2_style))
    elements.append(_monthly_bar_drawing(monthly_df))
    elements.append(Spacer(1, 6))

    elements.append(Paragraph("3. AI-Generated Monthly Explanation", h2_style))
    elements.append(Paragraph(clean_summary, body_style))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("4. Top Recommendations & Savings Estimates", h2_style))
    if recommendations:
        rec_rows = [["ID", "Recommendation", "Effort", "Est. kWh/mo", "Est. INR/mo"]]
        for r in recommendations[:6]:
            rec_rows.append([
                str(r.get("id", "")),
                Paragraph(f"<b>{r.get('title', '')}</b><br/>{r.get('action', '')}", body_style),
                str(r.get("effort", "")).title(),
                f"{float(r.get('kwh_saved_month', 0.0)):,.2f}",
                f"INR {float(r.get('inr_saved_month', 0.0)):,.2f}",
            ])
        rec_table = Table(rec_rows, colWidths=[1.2 * cm, 9.4 * cm, 1.8 * cm, 2.3 * cm, 2.3 * cm])
        rec_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F766E")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(rec_table)
    else:
        elements.append(Paragraph("No optimization recommendations were triggered for this period.", body_style))

    elements.append(Spacer(1, 8))
    elements.append(Paragraph("5. Detected Energy Anomalies", h2_style))
    if anomalies:
        anom_rows = [["Date", "Type", "Observed", "Expected", "Reason"]]
        for a in anomalies[:10]:
            unit = str(a.get("unit", ""))
            anom_rows.append([
                str(a.get("date", "")),
                str(a.get("type", "")),
                f"{a.get('value', '')} {unit}".strip(),
                f"{a.get('expected', '')} {unit}".strip(),
                Paragraph(str(a.get("reason", "")), body_style),
            ])
        anom_table = Table(anom_rows, colWidths=[2.3 * cm, 3.0 * cm, 2.2 * cm, 2.2 * cm, 7.3 * cm])
        anom_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#B45309")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(anom_table)
    else:
        elements.append(Paragraph("No significant energy anomalies were detected during this period.", body_style))

    elements.append(Spacer(1, 10))
    elements.append(Paragraph(
        "<i>Note: Savings and payback projections are estimates based on historical meter readings and user-selected tariffs, not financial guarantees.</i>",
        body_style,
    ))

    doc.build(elements)
    return buffer.getvalue()
