"""
Module: Automated PDF Report Generator
Purpose: Ingests local flat files, performs statistical aggregation,
         and outputs formatted PDF documentation.
Author: Himanshu
Dependencies: pandas, reportlab
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, HRFlowable, PageBreak
)


# ── Constants ────────────────────────────────────────────────────────────────

INPUT_CSV = "sales_data.csv"
OUTPUT_PDF = "generated_report.pdf"
REPORT_TITLE = "Sales Performance Report"
COMPANY_NAME = "RetailEdge Analytics"

# Brand palette
CLR_HEADER = colors.HexColor("#1B2A4A")   # Deep navy
CLR_ACCENT = colors.HexColor("#2E86AB")   # Steel blue
CLR_ROW_A  = colors.HexColor("#EDF4FB")   # Alternate row tint
CLR_WHITE  = colors.white
CLR_TEXT   = colors.HexColor("#2D2D2D")


# ── Data Ingestion ────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load tabular data from a CSV file into a Pandas DataFrame.

    Args:
        filepath: Relative or absolute path to the source CSV.

    Returns:
        Raw DataFrame with validated column presence.

    Raises:
        FileNotFoundError: If the CSV does not exist at the given path.
        ValueError: If required columns are absent in the loaded file.
    """
    # Initializing data ingestion pipeline; implementing standard file I/O exception handling.
    required_cols = {"region", "category", "product",
                     "units_sold", "unit_price", "discount_pct", "month"}

    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"[CRITICAL] Source file not found: '{filepath}'. "
            "Ensure the CSV exists in the working directory."
        )
    except pd.errors.EmptyDataError:
        raise ValueError(f"[ERROR] The file '{filepath}' is empty or malformed.")

    # Validate that all expected columns are present before downstream processing.
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"[ERROR] Missing required columns: {missing}")

    print(f"[INFO] Data loaded successfully — {len(df)} records, {len(df.columns)} fields.")
    return df


# ── Data Analysis ─────────────────────────────────────────────────────────────

def compute_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive net revenue per transaction after applying discount.

    Revenue = units_sold × unit_price × (1 − discount_pct / 100)
    """
    df = df.copy()
    df["revenue"] = (
        df["units_sold"] * df["unit_price"] * (1 - df["discount_pct"] / 100)
    )
    return df


def analyze_data(df: pd.DataFrame) -> dict:
    """
    Perform multi-dimensional aggregation on the sales DataFrame.

    Computes:
        - Executive KPIs  : total revenue, units sold, avg. discount
        - Category summary: revenue and units grouped by product category
        - Regional summary: revenue and units grouped by region
        - Monthly trend   : total revenue per month (preserving calendar order)
        - Top 5 products  : ranked by net revenue

    Args:
        df: Raw DataFrame returned by load_data().

    Returns:
        Dictionary containing named summary DataFrames and scalar KPIs.
    """
    # Aggregating dataset by category to compute standard performance metrics.
    df = compute_revenue(df)

    # ── KPI scalars ──────────────────────────────────────────────────────────
    kpis = {
        "total_revenue":   df["revenue"].sum(),
        "total_units":     df["units_sold"].sum(),
        "avg_discount":    df["discount_pct"].mean(),
        "total_orders":    len(df),
        "unique_products": df["product"].nunique(),
    }

    # ── Category-level aggregation ───────────────────────────────────────────
    category_summary = (
        df.groupby("category", as_index=False)
          .agg(
              total_revenue=("revenue", "sum"),
              total_units=("units_sold", "sum"),
              avg_discount=("discount_pct", "mean"),
              order_count=("order_id", "count"),
          )
          .sort_values("total_revenue", ascending=False)
          .reset_index(drop=True)
    )

    # ── Regional aggregation ─────────────────────────────────────────────────
    region_summary = (
        df.groupby("region", as_index=False)
          .agg(
              total_revenue=("revenue", "sum"),
              total_units=("units_sold", "sum"),
          )
          .sort_values("total_revenue", ascending=False)
          .reset_index(drop=True)
    )

    # ── Monthly revenue trend (preserve natural calendar order) ──────────────
    month_order = [
        "January", "February", "March", "April",
        "May", "June", "July", "August",
        "September", "October", "November", "December",
    ]
    monthly_trend = (
        df.groupby("month", as_index=False)
          .agg(monthly_revenue=("revenue", "sum"))
    )
    monthly_trend["month"] = pd.Categorical(
        monthly_trend["month"], categories=month_order, ordered=True
    )
    monthly_trend = monthly_trend.sort_values("month").reset_index(drop=True)

    # ── Top 5 products by revenue ─────────────────────────────────────────────
    top_products = (
        df.groupby("product", as_index=False)
          .agg(total_revenue=("revenue", "sum"), total_units=("units_sold", "sum"))
          .sort_values("total_revenue", ascending=False)
          .head(5)
          .reset_index(drop=True)
    )

    print("[INFO] Aggregation complete — KPIs, category, region, monthly, top-products computed.")
    return {
        "kpis":             kpis,
        "category_summary": category_summary,
        "region_summary":   region_summary,
        "monthly_trend":    monthly_trend,
        "top_products":     top_products,
    }


# ── PDF Construction Helpers ──────────────────────────────────────────────────

def _build_styles() -> dict:
    """Return a dictionary of named ParagraphStyles for consistent typography."""
    base = getSampleStyleSheet()

    styles = {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontSize=26,
            textColor=CLR_HEADER,
            spaceAfter=4,
            fontName="Helvetica-Bold",
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontSize=11,
            textColor=CLR_ACCENT,
            spaceAfter=2,
            fontName="Helvetica",
        ),
        "section": ParagraphStyle(
            "SectionHeading",
            parent=base["Heading2"],
            fontSize=13,
            textColor=CLR_HEADER,
            spaceBefore=14,
            spaceAfter=6,
            fontName="Helvetica-Bold",
        ),
        "body": ParagraphStyle(
            "BodyText",
            parent=base["Normal"],
            fontSize=9.5,
            textColor=CLR_TEXT,
            leading=14,
            fontName="Helvetica",
        ),
        "kpi_label": ParagraphStyle(
            "KPILabel",
            parent=base["Normal"],
            fontSize=9,
            textColor=CLR_ACCENT,
            fontName="Helvetica-Bold",
        ),
        "kpi_value": ParagraphStyle(
            "KPIValue",
            parent=base["Normal"],
            fontSize=18,
            textColor=CLR_HEADER,
            fontName="Helvetica-Bold",
        ),
        "footer": ParagraphStyle(
            "Footer",
            parent=base["Normal"],
            fontSize=8,
            textColor=colors.HexColor("#888888"),
            fontName="Helvetica",
        ),
    }
    return styles


def _make_table(headers: list, rows: list, col_widths: list) -> Table:
    """
    Construct a styled ReportLab Table with alternating row shading.

    Args:
        headers   : List of column header strings.
        rows      : List of row data lists.
        col_widths: List of column widths in points.

    Returns:
        Fully styled Table flowable.
    """
    table_data = [headers] + rows

    style_cmds = [
        # Header row
        ("BACKGROUND",    (0, 0), (-1, 0),  CLR_HEADER),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  CLR_WHITE),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  9),
        ("ALIGN",         (0, 0), (-1, 0),  "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, 0),  8),
        ("TOPPADDING",    (0, 0), (-1, 0),  8),
        # Data rows
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 1), (-1, -1), 9),
        ("ALIGN",         (1, 1), (-1, -1), "RIGHT"),
        ("ALIGN",         (0, 1), (0, -1),  "LEFT"),
        ("TOPPADDING",    (0, 1), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
        # Grid lines
        ("LINEBELOW",     (0, 0), (-1, 0),  1,   CLR_ACCENT),
        ("LINEBELOW",     (0, 1), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [CLR_WHITE, CLR_ROW_A]),
    ]

    tbl = Table(table_data, colWidths=col_widths)
    tbl.setStyle(TableStyle(style_cmds))
    return tbl


def _section_divider(story: list, styles: dict, heading: str) -> None:
    """Append a labelled horizontal rule section break to the story."""
    story.append(Spacer(1, 6))
    story.append(Paragraph(heading, styles["section"]))
    story.append(HRFlowable(
        width="100%", thickness=1.2,
        color=CLR_ACCENT, spaceAfter=8
    ))


# ── PDF Generation ────────────────────────────────────────────────────────────

def generate_pdf(analysis: dict, output_path: str) -> None:
    """
    Construct and render the formatted PDF report from aggregated analysis data.

    Pipeline:
        1. Configure document layout and typography.
        2. Render cover block (title, metadata, KPI cards).
        3. Render category, regional, monthly, and product tables.
        4. Commit document buffer to disk.

    Args:
        analysis   : Dictionary returned by analyze_data().
        output_path: Destination file path for the PDF output.
    """
    # Instantiating PDF document object and configuring global typography settings.
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = _build_styles()
    story  = []
    kpis   = analysis["kpis"]
    gen_ts = datetime.now().strftime("%d %B %Y, %I:%M %p")

    # ── Cover / Header block ──────────────────────────────────────────────────
    story.append(Paragraph(COMPANY_NAME, styles["subtitle"]))
    story.append(Paragraph(REPORT_TITLE, styles["title"]))
    story.append(Paragraph(f"Generated: {gen_ts}", styles["footer"]))
    story.append(HRFlowable(
        width="100%", thickness=2,
        color=CLR_ACCENT, spaceBefore=8, spaceAfter=16
    ))

    # ── Executive KPI Cards (rendered as a compact 5-column table) ────────────
    _section_divider(story, styles, "Executive Summary")

    kpi_labels = ["Total Revenue", "Units Sold", "Total Orders",
                  "Avg. Discount", "Products"]
    kpi_values = [
        f"Rs {kpis['total_revenue']:,.0f}",
        f"{kpis['total_units']:,}",
        str(kpis["total_orders"]),
        f"{kpis['avg_discount']:.1f}%",
        str(kpis["unique_products"]),
    ]

    kpi_row_labels = [[Paragraph(l, styles["kpi_label"]) for l in kpi_labels]]
    kpi_row_values = [[Paragraph(v, styles["kpi_value"]) for v in kpi_values]]

    kpi_table = Table(
        kpi_row_labels + kpi_row_values,
        colWidths=[3.3 * cm] * 5,
    )
    kpi_table.setStyle(TableStyle([
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("BACKGROUND",    (0, 0), (-1, -1), CLR_ROW_A),
        ("BOX",           (0, 0), (-1, -1), 1, CLR_ACCENT),
        ("INNERGRID",     (0, 0), (-1, -1), 0.5, colors.HexColor("#C8DCF0")),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 12))

    # ── Category Performance Table ────────────────────────────────────────────
    _section_divider(story, styles, "Performance by Category")

    cat_df = analysis["category_summary"]
    cat_headers = ["Category", "Revenue (Rs)", "Units Sold",
                   "Avg. Discount (%)", "Orders"]
    cat_rows = [
        [
            row["category"],
            f"{row['total_revenue']:,.0f}",
            f"{row['total_units']:,}",
            f"{row['avg_discount']:.1f}",
            str(int(row["order_count"])),
        ]
        for _, row in cat_df.iterrows()
    ]
    story.append(_make_table(cat_headers, cat_rows,
                             [4.5*cm, 3.5*cm, 3*cm, 3.5*cm, 2.5*cm]))

    # ── Regional Breakdown Table ──────────────────────────────────────────────
    _section_divider(story, styles, "Regional Breakdown")

    reg_df = analysis["region_summary"]
    reg_headers = ["Region", "Total Revenue (Rs)", "Total Units Sold",
                   "Revenue Share (%)"]
    total_rev = reg_df["total_revenue"].sum()
    reg_rows = [
        [
            row["region"],
            f"{row['total_revenue']:,.0f}",
            f"{row['total_units']:,}",
            f"{(row['total_revenue'] / total_rev * 100):.1f}",
        ]
        for _, row in reg_df.iterrows()
    ]
    story.append(_make_table(reg_headers, reg_rows,
                             [3.5*cm, 4.5*cm, 4.5*cm, 4.5*cm]))

    # ── Monthly Revenue Trend Table ───────────────────────────────────────────
    _section_divider(story, styles, "Monthly Revenue Trend")

    mon_df = analysis["monthly_trend"]
    mon_headers = ["Month", "Revenue (Rs)", "Share of Period (%)"]
    period_total = mon_df["monthly_revenue"].sum()
    mon_rows = [
        [
            str(row["month"]),
            f"{row['monthly_revenue']:,.0f}",
            f"{(row['monthly_revenue'] / period_total * 100):.1f}",
        ]
        for _, row in mon_df.iterrows()
    ]
    story.append(_make_table(mon_headers, mon_rows,
                             [5*cm, 5*cm, 5*cm]))

    # ── Top 5 Products Table ──────────────────────────────────────────────────
    story.append(PageBreak())
    _section_divider(story, styles, "Top 5 Products by Revenue")

    top_df = analysis["top_products"]
    top_headers = ["Rank", "Product", "Total Revenue (Rs)", "Units Sold"]
    top_rows = [
        [
            str(i + 1),
            row["product"],
            f"{row['total_revenue']:,.0f}",
            f"{row['total_units']:,}",
        ]
        for i, (_, row) in enumerate(top_df.iterrows())
    ]
    story.append(_make_table(top_headers, top_rows,
                             [2*cm, 5*cm, 5*cm, 4*cm]))

    # ── Footer note ───────────────────────────────────────────────────────────
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor("#AAAAAA")))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"<i>This report was auto-generated by {COMPANY_NAME}'s Reporting Pipeline on {gen_ts}. "
        "All figures are derived from transactional CSV data and rounded to the nearest integer.</i>",
        styles["footer"]
    ))

    # Committing PDF buffer to disk. Output path: ./generated_report.pdf
    doc.build(story)
    print(f"[SUCCESS] Report written to: {output_path}")


# ── Orchestration Pipeline ────────────────────────────────────────────────────

def run_pipeline(csv_path: str = INPUT_CSV,
                 pdf_path: str = OUTPUT_PDF) -> None:
    """
    Entry point — orchestrates the full ETL-to-PDF reporting pipeline.

    Stages:
        1. load_data()    → raw DataFrame
        2. analyze_data() → aggregated metrics dict
        3. generate_pdf() → formatted PDF on disk
    """
    print("=" * 55)
    print(f"  {REPORT_TITLE} — Pipeline Start")
    print("=" * 55)

    try:
        raw_df   = load_data(csv_path)
        analysis = analyze_data(raw_df)
        generate_pdf(analysis, pdf_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[PIPELINE ABORTED] {exc}", file=sys.stderr)
        sys.exit(1)

    print("=" * 55)
    print("  Pipeline Complete")
    print("=" * 55)


if __name__ == "__main__":
    run_pipeline()
