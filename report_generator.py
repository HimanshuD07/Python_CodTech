# First, we need to install the missing dependency

!pip install -q reportlab

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

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load tabular data from a CSV file into a Pandas DataFrame.
    """
    required_cols = {"region", "category", "product",
                     "units_sold", "unit_price", "discount_pct", "month"}

    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Could not find the file: '{filepath}'. "
            "Make sure the CSV is in the same folder as this script."
        )
    except pd.errors.EmptyDataError:
        raise ValueError(f"The file {filepath} seems to be empty.")

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print(f"Loaded {len(df)} records successfully.")
    return df

def compute_revenue(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["revenue"] = (
        df["units_sold"] * df["unit_price" ] * (1 - df["discount_pct"] / 100)
    )
    return df

def analyze_data(df: pd.DataFrame) -> dict:
    df = compute_revenue(df)

    kpis = {
        "total_revenue":   df["revenue"].sum(),
        "total_units":     df["units_sold"].sum(),
        "avg_discount":    df["discount_pct"].mean(),
        "total_orders":    len(df),
        "unique_products": df["product"].nunique(),
    }

    category_summary = (
        df.groupby("category", as_index=False)
          .agg(
              total_revenue=("revenue", "sum"),
              total_units=("units_sold", "sum"),
              avg_discount=("discount_pct", "mean"),
              order_count=("month", "count"),
          )
          .sort_values("total_revenue", ascending=False)
          .reset_index(drop=True)
    )

    region_summary = (
        df.groupby("region", as_index=False)
          .agg(
              total_revenue=("revenue", "sum"),
              total_units=("units_sold", "sum"),
          )
          .sort_values("total_revenue", ascending=False)
          .reset_index(drop=True)
    )

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

    top_products = (
        df.groupby("product", as_index=False)
          .agg(total_revenue=("revenue", "sum"), total_units=("units_sold", "sum"))
          .sort_values("total_revenue", ascending=False)
          .head(5)
          .reset_index(drop=True)
    )

    return {
        "kpis":             kpis,
        "category_summary": category_summary,
        "region_summary":   region_summary,
        "monthly_trend":    monthly_trend,
        "top_products":     top_products,
    }

def _build_styles() -> dict:
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle("ReportTitle", parent=base["Title"], fontSize=26, textColor=CLR_HEADER),
        "subtitle": ParagraphStyle("Subtitle", parent=base["Normal"], fontSize=11, textColor=CLR_ACCENT),
        "section": ParagraphStyle("SectionHeading", parent=base["Heading2"], fontSize=13, textColor=CLR_HEADER),
        "body": ParagraphStyle("BodyText", parent=base["Normal"], fontSize=9.5, textColor=CLR_TEXT),
        "kpi_label": ParagraphStyle("KPILabel", parent=base["Normal"], fontSize=9, textColor=CLR_ACCENT),
        "kpi_value": ParagraphStyle("KPIValue", parent=base["Normal"], fontSize=18, textColor=CLR_HEADER),
        "footer": ParagraphStyle("Footer", parent=base["Normal"], fontSize=8, textColor=colors.HexColor("#888888")),
    }
    return styles

def _make_table(headers: list, rows: list, col_widths: list) -> Table:
    table_data = [headers] + rows
    style_cmds = [
        ("BACKGROUND",    (0, 0), (-1, 0),  CLR_HEADER),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  CLR_WHITE),
        ("ALIGN",         (0, 0), (-1, 0),  "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [CLR_WHITE, CLR_ROW_A]),
        ("LINEBELOW",     (0, 0), (-1, 0),  1,   CLR_ACCENT),
    ]
    tbl = Table(table_data, colWidths=col_widths)
    tbl.setStyle(TableStyle(style_cmds))
    return tbl

def _section_divider(story: list, styles: dict, heading: str) -> None:
    story.append(Paragraph(heading, styles["section"]))
    story.append(HRFlowable(width="100%", thickness=1.2, color=CLR_ACCENT, spaceAfter=8))

def generate_pdf(analysis: dict, output_path: str) -> None:
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )

    styles = _build_styles()
    story  = []
    kpis   = analysis["kpis"]
    gen_ts = datetime.now().strftime("%d %B %Y, %I:%M %p")

    story.append(Paragraph(COMPANY_NAME, styles["subtitle"]))
    story.append(Paragraph(REPORT_TITLE, styles["title"]))
    story.append(Paragraph(f"Generated: {gen_ts}", styles["footer"]))
    story.append(Spacer(1, 12))

    _section_divider(story, styles, "Executive Summary")
    story.append(Spacer(1, 12))

    cat_df = analysis["category_summary"]
    cat_headers = ["Category", "Revenue (Rs)", "Units Sold"]
    cat_rows = [[r["category"], f"{r['total_revenue']:,.0f}", str(r["total_units"])] for _, r in cat_df.iterrows()]
    story.append(_make_table(cat_headers, cat_rows, [6*cm, 5*cm, 4*cm]))

    doc.build(story)
    print(f"Done! PDF report saved as {output_path}")

def run_pipeline(csv_path: str = INPUT_CSV, pdf_path: str = OUTPUT_PDF) -> None:
    try:
        raw_df   = load_data(csv_path)
        analysis = analyze_data(raw_df)
        generate_pdf(analysis, pdf_path)
    except Exception as exc:
        print(f"Something went wrong: {exc}")

if __name__ == "__main__":

    # Creating a sample file if it doesn't exist

    if not Path(INPUT_CSV).exists():
        pd.DataFrame({
            'region': ['North', 'South'],
            'category': ['Tech', 'Home'],
            'product': ['Laptop', 'Lamp'],
            'units_sold': [10, 5],
            'unit_price': [50000, 1000],
            'discount_pct': [5, 10],
            'month': ['January', 'January']
        }).to_csv(INPUT_CSV, index=False)

    run_pipeline()

    # TODO: Add dynamic chart generation using Matplotlib to include in the PDF
