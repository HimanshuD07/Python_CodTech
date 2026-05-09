# **Task 2: Automated Report Generation**

```markdown
# Task 2: Automated Report Generation

## Overview
This repository contains the solution for **Task 2** of the CODTECH IT Solutions Python Internship. The objective of this project is to build an automated data pipeline that ingests raw tabular data, aggregates key performance indicators (KPIs) using Pandas, and dynamically generates a formatted, professional PDF report.

---

## Features
* **Automated Data Processing:** Uses Pandas to compute revenue, calculate averages, and aggregate sales data by category and region.
* **Dynamic PDF Construction:** Utilizes ReportLab to programmatically generate documents with customized styling, headers, and structured data tables.
* **Fail-Safe Execution:** Includes built-in mechanisms to catch missing files or malformed data, and automatically generates a sample dataset if no input CSV is found.

---

## Prerequisites
Before running this script, ensure you have Python installed along with the required data processing and PDF generation libraries.

```bash
pip install pandas reportlab
