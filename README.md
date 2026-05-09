# Task 1: API Integration and Data Visualization

## Overview
This repository contains the solution for **Task 1** of the CODTECH IT Solutions Python Internship. The objective of this project is to build a robust data pipeline that fetches real-time forecast data from the OpenWeatherMap API, processes the JSON responses, and generates an analytical dashboard using Python's visualization libraries.

---

## Features
* **Automated Data Acquisition:** Retrieves 5-day / 3-hour forecast data for multiple cities.
* **Robust Error Handling:** Safely catches and logs HTTP errors, timeouts, and connection drops without crashing.
* **Secure Credential Management:** Built to extract API keys securely from environment variables rather than hardcoding them into the script.
* **Programmatic Visualization:** Generates a 3-panel high-resolution dashboard comparing temperature trends, average temperatures, and average humidity across different regions.

---

## Prerequisites
Before running this script, ensure you have the following installed:
* **Python 3.8+**
* The required Python libraries (can be installed via the command below):

    ```bash
    pip install requests matplotlib seaborn
    ```

### API Key Setup
This script requires a free API key from [OpenWeatherMap](https://openweathermap.org/).
1. Create an account and generate an API key.
2. Store the key securely based on your environment:
   * **Local IDE:** Create a `.env` file or export it to your system environment as `OPENWEATHER_API_KEY="your_actual_key_here"`.
   * **Google Colab:** Add it to the built-in Secrets tab with the name `OPENWEATHER_API_KEY` and toggle notebook access.

*Note: Newly generated OpenWeatherMap API keys may take 1 to 2 hours to activate on their servers.*

---

## Usage
To execute the pipeline and generate the dashboard, run the main Python script from your terminal:

```bash
python task_1_weather_dashboard.py
