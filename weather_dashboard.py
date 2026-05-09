import sys
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
from google.colab import userdata

# --- Settings ---

# Using Colab secrets to keep the key safe
try:
    API_KEY = userdata.get('OPENWEATHER_API_KEY')
except userdata.SecretNotFoundError:
    # Fallback key if secret isn't set up yet
    API_KEY = "aff7a977f7b045cc158646d8bf65b41d"

BASE_URL = "https://api.openweathermap.org/data/2.5/forecast"
# TODO: maybe add more cities later if needed
CITIES = ["Nagpur", "Mumbai", "Delhi", "Pune", "Hyderabad"]
UNITS = "metric"
OUTPUT_FILE = "weather_dashboard.png"

# --- Data Functions ---

def fetch_data(city):
    """Fetches the forecast for a specific city from the API."""
    params = {
        "q": city,
        "appid": API_KEY,
        "units": UNITS,
        "cnt": 8
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Could not get data for {city}: {e}")
        return {}

def process_data(raw_data):
    """Extracts the time, temp, and humidity from the raw JSON response."""
    timestamps, temperatures, humidity_values = [], [], []

    if not raw_data or "list" not in raw_data:
        return timestamps, temperatures, humidity_values

    for entry in raw_data["list"]:
        try:
            timestamps.append(datetime.fromtimestamp(entry["dt"]))
            temperatures.append(entry["main"]["temp"])
            humidity_values.append(entry["main"]["humidity"])
        except KeyError:
            continue

    return timestamps, temperatures, humidity_values

def get_city_averages(cities):
    """Loops through cities to get their average metrics."""
    valid_cities, avg_temps, avg_humidity = [], [], []

    for city in cities:
        raw = fetch_data(city)
        _, temps, hums = process_data(raw)

        if temps:
            avg_temps.append(round(sum(temps) / len(temps), 2))
            avg_humidity.append(round(sum(hums) / len(hums), 2))
            valid_cities.append(city)

    return valid_cities, avg_temps, avg_humidity

# --- Charting ---

def create_dashboard(times, temps, hums, city_names, avg_t, avg_h):
    """Plots the weather data into a 3-panel dashboard."""
    sns.set_theme(style="darkgrid", palette="muted")
    palette = sns.color_palette("coolwarm", len(city_names))

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6), dpi=150)
    fig.suptitle("Weather Forecast Dashboard", fontsize=15, fontweight="bold", y=1.02)

    # Panel 1: Main City Trend
    ax1 = axes[0]
    ax1.plot(times, temps, marker="o", linewidth=2, color="#E74C3C", label="Temp (°C)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax1.set_title(f"24-Hour Trend: {city_names[0] if city_names else ''}")

    # Panel 2: Temp Comparison
    axes[1].bar(city_names, avg_t, color=palette)
    axes[1].set_title("Average Temp per City")

    # Panel 3: Humidity Comparison
    axes[2].bar(city_names, avg_h, color=palette)
    axes[2].set_title("Average Humidity per City")

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, bbox_inches="tight")
    print(f"All done! Saved the dashboard to {OUTPUT_FILE}")
    plt.show()

# --- Main Run ---

if __name__ == "__main__":
    print("Starting the weather script...")

    # Get trend data for the first city
    main_city = CITIES[0]
    data = fetch_data(main_city)
    times, temps, hums = process_data(data)

    if not temps:
        print(f"Looks like we couldn't find data for {main_city}. Is the API key correct?")
    else:
        # Get comparison stats for all cities
        city_names, avg_t, avg_h = get_city_averages(CITIES)
        create_dashboard(times, temps, hums, city_names, avg_t, avg_h)