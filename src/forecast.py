import openmeteo_requests
import requests_cache
from retry_requests import retry
import numpy as np
import pandas as pd

# Country coordinates for all 30 countries
COUNTRY_COORDS = {
    "Afghanistan": (34.5, 69.2),
    "Australia": (-25.3, 133.8),
    "Bangladesh": (23.7, 90.4),
    "Brazil": (-14.2, -51.9),
    "Canada": (56.1, -106.3),
    "China": (35.9, 104.2),
    "Cuba": (21.5, -77.8),
    "Egypt": (26.8, 30.8),
    "Ethiopia": (9.1, 40.5),
    "France": (46.2, 2.2),
    "Germany": (51.2, 10.4),
    "India": (20.6, 79.0),
    "Indonesia": (-0.8, 113.9),
    "Iran": (32.4, 53.7),
    "Italy": (41.9, 12.6),
    "Japan": (36.2, 138.3),
    "Mexico": (23.6, -102.6),
    "Myanmar": (17.1, 96.9),
    "Nigeria": (9.1, 8.7),
    "Pakistan": (30.4, 69.3),
    "Philippines": (12.9, 121.8),
    "Russia": (61.5, 105.3),
    "Saudi Arabia": (23.9, 45.1),
    "South Korea": (35.9, 127.8),
    "Spain": (40.5, -3.7),
    "Turkey": (38.9, 35.2),
    "Ukraine": (48.4, 31.2),
    "United Kingdom": (55.4, -3.4),
    "United States": (37.1, -95.7),
    "Venezuela": (6.4, -66.6),
}

# Weather code → index (matching your model training)
def weathercode_to_index(code):
    if code == 0:
        return 0   # Clear
    elif code in [1, 2, 3]:
        return 3   # Cloudy
    elif code in [51, 53, 55, 61, 63, 65, 80, 81, 82]:
        return 1   # Rain
    elif code in [71, 73, 75, 77, 85, 86]:
        return 4   # Snow
    else:
        return 2   # Storm


def get_7day_forecast(country, model, df):
    if country not in COUNTRY_COORDS:
        return None

    lat, lon = COUNTRY_COORDS[country]

    # Fetch weather
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    om = openmeteo_requests.Client(session=retry_session)

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["weathercode", "precipitation"],
        "forecast_days": 7
    }

    responses = om.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
    r = responses[0]
    hourly = r.Hourly()

    weathercodes = hourly.Variables(0).ValuesAsNumpy()

    # Get country averages from historical data
    country_data = df[df["country"] == country]
    avg = country_data.mean(numeric_only=True)

    # Predict for all 168 hours (7 days x 24 hours)
    results = []
    for i in range(168):
        day = i // 24
        hour = i % 24
        weather_index = weathercode_to_index(weathercodes[i])

        features = np.array([[
            hour, 3, 6,
            avg["bgp_signal"],
            avg["active_probing"],
            avg["traffic_drop_pct"],
            avg["latency_ms"],
            weather_index,
            avg["country_index"],
            avg["base_outage_prob"],
        ]])

        prob = model.predict_proba(features)[0][1]
        results.append({
            "day": day,
            "hour": hour,
            "weather_index": weather_index,
            "risk": round(prob * 100, 1)
        })

    return pd.DataFrame(results)