import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

print("Generating synthetic internet outage dataset...")

np.random.seed(42)
os.makedirs("data", exist_ok=True)

# Countries with their base outage probability (realistic values)
countries = [
    ("Pakistan", "PK", 0.35),
    ("India", "IN", 0.25),
    ("United States", "US", 0.05),
    ("Germany", "DE", 0.04),
    ("Brazil", "BR", 0.20),
    ("Nigeria", "NG", 0.40),
    ("Iran", "IR", 0.55),
    ("Russia", "RU", 0.30),
    ("China", "CN", 0.28),
    ("United Kingdom", "GB", 0.04),
    ("France", "FR", 0.05),
    ("Turkey", "TR", 0.22),
    ("Bangladesh", "BD", 0.32),
    ("Egypt", "EG", 0.30),
    ("Indonesia", "ID", 0.25),
    ("Ethiopia", "ET", 0.50),
    ("Mexico", "MX", 0.18),
    ("Philippines", "PH", 0.28),
    ("Ukraine", "UA", 0.45),
    ("Afghanistan", "AF", 0.60),
    ("Japan", "JP", 0.03),
    ("South Korea", "KR", 0.04),
    ("Australia", "AU", 0.05),
    ("Canada", "CA", 0.05),
    ("Italy", "IT", 0.08),
    ("Spain", "ES", 0.07),
    ("Saudi Arabia", "SA", 0.15),
    ("Venezuela", "VE", 0.45),
    ("Myanmar", "MM", 0.50),
    ("Cuba", "CU", 0.55),
]

weather_conditions = ["Clear", "Cloudy", "Rain", "Storm", "Snow"]
weather_outage_multiplier = {
    "Clear": 1.0,
    "Cloudy": 1.1,
    "Rain": 1.3,
    "Storm": 1.7,
    "Snow": 1.4,
}

records = []

# Generate 500 records per country = 15,000+ rows total
start_date = datetime(2024, 1, 1)
end_date = datetime(2026, 5, 1)
date_range_days = (end_date - start_date).days

for country_name, iso_code, base_prob in countries:
    for _ in range(2000):
        # Random timestamp
        random_days = np.random.randint(0, date_range_days)
        random_hours = np.random.randint(0, 24)
        timestamp = start_date + timedelta(days=int(random_days), hours=int(random_hours))

        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        month = timestamp.month

        # Night hours increase outage probability
        hour_multiplier = 1.4 if (hour >= 22 or hour <= 4) else 1.0

        # Weekend slightly less outages
        day_multiplier = 0.9 if day_of_week >= 5 else 1.0

        # Weather effect
        weather = np.random.choice(
            weather_conditions, p=[0.40, 0.25, 0.20, 0.10, 0.05]
        )
        weather_multiplier = weather_outage_multiplier[weather]

        # Final outage probability
        final_prob = min(
            base_prob * hour_multiplier * day_multiplier * weather_multiplier, 0.95
        )

        # Signal values (lower = worse connectivity)
        bgp_signal = round(np.random.uniform(60, 100) * (1 - base_prob * 0.5), 2)
        active_probing = round(np.random.uniform(50, 100) * (1 - base_prob * 0.4), 2)
        traffic_drop = round(np.random.uniform(0, 50) * base_prob * weather_multiplier, 2)
        latency = round(np.random.uniform(20, 200) * (1 + base_prob), 2)

        # Outage label
        outage = 1 if np.random.random() < final_prob else 0

        records.append({
            "country": country_name,
            "iso_code": iso_code,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "hour": hour,
            "day_of_week": day_of_week,
            "month": month,
            "weather": weather,
            "bgp_signal": bgp_signal,
            "active_probing": active_probing,
            "traffic_drop_pct": traffic_drop,
            "latency_ms": latency,
            "base_outage_prob": base_prob,
            "outage": outage,
        })

df = pd.DataFrame(records)
df = df.sample(frac=1).reset_index(drop=True)  # shuffle rows

output_path = "data/raw_outage_data.csv"
df.to_csv(output_path, index=False)

print(f"Total records generated: {len(df)}")
print(f"Outage distribution:\n{df['outage'].value_counts()}")
print(f"Countries: {df['country'].nunique()}")
print(f"Saved to {output_path}")