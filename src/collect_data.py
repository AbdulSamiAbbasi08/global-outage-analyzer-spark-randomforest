import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os

print("Starting data collection...")

# Output path
os.makedirs("data", exist_ok=True)

# Countries to collect (name, ISO code)
countries = [
    ("Pakistan", "PK"), ("India", "IN"), ("United States", "US"),
    ("Germany", "DE"), ("Brazil", "BR"), ("Nigeria", "NG"),
    ("Iran", "IR"), ("Russia", "RU"), ("China", "CN"),
    ("United Kingdom", "GB"), ("France", "FR"), ("Turkey", "TR"),
    ("Bangladesh", "BD"), ("Egypt", "EG"), ("Indonesia", "ID"),
    ("Ethiopia", "ET"), ("Mexico", "MX"), ("Philippines", "PH"),
    ("Ukraine", "UA"), ("Afghanistan", "AF")
]

# Date range — last 6 months
end_date = datetime.utcnow()
start_date = end_date - timedelta(days=180)

start_str = start_date.strftime("%Y-%m-%dT%H:%M:%S")
end_str = end_date.strftime("%Y-%m-%dT%H:%M:%S")

all_records = []

for country_name, iso_code in countries:
    print(f"Fetching: {country_name}...")
    try:
        url = f"https://api.ioda.caida.org/v2/signals/raw/country/{iso_code}"
        params = {
            "from": start_str,
            "until": end_str,
        }
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if data and "data" in data:
                for series in data["data"]:
                    datasource = series.get("datasource", "unknown")
                    for point in series.get("values", []):
                        if point is not None:
                            all_records.append({
                                "country": country_name,
                                "iso_code": iso_code,
                                "datasource": datasource,
                                "value": point[1] if isinstance(point, list) else point,
                                "timestamp": point[0] if isinstance(point, list) else None,
                            })
        else:
            print(f"  Failed: status {response.status_code}")
    except Exception as e:
        print(f"  Error: {e}")
    
    time.sleep(1)

print(f"\nTotal records collected: {len(all_records)}")

df = pd.DataFrame(all_records)
df.to_csv("data/raw_outage_data.csv", index=False)
print("Saved to data/raw_outage_data.csv")