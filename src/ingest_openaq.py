import requests
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAQ_API_KEY")
BASE_URL = "https://api.openaq.org/v3"

if not API_KEY:
    raise ValueError("OPENAQ_API_KEY not found. Check .env file")

HEADERS = {
    "accept": "application/json",
    "X-API-Key": API_KEY
}

SENSOR_ID = int(os.getenv("SENSOR_ID", "5049"))
LIMIT = int(os.getenv("LIMIT", "100"))

def to_utc_iso(dt):
    """Convert datetime to UTC ISO format for API"""
    return dt.replace(microsecond=0).isoformat() + "Z"

def fetch_daily_measurements(sensor_id, datetime_from=None, datetime_to=None, limit=100, page=1):
    url = f"{BASE_URL}/sensors/{sensor_id}/measurements/daily"
    params = {
        "limit": limit,
        "page": page
    }
    if datetime_from:
        params["datetime_from"] = to_utc_iso(datetime_from)
    if datetime_to:
        params["datetime_to"] = to_utc_iso(datetime_to)

    response = requests.get(url, headers=HEADERS, params=params)
    response.raise_for_status()
    return response.json()

def save_raw(data, sensor_id, suffix=""):
    os.makedirs("data/raw", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"data/raw/pm25_sensor_{sensor_id}_{suffix}_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {path}")
    return path

def fetch_all_pages(sensor_id, datetime_from=None, datetime_to=None):
    """Fetch all available pages"""
    all_results = []
    page = 1

    while True:
        print(f"Fetching page {page}...")
        data = fetch_daily_measurements(
            sensor_id,
            datetime_from=datetime_from,
            datetime_to=datetime_to,
            limit=LIMIT,
            page=page
        )

        results = data.get("results", [])
        all_results.extend(results)

        meta = data.get("meta", {})
        found = meta.get("found", 0)
        limit = meta.get("limit", 100)

        print(f"  -> Got {len(results)} records (total: {found})")

        if len(results) < limit:
            break  # No more pages
        page += 1

    # Save combined data
    combined_data = {
        "meta": {**meta, "page": 1, "limit": len(all_results)},
        "results": all_results
    }
    return combined_data

if __name__ == "__main__":
    # Default: fetch all available data (no date filter)
    # Set DATETIME_FROM/DATETIME_TO env vars to filter by date range
    datetime_from_str = os.getenv("DATETIME_FROM", "")
    datetime_to_str = os.getenv("DATETIME_TO", "")

    datetime_from = datetime.strptime(datetime_from_str, "%Y-%m-%d") if datetime_from_str else None
    datetime_to = datetime.strptime(datetime_to_str, "%Y-%m-%d") if datetime_to_str else None

    date_range_str = f"{datetime_from.date() if datetime_from else 'beginning'} to {datetime_to.date() if datetime_to else 'now'}"

    print(f"Fetching PM2.5 data for sensor {SENSOR_ID}")
    print(f"Date range: {date_range_str}")
    print("-" * 50)

    data = fetch_all_pages(SENSOR_ID, datetime_from=datetime_from, datetime_to=datetime_to)

    if data.get("results"):
        save_raw(data, SENSOR_ID, "all")
        print(f"\nTotal records fetched: {len(data['results'])}")
    else:
        print("No data found for the specified date range")
