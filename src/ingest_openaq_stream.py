import json
import os
import time
from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAQ_API_KEY")
BASE_URL = "https://api.openaq.org/v3"

if not API_KEY:
    raise RuntimeError("OPENAQ_API_KEY not found in .env")

# -----------------------
# Config (env overrides)
# -----------------------
SENSOR_ID = int(os.getenv("SENSOR_ID", "5049"))
PARAMETER = os.getenv("PARAMETER", "pm25")
INTERVAL = os.getenv("INTERVAL", "hourly").lower()  # hourly or daily
POLL_SECONDS = int(os.getenv("OPENAQ_POLL_SECONDS", "600"))
LAG_HOURS = int(os.getenv("LAG_HOURS", "24"))
WINDOW_HOURS = int(os.getenv("WINDOW_HOURS", "24"))
LIMIT = int(os.getenv("LIMIT", "200"))
RAW_DIR = os.getenv("RAW_DIR", "data/raw/hourly")

ALLOW_FALLBACK = os.getenv("ALLOW_FALLBACK", "1") != "0"
FALLBACK_DAYS = int(os.getenv("FALLBACK_DAYS", "30"))
ONE_SHOT = os.getenv("ONE_SHOT", "0") == "1"
SAVE_EMPTY = os.getenv("SAVE_EMPTY", "0") == "1"

HEADERS = {
    "X-API-Key": API_KEY,
    "accept": "application/json"
}

os.makedirs(RAW_DIR, exist_ok=True)


def to_utc_iso(dt):
    return dt.replace(microsecond=0).isoformat() + "Z"


def parse_utc(ts):
    if not ts:
        return None
    # Example: 2024-01-13T00:00:00Z
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def fetch_measurements(interval, datetime_from, datetime_to):
    url = f"{BASE_URL}/sensors/{SENSOR_ID}/measurements/{interval}"
    params = {
        "parameter": PARAMETER,
        "datetime_from": to_utc_iso(datetime_from),
        "datetime_to": to_utc_iso(datetime_to),
        "limit": LIMIT
    }
    response = requests.get(url, headers=HEADERS, params=params, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"API error {response.status_code}: {response.text}")
    return response.json()


def get_latest_datetime(results):
    latest = None
    for row in results:
        dt = parse_utc(
            row.get("period", {})
            .get("datetimeTo", {})
            .get("utc")
        )
        if dt and (latest is None or dt > latest):
            latest = dt
    return latest


def save_raw(data, interval_used, now):
    ts = now.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        RAW_DIR,
        f"pm25_sensor_{SENSOR_ID}_{interval_used}_{ts}.json"
    )
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    return output_file


def main():
    print("Starting OpenAQ micro-batch ingest")
    print(f"Sensor ID: {SENSOR_ID} | Interval: {INTERVAL} | Poll: {POLL_SECONDS}s")

    last_time = datetime.utcnow() - timedelta(hours=LAG_HOURS + WINDOW_HOURS)

    while True:
        try:
            now = datetime.utcnow() - timedelta(hours=LAG_HOURS)
            datetime_from = last_time
            datetime_to = now

            print(f"Fetching {INTERVAL} data from {to_utc_iso(datetime_from)} -> {to_utc_iso(datetime_to)}")
            data = fetch_measurements(INTERVAL, datetime_from, datetime_to)
            results = data.get("results", [])
            interval_used = INTERVAL

            if not results and ALLOW_FALLBACK:
                fallback_from = now - timedelta(days=FALLBACK_DAYS)
                print(f"No data, fallback to daily window {FALLBACK_DAYS} days")
                data = fetch_measurements("daily", fallback_from, datetime_to)
                results = data.get("results", [])
                interval_used = "daily"

            if results or SAVE_EMPTY:
                output_file = save_raw(data, interval_used, now)
                print(f"Saved {len(results)} records -> {output_file}")
            else:
                print("No new data in this interval")

            latest = get_latest_datetime(results)
            last_time = latest + timedelta(seconds=1) if latest else now

        except Exception as exc:
            print(f"Ingest error: {exc}")

        if ONE_SHOT:
            break

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
