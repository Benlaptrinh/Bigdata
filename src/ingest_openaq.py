import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAQ_API_KEY")   # <-- SỬA Ở ĐÂY
BASE_URL = "https://api.openaq.org/v3"

if not API_KEY:
    raise ValueError("OPENAQ_API_KEY not found. Check .env file")

HEADERS = {
    "accept": "application/json",
    "X-API-Key": API_KEY
}

def fetch_daily_measurements(sensor_id, limit=100, page=1):
    url = f"{BASE_URL}/sensors/{sensor_id}/measurements/daily"
    params = {
        "limit": limit,
        "page": page
    }
    response = requests.get(url, headers=HEADERS, params=params)
    response.raise_for_status()
    return response.json()

def save_raw(data, sensor_id, page):
    os.makedirs("data/raw", exist_ok=True)
    path = f"data/raw/pm25_sensor_{sensor_id}_page_{page}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {path}")

if __name__ == "__main__":
    SENSOR_ID = 5049
    PAGE = 1

    data = fetch_daily_measurements(SENSOR_ID, page=PAGE)
    save_raw(data, SENSOR_ID, PAGE)
