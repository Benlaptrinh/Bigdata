import os
import time
import json
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# =========================
# Load environment variables
# =========================
load_dotenv()

API_KEY = os.getenv("OPENAQ_API_KEY")
BASE_URL = "https://api.openaq.org/v3"

if not API_KEY:
    raise RuntimeError("❌ OPENAQ_API_KEY not found in .env")

HEADERS = {
    "X-API-Key": API_KEY
}

# =========================
# Config
# =========================
SENSOR_ID = 5049          # PM2.5 sensor bạn đang dùng
PARAMETER = "pm25"
INTERVAL_SECONDS = 600    # 10 phút (demo streaming)
RAW_DIR = "data/raw/hourly"

os.makedirs(RAW_DIR, exist_ok=True)

# =========================
# Helper: fetch hourly data
# =========================
def fetch_hourly_data(sensor_id, datetime_from, datetime_to):
    url = f"{BASE_URL}/sensors/{sensor_id}/measurements/hourly"
    params = {
        "parameter": PARAMETER,
        "datetime_from": datetime_from,
        "datetime_to": datetime_to,
        "limit": 100
    }

    response = requests.get(url, headers=HEADERS, params=params, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(f"API error {response.status_code}: {response.text}")

    return response.json()

# =========================
# Main streaming loop
# =========================
def main():
    print("🚀 Starting OpenAQ streaming ingest (Level 2 - micro-batch)")
    print(f"⏱ Interval: {INTERVAL_SECONDS} seconds")

    # Lần đầu: lấy dữ liệu 1 giờ trước
    last_time = datetime.utcnow() - timedelta(hours=1)

    while True:
        try:
            now = datetime.utcnow()

            datetime_from = last_time.isoformat() + "Z"
            datetime_to = now.isoformat() + "Z"

            print(f"\n📡 Fetching data from {datetime_from} → {datetime_to}")

            data = fetch_hourly_data(
                SENSOR_ID,
                datetime_from,
                datetime_to
            )

            if "results" in data and len(data["results"]) > 0:
                ts = now.strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(
                    RAW_DIR,
                    f"pm25_sensor_{SENSOR_ID}_{ts}.json"
                )

                with open(output_file, "w") as f:
                    json.dump(data, f, indent=2)

                print(f"✅ Saved {len(data['results'])} records → {output_file}")
            else:
                print("⚠️ No new data in this interval")

            last_time = now

        except Exception as e:
            print("❌ Error during ingest:", str(e))
            print("⏳ Will retry in next interval")

        print(f"🕒 Sleeping {INTERVAL_SECONDS} seconds...\n")
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
