"""
Feature Engineering for PM2.5 Data
- Creates lag features (lag_1 to lag_24)
- Creates time-based features (hour, day_of_week, month)
- IMPORTANT: This multiplies data for ML training
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour as hr, dayofweek, month, lag, when
from pyspark.sql.window import Window

# -----------------------
# Spark Session
# -----------------------
spark = SparkSession.builder \
    .appName("PM25 Feature Engineering") \
    .master("local[*]") \
    .config("spark.driver.host", "127.0.0.1") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# -----------------------
# Paths
# -----------------------
# Use hourly data for more rows!
INPUT_PATH = "data/processed/pm25_hourly"
OUTPUT_PATH = "data/processed/features"

# -----------------------
# Read hourly data
# -----------------------
print("📂 Reading hourly data...")
df = spark.read.parquet(INPUT_PATH)
print(f"   Hourly records: {df.count()}")

# -----------------------
# Feature engineering
# -----------------------
print("\n🔧 Creating features...")

# Window spec: order by datetime within each sensor
# This ensures lag features are computed correctly per sensor
window_spec = Window.partitionBy("sensor_id").orderBy("datetime")

df_features = df

# Create lag_1 to lag_24 features (sliding window)
# This creates training samples from time-series data
for i in range(1, 25):
    df_features = df_features.withColumn(
        f"lag_{i}",
        lag(col("pm25"), i).over(window_spec)
    )
    if i % 6 == 0:
        print(f"   Created lag_{i} features...")

# Drop rows with null lag values (first 24 rows per sensor)
# This is necessary because lag features are null for early rows
print("\n🧹 Dropping rows with null lag values...")
df_features = df_features.dropna(
    subset=[f"lag_{i}" for i in range(1, 25)] + ["pm25"]
)

print(f"   Features records after dropna: {df_features.count()}")

# Select final columns for ML
# Features: hour, day_of_week, month, lag_1..lag_24
# Target: pm25
feature_cols = ["hour", "day_of_week", "month"] + [f"lag_{i}" for i in range(1, 25)]

df_final = df_features.select(
    "sensor_id",
    "datetime",
    "pm25",
    "hour",
    "day_of_week",
    "month",
    *[f"lag_{i}" for i in range(1, 25)]
)

# Order by sensor and time
df_final = df_final.orderBy("sensor_id", "datetime")

# -----------------------
# Save features
# -----------------------
print("\n💾 Saving features...")
df_final.write.mode("overwrite").parquet(OUTPUT_PATH)

print("\n" + "=" * 60)
print("📊 FEATURE ENGINEERING SUMMARY")
print("=" * 60)
print(f"Input records (hourly): {df.count()}")
print(f"Output records (with lags): {df_final.count()}")
print(f"Features created: {len(feature_cols)}")
print(f"  - hour, day_of_week, month")
print(f"  - lag_1 to lag_24 (sliding window)")
print(f"Output path: {OUTPUT_PATH}")
print("=" * 60)

spark.stop()
