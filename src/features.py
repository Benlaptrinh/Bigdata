from pyspark.sql import SparkSession
from pyspark.sql.functions import col, dayofweek, month, lag
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
INPUT_PATH = "data/processed/pm25_clean"
OUTPUT_PATH = "data/processed/features"

# -----------------------
# Read clean data
# -----------------------
df = spark.read.parquet(INPUT_PATH)

# -----------------------
# Feature engineering
# -----------------------

# Sắp xếp theo thời gian (bắt buộc cho lag)
window_spec = Window.orderBy("date")

df_features = (
    df
    .withColumn("day_of_week", dayofweek(col("date")))
    .withColumn("month", month(col("date")))
    .withColumn("lag_1", lag(col("pm25"), 1).over(window_spec))
)

# Drop dòng đầu tiên (lag bị null)
df_features = df_features.dropna()

# -----------------------
# Save features
# -----------------------
df_features.write.mode("overwrite").parquet(OUTPUT_PATH)

print("FEATURE ENGINEERING DONE.")
print("Output saved to:", OUTPUT_PATH)

spark.stop()
