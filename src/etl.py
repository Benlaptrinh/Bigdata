from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date

# 1. Khởi tạo Spark
spark = SparkSession.builder \
    .appName("PM25_ETL") \
    .master("local[*]") \
    .config("spark.driver.host", "127.0.0.1") \
    .config("spark.driver.FbindAddress", "127.0.0.1") \
    .getOrCreate()

# 2. Đường dẫn dữ liệu
RAW_PATH = "data/raw/*.json"
OUTPUT_PATH = "data/processed/pm25_clean"

# 3. Đọc JSON
df_raw = spark.read \
    .option("multiLine", True) \
    .json(RAW_PATH)

# 4. Flatten dữ liệu (results là array)
df = df_raw.selectExpr("explode(results) as r")

# 5. Chọn field cần thiết
df_selected = df.select(
    col("r.value").alias("pm25"),
    col("r.period.datetimeTo.utc").alias("datetime"),
    col("r.parameter.name").alias("parameter")
)

# 6. Lọc dữ liệu PM2.5 hợp lệ
df_clean = df_selected.filter(
    (col("parameter") == "pm25") &
    (col("pm25") > 0) &
    (col("pm25") != -999)
)

# 7. Chuẩn hóa ngày
df_final = df_clean.withColumn(
    "date",
    to_date(col("datetime"))
).select(
    "date",
    "pm25"
)

# 8. Ghi ra Parquet
df_final.write.mode("append").parquet(OUTPUT_PATH)

print("ETL DONE. Output saved to:", OUTPUT_PATH)

spark.stop()
