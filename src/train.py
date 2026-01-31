from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# -----------------------
# Spark Session
# -----------------------
spark = SparkSession.builder \
    .appName("PM25 Training Model") \
    .master("local[*]") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# -----------------------
# Paths
# -----------------------
INPUT_PATH = "data/processed/features"
LR_MODEL_PATH = "models/pm25_lr_model"
RF_MODEL_PATH = "models/pm25_rf_model"
METRICS_PATH = "data/processed/model_metrics"

# -----------------------
# Load feature data
# -----------------------
df = spark.read.parquet(INPUT_PATH)

# -----------------------
# Feature vector
# -----------------------
feature_cols = ["day_of_week", "month", "lag_1"]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

df = assembler.transform(df)

# -----------------------
# Train / Test split (Time-Series based)
# -----------------------
# IMPORTANT: For time-series data, we MUST NOT use randomSplit
# because it causes data leakage (future data leaks into training)
# Instead, we use time-based split: train = earlier dates, test = later dates
df_sorted = df.orderBy("date")

# Calculate split point (e.g., 80% of data for training)
total_count = df_sorted.count()
split_index = int(total_count * 0.8)

# Get split date from the sorted data
split_row = df_sorted.dropna().limit(1).offset(split_index).collect()
if split_row:
    split_date = split_row[0]["date"]
else:
    # Fallback: use a fixed date based on data range
    min_date = df_sorted.agg({"date": "min"}).collect()[0][0]
    split_date = min_date  # Use min_date if can't calculate

train_df = df_sorted.filter(col("date") < split_date)
test_df = df_sorted.filter(col("date") >= split_date)

print(f"Time-based split:")
print(f"  - Train: {train_df.count()} records (date < {split_date})")
print(f"  - Test: {test_df.count()} records (date >= {split_date})")

def evaluate_metrics(predictions, label_col="pm25"):
    metrics = {}
    for metric in ["rmse", "mae", "r2"]:
        evaluator = RegressionEvaluator(
            labelCol=label_col,
            predictionCol="prediction",
            metricName=metric
        )
        metrics[metric] = evaluator.evaluate(predictions)
    return metrics


# -----------------------
# Train Linear Regression
# -----------------------
lr = LinearRegression(
    featuresCol="features",
    labelCol="pm25"
)

lr_model = lr.fit(train_df)
lr_predictions = lr_model.transform(test_df)
lr_metrics = evaluate_metrics(lr_predictions)

# -----------------------
# Train Random Forest Regressor
# -----------------------
rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="pm25",
    numTrees=50,
    maxDepth=6,
    seed=42
)

rf_model = rf.fit(train_df)
rf_predictions = rf_model.transform(test_df)
rf_metrics = evaluate_metrics(rf_predictions)

print("TRAINING DONE.")
print("Linear Regression:", lr_metrics)
print("Random Forest:", rf_metrics)

# -----------------------
# Save models
# -----------------------
lr_model.write().overwrite().save(LR_MODEL_PATH)
print("Linear Regression model saved to:", LR_MODEL_PATH)

rf_model.write().overwrite().save(RF_MODEL_PATH)
print("Random Forest model saved to:", RF_MODEL_PATH)

# -----------------------
# Save metrics
# -----------------------
metrics_rows = [
    ("linear_regression", lr_metrics["rmse"], lr_metrics["mae"], lr_metrics["r2"]),
    ("random_forest", rf_metrics["rmse"], rf_metrics["mae"], rf_metrics["r2"])
]

metrics_df = spark.createDataFrame(
    metrics_rows,
    ["model", "rmse", "mae", "r2"]
)

metrics_df.write.mode("overwrite").parquet(METRICS_PATH)
print("Metrics saved to:", METRICS_PATH)

spark.stop()
