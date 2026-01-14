from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
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
MODEL_PATH = "models/pm25_lr_model"

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
# Train / Test split
# -----------------------
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# -----------------------
# Train model
# -----------------------
lr = LinearRegression(
    featuresCol="features",
    labelCol="pm25"
)

model = lr.fit(train_df)

# -----------------------
# Predict
# -----------------------
predictions = model.transform(test_df)

# -----------------------
# Evaluate
# -----------------------
evaluator = RegressionEvaluator(
    labelCol="pm25",
    predictionCol="prediction",
    metricName="rmse"
)

rmse = evaluator.evaluate(predictions)

print("TRAINING DONE.")
print("RMSE:", rmse)

# -----------------------
# Save model
# -----------------------
model.write().overwrite().save(MODEL_PATH)
print("Model saved to:", MODEL_PATH)

spark.stop()
