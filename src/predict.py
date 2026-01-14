from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col

spark = (
    SparkSession.builder
    .appName("pm25-predict")
    .master("local[1]")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)

df = spark.read.parquet("data/processed/features")
print("Feature data loaded")
df.show(5)

assembler = VectorAssembler(
    inputCols=["day_of_week", "month", "lag_1"],
    outputCol="features"
)

df_features = assembler.transform(df)

model = LinearRegressionModel.load("models/pm25_lr_model")
print("Model loaded successfully")

predictions = model.transform(df_features)

result = predictions.select(
    "date",
    col("pm25").alias("actual_pm25"),
    col("prediction").alias("predicted_pm25")
)

result.show(10)

result.write.mode("overwrite").parquet("data/processed/predictions")

spark.stop()
print("STEP 6 PREDICT DONE")
