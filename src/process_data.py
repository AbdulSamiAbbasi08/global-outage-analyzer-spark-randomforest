from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, mean, stddev
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
import os

print("Starting Spark data processing...")

# Create Spark session
spark = SparkSession.builder \
    .appName("GlobalInternetOutageAnalyzer") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Load raw data
df = spark.read.csv("data/raw_outage_data.csv", header=True, inferSchema=True)

print(f"Raw records loaded: {df.count()}")
print("Schema:")
df.printSchema()

# Step 1 — Remove duplicates
df = df.dropDuplicates()
print(f"After removing duplicates: {df.count()}")

# Step 2 — Drop nulls
df = df.dropna()
print(f"After dropping nulls: {df.count()}")

# Step 3 — Remove impossible values
df = df.filter(
    (col("bgp_signal") >= 0) & (col("bgp_signal") <= 100) &
    (col("active_probing") >= 0) & (col("active_probing") <= 100) &
    (col("latency_ms") >= 0) &
    (col("traffic_drop_pct") >= 0)
)
print(f"After filtering invalid values: {df.count()}")

# Step 4 — Encode weather column (text → number)
indexer = StringIndexer(inputCol="weather", outputCol="weather_index")
df = indexer.fit(df).transform(df)

# Step 5 — Encode country column
country_indexer = StringIndexer(inputCol="country", outputCol="country_index")
df = country_indexer.fit(df).transform(df)

# Step 6 — Show summary stats
print("\nSummary statistics:")
df.select(
    "bgp_signal", "active_probing",
    "latency_ms", "traffic_drop_pct", "outage"
).describe().show()

# Step 7 — Show outage distribution
print("Outage distribution:")
df.groupBy("outage").count().show()

# Step 8 — Show outage by country
print("Top 10 countries by outage rate:")
df.groupBy("country") \
    .agg(
        mean("outage").alias("outage_rate"),
    ) \
    .orderBy("outage_rate", ascending=False) \
    .show(10)

# Step 9 — Select final features and save
final_df = df.select(
    "country",
    "iso_code",
    "timestamp",
    "hour",
    "day_of_week",
    "month",
    "weather",
    "weather_index",
    "country_index",
    "bgp_signal",
    "active_probing",
    "traffic_drop_pct",
    "latency_ms",
    "base_outage_prob",
    "outage"
)

# Save cleaned data as CSV
os.makedirs("data", exist_ok=True)
final_df.toPandas().to_csv("data/cleaned_outage_data.csv", index=False)
print("\nCleaned data saved to data/cleaned_outage_data.csv")

spark.stop()
print("Spark session stopped. Processing complete!")