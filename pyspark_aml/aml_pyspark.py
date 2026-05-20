"""
AML Fraud Detection with PySpark — Local VSCode Edition
========================================================
Run:  python aml_pyspark.py
Needs: pip install pyspark  +  Java 8/11/17 installed

Pipeline stages
---------------
1. Load transactions, customers, high-risk-country reference data
2. Feature engineering  (structuring, velocity, off-hours, geo-risk)
3. Window functions     (rolling counts and sums per customer)
4. Risk scoring         (weighted rule-based AML score)
5. Alert generation     (threshold-based flagging)
6. Reporting            (console + CSV output)
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType, IntegerType, DateType,
)

# ---------------------------------------------------------------------------
# Paths  (relative to this script — works anywhere you clone the repo)
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "sample_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

TXN_FILE      = os.path.join(DATA_DIR, "transactions.csv")
CUST_FILE     = os.path.join(DATA_DIR, "customers.csv")
COUNTRY_FILE  = os.path.join(DATA_DIR, "high_risk_countries.csv")

# ---------------------------------------------------------------------------
# 1. SparkSession — local mode (no cluster needed)
# ---------------------------------------------------------------------------

def create_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("AML_Fraud_Detection")
        .master("local[*]")                              # use all local CPU cores
        .config("spark.sql.shuffle.partitions", "4")     # small value for local runs
        .config("spark.ui.showConsoleProgress", "false") # less noise in VSCode terminal
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .getOrCreate()
    )

# ---------------------------------------------------------------------------
# 2. Load data
# ---------------------------------------------------------------------------

def load_transactions(spark: SparkSession):
    schema = StructType([
        StructField("txn_id",         StringType(),  False),
        StructField("customer_id",    StringType(),  False),
        StructField("amount",         DoubleType(),  False),
        StructField("txn_date",       StringType(),  False),   # read as string, cast later
        StructField("txn_hour",       IntegerType(), False),
        StructField("src_country",    StringType(),  False),
        StructField("dst_country",    StringType(),  False),
        StructField("is_cash",        IntegerType(), False),
        StructField("counterparty_id",StringType(),  False),
        StructField("channel",        StringType(),  False),
    ])
    return (
        spark.read
        .option("header", True)
        .schema(schema)
        .csv(TXN_FILE)
        .withColumn("txn_date", F.to_date("txn_date", "yyyy-MM-dd"))
        .withColumn("txn_ts",   F.col("txn_date").cast("timestamp"))
    )


def load_customers(spark: SparkSession):
    return (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(CUST_FILE)
    )


def load_high_risk_countries(spark: SparkSession):
    return (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(COUNTRY_FILE)
    )

# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------

# Countries on FATF blacklist get the top risk level
HIGH_RISK_CODES = ["IR", "SY", "KP", "AF", "MM"]

def engineer_features(df):
    """Add AML-relevant derived columns to the transactions DataFrame."""

    # ── 3a. Cross-border flag ──────────────────────────────────────────────
    df = df.withColumn(
        "is_cross_border",
        F.when(F.col("src_country") != F.col("dst_country"), 1).otherwise(0)
    )

    # ── 3b. Structuring flag ───────────────────────────────────────────────
    # Amounts within 5 % below $10k / $50k / $100k reporting thresholds
    df = df.withColumn(
        "structuring_flag",
        F.when(
            (F.col("amount").between(9_500,  9_999)) |
            (F.col("amount").between(47_500, 49_999)) |
            (F.col("amount").between(95_000, 99_999)),
            1
        ).otherwise(0)
    )

    # ── 3c. Off-hours flag (midnight – 6 am) ──────────────────────────────
    df = df.withColumn(
        "off_hours_flag",
        F.when(F.col("txn_hour") < 6, 1).otherwise(0)
    )

    # ── 3d. Round-number flag ─────────────────────────────────────────────
    # Amount is an exact multiple of 1 000 — common in structured layering
    df = df.withColumn(
        "round_amount_flag",
        F.when((F.col("amount") % 1000) == 0, 1).otherwise(0)
    )

    # ── 3e. Destination country risk ──────────────────────────────────────
    df = df.withColumn(
        "high_risk_dst",
        F.when(F.col("dst_country").isin(HIGH_RISK_CODES), 1).otherwise(0)
    )

    # ── 3f. Log-transform amount (reduces skew for scoring) ───────────────
    df = df.withColumn("log_amount", F.log1p(F.col("amount")))

    return df

# ---------------------------------------------------------------------------
# 4. Window functions — velocity & aggregation per customer
# ---------------------------------------------------------------------------

def add_velocity_features(df):
    """
    Rolling window statistics per customer ordered by transaction timestamp.
    These capture the velocity patterns that signal smurfing and layering.
    """

    # Order by timestamp within each customer partition
    w_all = (
        Window
        .partitionBy("customer_id")
        .orderBy(F.col("txn_ts").cast("long"))
    )

    # Unbounded window — cumulative counts / totals
    w_unbounded = w_all.rowsBetween(Window.unboundedPreceding, Window.currentRow)

    # Rolling 3-row (transaction) window — recent burst detection
    w_3txn = w_all.rowsBetween(-2, 0)

    df = (
        df
        # Total transactions so far for this customer (cumulative)
        .withColumn("cumulative_txn_count",
                    F.count("txn_id").over(w_unbounded))

        # Sum of amounts in the last 3 transactions
        .withColumn("rolling_3txn_amount",
                    F.sum("amount").over(w_3txn))

        # Max single amount this customer has ever sent (cumulative)
        .withColumn("cumulative_max_amount",
                    F.max("amount").over(w_unbounded))

        # Running average amount per customer
        .withColumn("running_avg_amount",
                    F.avg("amount").over(w_unbounded))

        # How many unique counterparties has this customer dealt with so far?
        .withColumn("unique_counterparties",
                    F.approx_count_distinct("counterparty_id").over(w_unbounded))

        # Deviation of current amount from running average
        .withColumn("amount_deviation",
                    F.abs(F.col("amount") - F.col("running_avg_amount"))
                    / (F.col("running_avg_amount") + F.lit(1.0)))

        # Count of cash transactions so far for this customer
        .withColumn("cash_txn_count",
                    F.sum("is_cash").over(w_unbounded))

        # Count of off-hours transactions so far
        .withColumn("off_hours_count",
                    F.sum("off_hours_flag").over(w_unbounded))

        # Count of structuring-flagged transactions so far
        .withColumn("structuring_count",
                    F.sum("structuring_flag").over(w_unbounded))
    )

    return df

# ---------------------------------------------------------------------------
# 5. Enrich with customer profile and country risk
# ---------------------------------------------------------------------------

def enrich(txn_df, cust_df, country_df):
    """
    Left-join transactions with:
    - Customer master (account age, PEP status, risk score)
    - High-risk country reference (risk level, FATF blacklist)
    """

    # Rename country columns to avoid collision on join
    country_df = (
        country_df
        .withColumnRenamed("country_code", "dst_country")
        .withColumnRenamed("risk_level",   "country_risk_level")
        .withColumnRenamed("fatf_blacklist","fatf_flag")
        .select("dst_country", "country_risk_level", "fatf_flag")
    )

    enriched = (
        txn_df
        .join(F.broadcast(cust_df),     on="customer_id", how="left")
        .join(F.broadcast(country_df),  on="dst_country",  how="left")
        .fillna({"country_risk_level": "LOW", "fatf_flag": 0})
    )

    return enriched

# ---------------------------------------------------------------------------
# 6. AML Risk Scoring
# ---------------------------------------------------------------------------

def compute_aml_score(df):
    """
    Weighted rule-based AML score (0–100).
    Each red flag contributes a fixed weight; the sum is the alert score.

    Weights are calibrated to real-world AML typologies:
    - FATF blacklisted country is the highest single risk signal.
    - Structuring is the most common layering technique.
    - PEP involvement requires enhanced due diligence by law.
    """
    df = df.withColumn(
        "aml_score",
        (
            F.col("structuring_flag")     * 25 +   # smurfing pattern
            F.col("off_hours_flag")       * 10 +   # unusual timing
            F.col("is_cross_border")      * 10 +   # cross-border movement
            F.col("high_risk_dst")        * 15 +   # high-risk destination
            F.col("fatf_flag")            * 20 +   # FATF blacklisted country
            F.col("is_cash")              * 10 +   # cash = harder to trace
            F.col("round_amount_flag")    *  5 +   # round amounts
            F.col("previously_flagged")   * 15 +   # prior SAR / alert
            F.col("pep_flag")             * 20 +   # politically exposed person
            F.when(F.col("account_age_days") < 30,  15).otherwise(0) +  # new account
            F.when(F.col("amount_deviation") > 2.0, 10).otherwise(0) +  # outlier amount
            F.when(F.col("unique_counterparties") > 10, 10).otherwise(0) # fan-out
        ).cast(DoubleType())
    )

    # Cap score at 100
    df = df.withColumn("aml_score", F.least(F.col("aml_score"), F.lit(100.0)))

    # Map score to risk tier
    df = df.withColumn(
        "risk_tier",
        F.when(F.col("aml_score") >= 70, "CRITICAL")
         .when(F.col("aml_score") >= 50, "HIGH")
         .when(F.col("aml_score") >= 30, "MEDIUM")
         .otherwise("LOW")
    )

    return df

# ---------------------------------------------------------------------------
# 7. Alert generation
# ---------------------------------------------------------------------------

def generate_alerts(df, threshold: float = 50.0):
    """Return only the transactions that breach the alert threshold."""
    alerts = (
        df
        .filter(F.col("aml_score") >= threshold)
        .select(
            "txn_id",
            "customer_id",
            "amount",
            "txn_date",
            "txn_hour",
            "src_country",
            "dst_country",
            "channel",
            "is_cash",
            "aml_score",
            "risk_tier",
            "structuring_flag",
            "off_hours_flag",
            "is_cross_border",
            "fatf_flag",
            "pep_flag",
            "previously_flagged",
            "account_age_days",
            "customer_risk_score",
            "unique_counterparties",
            "rolling_3txn_amount",
        )
        .orderBy(F.col("aml_score").desc())
    )
    return alerts

# ---------------------------------------------------------------------------
# 8. Reporting helpers
# ---------------------------------------------------------------------------

def print_section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def report_summary(df, alerts):
    print_section("DATASET OVERVIEW")
    total    = df.count()
    flagged  = alerts.count()
    print(f"  Total transactions : {total}")
    print(f"  Alerts generated   : {flagged}  ({flagged/total*100:.1f} %)")

    print_section("RISK TIER DISTRIBUTION")
    df.groupBy("risk_tier") \
      .agg(F.count("txn_id").alias("count")) \
      .orderBy("count", ascending=False) \
      .show(truncate=False)

    print_section("TOP 10 HIGHEST-RISK TRANSACTIONS")
    alerts.select(
        "txn_id", "customer_id", "amount", "dst_country",
        "aml_score", "risk_tier", "structuring_flag", "fatf_flag", "pep_flag"
    ).show(10, truncate=False)

    print_section("CUSTOMERS WITH MULTIPLE ALERTS")
    alerts.groupBy("customer_id") \
          .agg(
              F.count("txn_id").alias("alert_count"),
              F.sum("amount").alias("total_flagged_amount"),
              F.max("aml_score").alias("max_score"),
              F.first("risk_tier").alias("risk_tier"),
          ) \
          .filter(F.col("alert_count") > 1) \
          .orderBy(F.col("max_score").desc()) \
          .show(truncate=False)

    print_section("STRUCTURING PATTERN ANALYSIS")
    df.filter(F.col("structuring_flag") == 1) \
      .groupBy("customer_id") \
      .agg(
          F.count("txn_id").alias("structuring_txn_count"),
          F.sum("amount").alias("total_structured_amount"),
          F.collect_list("amount").alias("amounts"),
      ) \
      .orderBy(F.col("structuring_txn_count").desc()) \
      .show(truncate=False)

    print_section("FATF BLACKLISTED COUNTRY FLOWS")
    df.filter(F.col("fatf_flag") == 1) \
      .select("txn_id", "customer_id", "amount", "src_country", "dst_country", "aml_score") \
      .orderBy(F.col("amount").desc()) \
      .show(truncate=False)


def save_alerts(alerts, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "aml_alerts")
    # coalesce(1) merges all partitions into a single CSV file
    alerts.coalesce(1).write.mode("overwrite").option("header", True).csv(out_path)
    print(f"\n  Alerts saved → {out_path}/")

# ---------------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------------

def main():
    print("\nStarting AML PySpark pipeline (local mode) …")
    spark = create_spark()
    spark.sparkContext.setLogLevel("ERROR")   # suppress INFO/WARN spam

    # ── Load ────────────────────────────────────────────────────────────────
    print("Loading data …")
    txn_df     = load_transactions(spark)
    cust_df    = load_customers(spark)
    country_df = load_high_risk_countries(spark)

    # ── Feature engineering ─────────────────────────────────────────────────
    print("Engineering features …")
    txn_df = engineer_features(txn_df)
    txn_df = add_velocity_features(txn_df)

    # ── Enrich ──────────────────────────────────────────────────────────────
    print("Enriching with customer and country data …")
    enriched = enrich(txn_df, cust_df, country_df)

    # ── Score ────────────────────────────────────────────────────────────────
    print("Computing AML risk scores …")
    scored = compute_aml_score(enriched)

    # Cache: the DataFrame is used multiple times in reporting
    scored.cache()

    # ── Alerts ───────────────────────────────────────────────────────────────
    print("Generating alerts (threshold = 50) …")
    alerts = generate_alerts(scored, threshold=50.0)

    # ── Report ───────────────────────────────────────────────────────────────
    report_summary(scored, alerts)
    save_alerts(alerts, OUTPUT_DIR)

    spark.stop()
    print("\nPipeline complete.\n")


if __name__ == "__main__":
    main()
