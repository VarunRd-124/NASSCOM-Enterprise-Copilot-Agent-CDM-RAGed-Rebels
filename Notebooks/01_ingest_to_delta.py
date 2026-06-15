# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Ingest Docs / SOPs / Tickets / Logs to Delta Lake (hackathon schema)
# MAGIC
# MAGIC **Inputs (in `/Volumes/dev_digital_engineering_services/hackathon/hackathon/`):**
# MAGIC - `docs/kubernetes/**/*.md`, `docs/fastapi/**/*.md`, `docs/docker/**/*.md`, `docs/kafka/**/*.md`
# MAGIC - `*_sop_synthetic_pdf_dataset/*.pdf`
# MAGIC - `tickets.csv`, `logs.csv`
# MAGIC
# MAGIC **Outputs (`dev_digital_engineering_services.hackathon`):**
# MAGIC - Bronze: `bronze_sop_pdfs_raw`, `bronze_docs_md_raw`, `bronze_tickets_raw`, `bronze_logs_raw`
# MAGIC - Silver: `silver_sop_chunks`, `silver_doc_chunks`, `silver_tickets`, `silver_logs`
# MAGIC - Gold:   `gold_knowledge_chunks`

# COMMAND ----------

import time

duration_minutes = 30
interval_seconds = 60  # heartbeat every 60s

end_time = time.time() + (duration_minutes * 60)
elapsed = 0

print(f"⏳ Keeping cluster alive for {duration_minutes} minutes...")

while time.time() < end_time:
    remaining = round((end_time - time.time()) / 60, 1)
    print(f"  💓 Heartbeat — {remaining} min remaining", end="\r")
    time.sleep(interval_seconds)

print(f"\n✅ Done! Cluster was kept active for {duration_minutes} minutes.")

# COMMAND ----------

# MAGIC %pip install pypdf==5.0.1 langchain-text-splitters==0.3.2

# COMMAND ----------

CATALOG    = "dev_digital_engineering_services"
SCHEMA     = "hackathon"
RAW_VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/{SCHEMA}"   # /Volumes/dev_digital_engineering_services/hackathon/hackathon

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# Skip unreadable/corrupted files rather than failing the whole job
spark.conf.set("spark.sql.files.ignoreCorruptFiles", "true")
spark.conf.set("spark.sql.files.ignoreMissingFiles", "true")

# COMMAND ----------

# MAGIC %md ## 1. Bronze — raw bytes (PDFs + markdown) + CSV rows

# COMMAND ----------

from pyspark.sql import functions as F

# 1a. PDFs (synthetic SOPs)
pdf_df = (
    spark.read.format("binaryFile")
    .option("pathGlobFilter", "*.pdf")
    .option("recursiveFileLookup", "true")
    .load(f"{RAW_VOLUME}/")
    .withColumn("service", F.regexp_extract("path", r"/(kubernetes|docker|fastapi|kafka)_sop", 1))
    .withColumn("source_type", F.lit("sop_pdf"))
    .withColumn("ingest_ts", F.current_timestamp())
)
pdf_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.bronze_sop_pdfs_raw")
print("Bronze SOP PDFs:", spark.table(f"{CATALOG}.{SCHEMA}.bronze_sop_pdfs_raw").count())

# 1b. Markdown docs — load from root and filter by /docs/ path to avoid subdirectory resolution issues
md_df = (
    spark.read.format("binaryFile")
    .option("pathGlobFilter", "*.md")
    .option("recursiveFileLookup", "true")
    .load(f"{RAW_VOLUME}/")
    .filter(F.col("path").contains("/docs/"))
    .withColumn("service",
        F.initcap(F.regexp_extract("path", r"/docs/(kubernetes|docker|fastapi|kafka)/", 1)))
    .withColumn("source_type", F.lit("doc_md"))
    .withColumn("ingest_ts", F.current_timestamp())
)
md_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.bronze_docs_md_raw")
print("Bronze MD docs:", spark.table(f"{CATALOG}.{SCHEMA}.bronze_docs_md_raw").count())

# 1c & 1d. Tickets + Logs
# Read as binaryFile (the only format that reliably works with this cluster's volume access),
# then decode bytes in-memory and parse with pandas — no direct POSIX filesystem access needed.
import pandas as pd
from io import StringIO

def _read_csv_from_volume(root, filename):
    raw = (
        spark.read.format("binaryFile")
        .option("pathGlobFilter", filename)
        .option("recursiveFileLookup", "true")
        .load(root)
        .collect()
    )
    content = raw[0]["content"].decode("utf-8")
    return spark.createDataFrame(pd.read_csv(StringIO(content)))

# CSVs are in csv/ subfolder (binaryFile recursive scan finds subdirectory files but not root-level files)
CSV_VOLUME = f"{RAW_VOLUME}/csv"
tickets_df = _read_csv_from_volume(CSV_VOLUME, "tickets.csv").withColumn("ingest_ts", F.current_timestamp())
tickets_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.bronze_tickets_raw")
print("Bronze tickets:", spark.table(f"{CATALOG}.{SCHEMA}.bronze_tickets_raw").count())

logs_df = _read_csv_from_volume(CSV_VOLUME, "logs.csv").withColumn("ingest_ts", F.current_timestamp())
logs_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.bronze_logs_raw")
print("Bronze logs:", spark.table(f"{CATALOG}.{SCHEMA}.bronze_logs_raw").count())

# COMMAND ----------

# MAGIC %md ## 2. Silver — parse PDFs, chunk text, normalize CSVs

# COMMAND ----------

import io, re
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pyspark.sql.types import StringType, ArrayType, StructType, StructField, IntegerType

def parse_pdf(content):
    try:
        reader = PdfReader(io.BytesIO(content))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception as e:
        return f"[PARSE_ERROR] {e}"

parse_udf = F.udf(parse_pdf, StringType())

_FRONTMATTER_RE    = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
_HUGO_SHORTCODE_RE = re.compile(r"\{\{[<%].*?[>%]\}\}", re.DOTALL)
_HTML_TAG_RE       = re.compile(r"<[^>]+>")
_MD_LINK_RE        = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MULTI_BLANK_RE    = re.compile(r"\n{3,}")

def parse_md(content):
    if not content:
        return ""
    try:
        text = content.decode("utf-8", errors="ignore")
    except Exception:
        return ""
    text = _FRONTMATTER_RE.sub("", text)
    text = _HUGO_SHORTCODE_RE.sub("", text)
    text = _HTML_TAG_RE.sub("", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()

parse_md_udf = F.udf(parse_md, StringType())

splitter = RecursiveCharacterTextSplitter(
    chunk_size=900, chunk_overlap=150,
    separators=["\n## ", "\n### ", "\n\n", "\n", " "]
)

def chunk_text(text):
    if not text:
        return []
    return [(i, c) for i, c in enumerate(splitter.split_text(text))]

chunk_udf = F.udf(chunk_text, ArrayType(StructType([
    StructField("chunk_idx", IntegerType()),
    StructField("text", StringType()),
])))

# Silver: SOP chunks
sop_parsed = (
    spark.table(f"{CATALOG}.{SCHEMA}.bronze_sop_pdfs_raw")
    .withColumn("full_text", parse_udf(F.col("content")))
    .withColumn("doc_id", F.regexp_extract("path", r"([^/]+)\.pdf$", 1))
)
sop_chunks = (
    sop_parsed
    .withColumn("chunks", chunk_udf("full_text"))
    .withColumn("c", F.explode("chunks"))
    .select(
        "doc_id", "service", "path",
        F.col("c.chunk_idx").alias("chunk_idx"),
        F.col("c.text").alias("text"),
        F.lit("sop").alias("source_type"),
    )
    .withColumn("chunk_id", F.concat_ws("::", "doc_id", "chunk_idx"))
    .filter(F.length("text") > 80)
)
sop_chunks.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.silver_sop_chunks")
print("Silver SOP chunks:", spark.table(f"{CATALOG}.{SCHEMA}.silver_sop_chunks").count())

# Silver: doc chunks
docs_parsed = (
    spark.table(f"{CATALOG}.{SCHEMA}.bronze_docs_md_raw")
    .withColumn("full_text", parse_md_udf(F.col("content")))
    .withColumn("doc_id",
        F.concat_ws("::",
            F.col("service"),
            F.regexp_replace(
                F.regexp_extract("path", r"/docs/[^/]+/(.+)\.md$", 1),
                "/", "::")))
    .filter(F.length("full_text") > 200)
)
doc_chunks = (
    docs_parsed
    .withColumn("chunks", chunk_udf("full_text"))
    .withColumn("c", F.explode("chunks"))
    .select(
        "doc_id", "service", "path",
        F.col("c.chunk_idx").alias("chunk_idx"),
        F.col("c.text").alias("text"),
        F.lit("doc").alias("source_type"),
    )
    .withColumn("chunk_id", F.concat_ws("::", "doc_id", "chunk_idx"))
    .filter(F.length("text") > 80)
)
doc_chunks.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.silver_doc_chunks")
print("Silver doc chunks:", spark.table(f"{CATALOG}.{SCHEMA}.silver_doc_chunks").count())

# Silver: tickets
tickets_silver = (
    spark.table(f"{CATALOG}.{SCHEMA}.bronze_tickets_raw")
    .withColumn("text", F.concat_ws(" | ",
        F.col("issue"), F.col("description"), F.lit("Resolution:"), F.col("resolution")))
)
tickets_silver.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.silver_tickets")

# Silver: logs
logs_silver = (
    spark.table(f"{CATALOG}.{SCHEMA}.bronze_logs_raw")
    .withColumn("text", F.concat_ws(" | ", "level", "service", "message", "error_code"))
)
logs_silver.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.silver_logs")

# COMMAND ----------

# MAGIC %md ## 3. Gold — unified `gold_knowledge_chunks`

# COMMAND ----------

sop = (
    spark.table(f"{CATALOG}.{SCHEMA}.silver_sop_chunks")
    .select(
        F.col("chunk_id").alias("id"),
        F.col("text"),
        F.col("service"),
        F.lit("sop").alias("source_type"),
        F.col("doc_id").alias("source_id"),
        F.col("path").alias("source_uri"),
    )
)

docs = (
    spark.table(f"{CATALOG}.{SCHEMA}.silver_doc_chunks")
    .select(
        F.col("chunk_id").alias("id"),
        F.col("text"),
        F.col("service"),
        F.lit("doc").alias("source_type"),
        F.col("doc_id").alias("source_id"),
        F.col("path").alias("source_uri"),
    )
)

tix = (
    spark.table(f"{CATALOG}.{SCHEMA}.silver_tickets")
    .select(
        F.concat(F.lit("ticket::"), F.col("ticket_id")).alias("id"),
        F.col("text"),
        F.col("service"),
        F.lit("ticket").alias("source_type"),
        F.col("ticket_id").alias("source_id"),
        F.lit("tickets.csv").alias("source_uri"),
    )
)

lg = (
    spark.table(f"{CATALOG}.{SCHEMA}.silver_logs")
    .select(
        F.concat(F.lit("log::"), F.col("log_id")).alias("id"),
        F.col("text"),
        F.col("service"),
        F.lit("log").alias("source_type"),
        F.col("log_id").alias("source_id"),
        F.lit("logs.csv").alias("source_uri"),
    )
)

gold = sop.unionByName(docs).unionByName(tix).unionByName(lg)

(gold.write.mode("overwrite")
    .option("delta.enableChangeDataFeed", "true")
    .saveAsTable(f"{CATALOG}.{SCHEMA}.gold_knowledge_chunks"))

spark.sql(f"""
ALTER TABLE {CATALOG}.{SCHEMA}.gold_knowledge_chunks
SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

print("Gold row count:", spark.table(f"{CATALOG}.{SCHEMA}.gold_knowledge_chunks").count())
display(spark.table(f"{CATALOG}.{SCHEMA}.gold_knowledge_chunks").limit(5))