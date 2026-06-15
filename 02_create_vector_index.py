# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Create Mosaic AI Vector Search index over `gold.knowledge_chunks`
# MAGIC
# MAGIC **Architecture step:** Section 4 of the diagram — Embeddings (`BAAI/bge-large-en`) + Mosaic AI Vector Search.
# MAGIC
# MAGIC We use a **Databricks-managed embedding** endpoint serving `bge-large-en` so the index
# MAGIC stays in sync automatically when the gold table updates (Delta CDF → managed sync).

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch==0.40 mlflow==2.17.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

CATALOG       = "dev_digital_engineering_services"
SCHEMA        = "hackathon"
SOURCE_TABLE  = f"{CATALOG}.{SCHEMA}.gold_knowledge_chunks"
# → dev_digital_engineering_services.hackathon.knowledge_chunks  ✅ (3-part)
VS_ENDPOINT = "hackathon_vs_endpoint"
VS_INDEX      = f"{CATALOG}.{SCHEMA}.gold_knowledge_chunks_idx"
# → dev_digital_engineering_services.hackathon.knowledge_chunks_idx  ✅
EMBEDDING_ENDPOINT = "databricks-bge-large-en"

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient(disable_notice=True)

# 1. Endpoint
existing = [e["name"] for e in vsc.list_endpoints().get("endpoints", [])]
if VS_ENDPOINT not in existing:
    vsc.create_endpoint(name=VS_ENDPOINT, endpoint_type="STANDARD")
    print(f"Created endpoint {VS_ENDPOINT}")
else:
    print(f"Endpoint {VS_ENDPOINT} already exists")

# 2. Delta-sync index (managed embeddings — Mosaic computes vectors from `text` col)
try:
    vsc.create_delta_sync_index(
        endpoint_name=VS_ENDPOINT,
        index_name=VS_INDEX,
        source_table_name=SOURCE_TABLE,
        pipeline_type="TRIGGERED",       # change to CONTINUOUS for prod
        primary_key="id",
        embedding_source_column="text",
        embedding_model_endpoint_name=EMBEDDING_ENDPOINT,
    )
    print(f"Created index {VS_INDEX}")
except Exception as e:
    print(f"Index may already exist: {e}")

# COMMAND ----------

idx = vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX)
print(idx.describe())

# COMMAND ----------

import time

idx = vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX)

while True:
    status = idx.describe()
    state = status.get("status", {}).get("ready", False)
    msg = status.get("status", {}).get("message", "")
    print(f"⏳ Ready: {state} | {msg}")
    if state:
        print("✅ Index is ONLINE and ready to query!")
        break
    time.sleep(30)

# COMMAND ----------

# Smoke test
res = idx.similarity_search(
    query_text="kafka consumer lag is growing",
    columns=["id", "text", "service", "source_type", "source_uri"],
    num_results=5,
)
display(res)