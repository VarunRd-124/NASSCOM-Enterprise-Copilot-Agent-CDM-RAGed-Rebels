# Enterprise Knowledge Copilot — Nasscomm Hackathon Demo

A working implementation of the proposed architecture: a hybrid (vector + graph) RAG
copilot for site-reliability incidents across **Kubernetes, Docker, FastAPI, Kafka**.

```
PDFs / Tickets / Logs ─► Delta (Bronze→Silver→Gold) ─► Mosaic AI Vector Search ─┐
                                                ▲                                │
                                                └─► Neo4j Knowledge Graph ──────►├──► Rerank (HF) ──► DBRX / Claude ──► Cited answer + feedback
User Query ─────────────────────────────────────────────────────────────────────┘
```

## Repo layout

```
Nasscomm_App/
├── data/
│   ├── docs/
│   │   ├── kubernetes/                  # 1,669 .md files from kubernetes/website
│   │   ├── fastapi/                     #   153 .md files from tiangolo/fastapi
│   │   ├── docker/                      # 1,258 .md files from docker/docs
│   │   └── kafka/                       #   107 .md files from apache/kafka
│   ├── *_sop_synthetic_pdf_dataset/     #  48 synthetic SOP PDFs (given)
│   ├── tickets.csv                      #  50 historical tickets
│   └── logs.csv                         #  50 log events
├── notebooks/
│   ├── 01_ingest_to_delta.py            # PDFs → Bronze/Silver/Gold Delta
│   ├── 02_create_vector_index.py        # Mosaic AI Vector Search index
│   ├── 03_build_knowledge_graph.py      # Entity extraction → Neo4j
│   └── 04_create_feedback_table.sql     # Feedback Delta table
├── app/
│   ├── streamlit_app.py                 # Main UI
│   ├── retrieval.py                     # Vector + graph + fusion + rerank
│   ├── generation.py                    # DBRX + Claude fallback
│   ├── feedback.py                      # Thumbs up/down → Delta
│   ├── config.py                        # Centralized env config
│   └── app.yaml                         # Databricks Apps manifest
├── requirements.txt
└── README.md
```

## One-time setup on Databricks

1. **Upload data**

   The docs corpus is ~3,000 markdown files and ~60 MB — too slow to push file-by-file. Use one of:

   **Option A (recommended) — sparse-clone directly on the cluster** via a small notebook cell:
   ```python
   %sh
   set -e
   cd /Volumes/ekc/raw/files
   rm -rf _t && mkdir -p docs _t && cd _t
   for repo in \
       "kubernetes/website:content/en/docs:kubernetes" \
       "tiangolo/fastapi:docs/en/docs:fastapi" \
       "docker/docs:content:docker" \
       "apache/kafka:docs:kafka"; do
     name="${repo##*:}"; sub="${repo#*:}"; sub="${sub%:*}"; r="${repo%%:*}"
     git clone --depth=1 --filter=blob:none --sparse "https://github.com/$r.git" "$name"
     (cd "$name" && git sparse-checkout set "$sub")
     mv "$name/$sub" "../docs/$name"
     rm -rf "$name"
   done
   cd .. && rm -rf _t && ls docs/
   ```

   **Option B — tarball upload**:
   ```powershell
   tar -czf data.tar.gz -C data .
   databricks fs cp data.tar.gz dbfs:/Volumes/ekc/raw/files/data.tar.gz
   # then in a notebook:  %sh tar -xzf /Volumes/ekc/raw/files/data.tar.gz -C /Volumes/ekc/raw/files/
   ```

   Also copy the PDFs and CSVs:
   ```
   databricks fs cp -r data/tickets.csv dbfs:/Volumes/ekc/raw/files/
   databricks fs cp -r data/logs.csv    dbfs:/Volumes/ekc/raw/files/
   databricks fs cp -r data/kubernetes_sop_synthetic_pdf_dataset dbfs:/Volumes/ekc/raw/files/
   databricks fs cp -r data/docker_sop_synthetic_pdf_dataset     dbfs:/Volumes/ekc/raw/files/
   databricks fs cp -r data/fastapi_sop_synthetic_pdf_dataset    dbfs:/Volumes/ekc/raw/files/
   databricks fs cp -r data/kafka_sop_synthetic_pdf_dataset      dbfs:/Volumes/ekc/raw/files/
   ```

2. **Create the BGE embedding endpoint**
   In *Serving → Create endpoint → Foundation Models*, deploy
   `BAAI/bge-large-en` as `databricks-bge-large-en`.

3. **Create the DBRX serving endpoint**
   `databricks-dbrx-instruct` (already available as a pay-per-token endpoint in most
   workspaces). No setup needed unless you self-host.

4. **Spin up Neo4j Aura Free** (≈3 min) and store creds in a secret scope:
   ```
   databricks secrets create-scope ekc
   databricks secrets put-secret ekc neo4j_uri
   databricks secrets put-secret ekc neo4j_user
   databricks secrets put-secret ekc neo4j_password
   ```

5. **Run the notebooks in order**: `01 → 04`. Each notebook is idempotent.

## Run the Streamlit app

### Locally (for dev)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

$env:DATABRICKS_HOST = "https://<your-workspace>.cloud.databricks.com"
$env:DATABRICKS_TOKEN = "<pat>"
$env:NEO4J_URI = "neo4j+s://<id>.databases.neo4j.io"
$env:NEO4J_USER = "neo4j"
$env:NEO4J_PASSWORD = "<pw>"
$env:DATABRICKS_WAREHOUSE_ID = "<warehouse_id>"

streamlit run app/streamlit_app.py
```

### Deployed as a Databricks App
```
databricks apps create ekc-app
databricks apps deploy ekc-app --source-code-path .
```
Secret refs in `app/app.yaml` map to your `ekc` secret scope.

## What the demo shows the judges

1. **Live retrieval trace** — the sidebar shows each stage of the architecture
   firing (service detect → vector → graph → fuse → rerank → generate) so judges
   can *see* the pipeline working, not just the answer.
2. **Hybrid wins where pure vector loses** — graph hits surface the *known*
   resolution from past tickets even when the user phrases the issue differently.
3. **Citations** — every claim is cited with `[1] [2] …` linking back to the
   exact SOP / ticket / log / graph fact.
4. **Escalation path** — when reranker confidence drops below `0.35`, the agent
   refuses to hallucinate and surfaces the closest SOPs with an explicit
   *escalate to on-call* banner.
5. **Closed loop** — thumbs up/down writes to `gold.user_feedback`, ready for
   the next training cycle.

## Demo script (≈3 min)

1. Click sample query *"Why is Kafka consumer lag increasing in cluster X?"* —
   point at the live trace, show vector + graph both contributing.
2. Click *"Pod stuck in CrashLoopBackOff"* — show the answer pulling from
   ticket `T002`, log `L005`, and SOP `01_kubernetes_pod_crashloopbackoff…`.
3. Type a noisy off-topic query (`"is the printer broken"`) — show the
   escalation banner kicking in.
4. Give 👎 on the off-topic query, add a note, show the row landing in
   `ekc.gold.user_feedback` via a SQL cell.

## Key knobs (in `app/config.py`)

| Setting | Default | What it controls |
| --- | --- | --- |
| `top_k_vector` | 15 | Mosaic AI candidates pulled |
| `top_k_graph` | 10 | Cypher candidates pulled |
| `top_k_final` | 5 | Sent to the LLM after rerank |
| `confidence_threshold` | 0.35 | Below this → escalation |
| `reranker_models` | bge-reranker-v2-m3 → jina-reranker-v3 → bge-reranker-large | Tried in order |
