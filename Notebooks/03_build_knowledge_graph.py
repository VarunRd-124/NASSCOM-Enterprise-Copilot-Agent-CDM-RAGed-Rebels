# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Entity extraction + Neo4j Knowledge Graph build
# MAGIC
# MAGIC **Architecture step:** Section 3 of the diagram — entity resolution, relationship creation, upsert to Neo4j.
# MAGIC
# MAGIC Graph schema:
# MAGIC ```
# MAGIC (:Service {name})           e.g. Kubernetes, Docker, FastAPI, Kafka
# MAGIC (:Issue {id, title, severity})
# MAGIC (:Component {name})         e.g. Pod, Broker, JWT, Volume
# MAGIC (:Solution {id, text})
# MAGIC (:SOP {doc_id, title, uri})
# MAGIC (:LogEvent {id, error_code, level})
# MAGIC
# MAGIC (Issue)-[:AFFECTS]->(Service)
# MAGIC (Issue)-[:INVOLVES]->(Component)
# MAGIC (Issue)-[:RESOLVED_BY]->(Solution)
# MAGIC (SOP)-[:DESCRIBES]->(Issue)
# MAGIC (LogEvent)-[:RELATES_TO]->(Issue)
# MAGIC ```

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

CATALOG = "dev_digital_engineering_services"
SCHEMA  = "hackathon"

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType

COMPONENT_VOCAB = {
    "Kubernetes": ["Pod", "Deployment", "Service", "Ingress", "ConfigMap", "Secret",
                   "Node", "HPA", "PersistentVolume", "DNS", "Namespace"],
    "Docker":     ["Container", "Image", "Volume", "Network", "Build", "Dockerfile",
                   "Registry", "Compose"],
    "FastAPI":    ["Endpoint", "JWT", "CORS", "Dependency", "Schema", "Middleware",
                   "Router", "BackgroundTask"],
    "Kafka":      ["Broker", "Topic", "Consumer", "Producer", "Partition", "Offset",
                   "ConsumerGroup", "Replication", "ZooKeeper"],
}


def extract_components(text, service):
    if not text or not service:
        return []
    vocab = COMPONENT_VOCAB.get(service, [])
    text_low = text.lower()
    return [c for c in vocab if c.lower() in text_low]


extract_udf = F.udf(extract_components, ArrayType(StringType()))

# Tickets
tickets = spark.table(f"{CATALOG}.{SCHEMA}.silver_tickets")
issues_df = (
    tickets
    .withColumn("service_norm", F.initcap("service"))
    .withColumn("components",
        extract_udf(F.concat_ws(" ", "issue", "description"), "service_norm"))
    .select(
        F.col("ticket_id").alias("issue_id"),
        F.col("issue").alias("title"),
        F.col("severity"),
        F.col("service_norm").alias("service"),
        F.col("description"),
        F.col("resolution"),
        F.col("components"),
    )
)

# SOPs
sops = spark.table(f"{CATALOG}.{SCHEMA}.silver_sop_chunks").select("doc_id", "service", "path").distinct()
sops_norm = (
    sops
    .withColumn("service", F.initcap("service"))
    .withColumn("title", F.regexp_replace("doc_id", "_", " "))
)

# Logs
logs = (
    spark.table(f"{CATALOG}.{SCHEMA}.silver_logs")
    .select("log_id", "service", "level", "message", "error_code", "related_ticket")
    .withColumn("service", F.initcap("service"))
)

print("✅ Spark tables loaded")

# COMMAND ----------

issues_rows = issues_df.toPandas()
sops_rows   = sops_norm.toPandas()
logs_rows   = logs.toPandas()

print(f"✅ Collected: {len(issues_rows)} issues, {len(sops_rows)} SOPs, {len(logs_rows)} logs")

# COMMAND ----------

import networkx as nx
import json

G = nx.DiGraph()

# ── Issues + Services + Components + Solutions ──
for r in issues_rows.itertuples(index=False):
    G.add_node(r.issue_id, type="Issue", title=r.title, severity=r.severity,
               description=r.description)
    G.add_node(r.service, type="Service")
    G.add_edge(r.issue_id, r.service, rel="AFFECTS")

    sol_id = f"{r.issue_id}::sol"
    G.add_node(sol_id, type="Solution", text=r.resolution)
    G.add_edge(r.issue_id, sol_id, rel="RESOLVED_BY")

    for comp in (r.components if r.components is not None else []):
        comp_id = f"{r.service}::{comp}"
        G.add_node(comp_id, type="Component", name=comp, service=r.service)
        G.add_edge(r.issue_id, comp_id, rel="INVOLVES")

# ── SOPs ──
for r in sops_rows.itertuples(index=False):
    G.add_node(r.doc_id, type="SOP", title=r.title, uri=r.path)
    G.add_node(r.service, type="Service")
    G.add_edge(r.doc_id, r.service, rel="DESCRIBES_SERVICE")

# ── Logs ──
for r in logs_rows.itertuples(index=False):
    G.add_node(r.log_id, type="LogEvent", level=r.level,
               message=r.message, error_code=r.error_code)
    G.add_node(r.service, type="Service")
    G.add_edge(r.log_id, r.service, rel="AFFECTS")
    if r.related_ticket:
        G.add_edge(r.log_id, r.related_ticket, rel="RELATES_TO")

print(f"✅ Knowledge graph built!")
print(f"   Nodes: {G.number_of_nodes()}")
print(f"   Edges: {G.number_of_edges()}")

# Quick stats by type
from collections import Counter
type_counts = Counter(data.get("type", "Unknown") for _, data in G.nodes(data=True))
for t, c in sorted(type_counts.items()):
    print(f"   {t}: {c}")

# COMMAND ----------

def get_node(node_id):
    """Get a node and its attributes"""
    if node_id in G:
        return {"id": node_id, **G.nodes[node_id]}
    return None

def get_neighbors(node_id, rel_type=None, direction="out"):
    """Get connected nodes, optionally filtered by relationship type"""
    results = []
    edges = G.out_edges(node_id, data=True) if direction == "out" else G.in_edges(node_id, data=True)
    for u, v, data in edges:
        target = v if direction == "out" else u
        if rel_type is None or data.get("rel") == rel_type:
            results.append({"id": target, "rel": data.get("rel"), **G.nodes[target]})
    return results

def find_by_type(node_type):
    """Find all nodes of a given type"""
    return [{"id": n, **data} for n, data in G.nodes(data=True) if data.get("type") == node_type]

def issues_for_service(service_name):
    """Get all issues affecting a service"""
    return [n for n in get_neighbors(service_name, rel_type="AFFECTS", direction="in")
            if n.get("type") == "Issue"]

def solution_for_issue(issue_id):
    """Get the solution for an issue"""
    sols = get_neighbors(issue_id, rel_type="RESOLVED_BY")
    return sols[0] if sols else None

# ── Test it ──
services = find_by_type("Service")
print(f"\n📊 Services: {[s['id'] for s in services]}")

for svc in services:
    issues = issues_for_service(svc["id"])
    print(f"\n🔧 {svc['id']}: {len(issues)} issues")
    for issue in issues[:3]:
        sol = solution_for_issue(issue["id"])
        print(f"   • [{issue.get('severity')}] {issue.get('title')}")
        if sol:
            print(f"     → {sol.get('text', '')[:80]}...")

# COMMAND ----------

# MAGIC %pip install pyvis

# COMMAND ----------

from pyvis.network import Network
import json

# Create interactive network
net = Network(
    height="800px",
    width="100%",
    bgcolor="#1a1a2e",
    font_color="white",
    directed=True,
    notebook=True,
    cdn_resources="in_line",  # needed for Databricks
)

# Color & size by node type
STYLE = {
    "Service":   {"color": "#e74c3c", "size": 40, "shape": "diamond"},
    "Issue":     {"color": "#f39c12", "size": 20, "shape": "dot"},
    "Solution":  {"color": "#2ecc71", "size": 15, "shape": "dot"},
    "Component": {"color": "#3498db", "size": 18, "shape": "triangle"},
    "SOP":       {"color": "#9b59b6", "size": 18, "shape": "square"},
    "LogEvent":  {"color": "#e67e22", "size": 12, "shape": "dot"},
}

# Add nodes
for node_id, data in G.nodes(data=True):
    node_type = data.get("type", "Unknown")
    style = STYLE.get(node_type, {"color": "#95a5a6", "size": 10, "shape": "dot"})

    # Build tooltip
    label = data.get("title", data.get("name", str(node_id)))
    if len(label) > 30:
        label = label[:27] + "..."

    tooltip = f"<b>{node_type}</b>: {node_id}<br>"
    for k, v in data.items():
        if k != "type":
            val = str(v)[:100]
            tooltip += f"{k}: {val}<br>"

    net.add_node(
        node_id,
        label=label,
        title=tooltip,
        color=style["color"],
        size=style["size"],
        shape=style["shape"],
        font={"size": 10},
    )

# Add edges
EDGE_COLORS = {
    "AFFECTS":           "#e74c3c",
    "RESOLVED_BY":       "#2ecc71",
    "INVOLVES":          "#3498db",
    "DESCRIBES_SERVICE": "#9b59b6",
    "RELATES_TO":        "#e67e22",
}

for u, v, data in G.edges(data=True):
    rel = data.get("rel", "")
    net.add_edge(
        u, v,
        title=rel,
        label=rel,
        color=EDGE_COLORS.get(rel, "#7f8c8d"),
        font={"size": 8, "color": "#cccccc"},
        arrows="to",
    )

# Physics settings for better layout
net.set_options(json.dumps({
    "physics": {
        "forceAtlas2Based": {
            "gravitationalConstant": -80,
            "centralGravity": 0.01,
            "springLength": 120,
            "springConstant": 0.08,
            "damping": 0.4,
        },
        "solver": "forceAtlas2Based",
        "stabilization": {"iterations": 150},
    },
    "interaction": {
        "hover": True,
        "tooltipDelay": 100,
        "zoomView": True,
    },
}))

# Save & display
output_path = "/tmp/knowledge_graph.html"
net.save_graph(output_path)

# Read and display in notebook
with open(output_path, "r") as f:
    html = f.read()

displayHTML(html)

# COMMAND ----------

CATALOG = "dev_digital_engineering_services"
SCHEMA  = "hackathon"

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType

COMPONENT_VOCAB = {
    "Kubernetes": ["Pod", "Deployment", "Service", "Ingress", "ConfigMap", "Secret",
                   "Node", "HPA", "PersistentVolume", "DNS", "Namespace"],
    "Docker":     ["Container", "Image", "Volume", "Network", "Build", "Dockerfile",
                   "Registry", "Compose"],
    "FastAPI":    ["Endpoint", "JWT", "CORS", "Dependency", "Schema", "Middleware",
                   "Router", "BackgroundTask"],
    "Kafka":      ["Broker", "Topic", "Consumer", "Producer", "Partition", "Offset",
                   "ConsumerGroup", "Replication", "ZooKeeper"],
}

def extract_components(text, service):
    if not text or not service:
        return []
    vocab = COMPONENT_VOCAB.get(service, [])
    text_low = text.lower()
    return [c for c in vocab if c.lower() in text_low]

extract_udf = F.udf(extract_components, ArrayType(StringType()))

tickets = spark.table(f"{CATALOG}.{SCHEMA}.silver_tickets")
issues_df = (
    tickets
    .withColumn("service_norm", F.initcap("service"))
    .withColumn("components",
        extract_udf(F.concat_ws(" ", "issue", "description"), "service_norm"))
    .select(
        F.col("ticket_id").alias("issue_id"),
        F.col("issue").alias("title"),
        F.col("severity"),
        F.col("service_norm").alias("service"),
        F.col("description"),
        F.col("resolution"),
        F.col("components"),
    )
)

sops = spark.table(f"{CATALOG}.{SCHEMA}.silver_sop_chunks").select("doc_id", "service", "path").distinct()
sops_norm = (
    sops
    .withColumn("service", F.initcap("service"))
    .withColumn("title", F.regexp_replace("doc_id", "_", " "))
)

logs = (
    spark.table(f"{CATALOG}.{SCHEMA}.silver_logs")
    .select("log_id", "service", "level", "message", "error_code", "related_ticket")
    .withColumn("service", F.initcap("service"))
)

print("✅ Spark tables loaded")

# COMMAND ----------

issues_rows = issues_df.toPandas()
sops_rows   = sops_norm.toPandas()
logs_rows   = logs.toPandas()

print(f"✅ Collected: {len(issues_rows)} issues, {len(sops_rows)} SOPs, {len(logs_rows)} logs")

# COMMAND ----------

import networkx as nx

G = nx.DiGraph()

for r in issues_rows.itertuples(index=False):
    G.add_node(r.issue_id, type="Issue", title=r.title, severity=r.severity, description=r.description)
    G.add_node(r.service, type="Service")
    G.add_edge(r.issue_id, r.service, rel="AFFECTS")
    sol_id = f"{r.issue_id}::sol"
    G.add_node(sol_id, type="Solution", text=r.resolution)
    G.add_edge(r.issue_id, sol_id, rel="RESOLVED_BY")
    for comp in (r.components if r.components is not None else []):
        comp_id = f"{r.service}::{comp}"
        G.add_node(comp_id, type="Component", name=comp, service=r.service)
        G.add_edge(r.issue_id, comp_id, rel="INVOLVES")

for r in sops_rows.itertuples(index=False):
    G.add_node(r.doc_id, type="SOP", title=r.title, uri=r.path)
    G.add_node(r.service, type="Service")
    G.add_edge(r.doc_id, r.service, rel="DESCRIBES_SERVICE")

for r in logs_rows.itertuples(index=False):
    G.add_node(r.log_id, type="LogEvent", level=r.level, message=r.message, error_code=r.error_code)
    G.add_node(r.service, type="Service")
    G.add_edge(r.log_id, r.service, rel="AFFECTS")
    if r.related_ticket:
        G.add_edge(r.log_id, r.related_ticket, rel="RELATES_TO")

print(f"✅ Knowledge graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
import networkx as nx
import json
import requests

# ── Vector Search Config ──
CATALOG       = "dev_digital_engineering_services"
SCHEMA        = "hackathon"
VS_ENDPOINT   = "hackathon_vs_endpoint"
VS_INDEX      = f"{CATALOG}.{SCHEMA}.gold_knowledge_chunks_idx"

# ── LLM Config (Databricks Foundation Model) ──
LLM_ENDPOINT = "databricks-claude-opus-4-6"
# ── Clients ──
vsc = VectorSearchClient(disable_notice=True)
idx = vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX)

# Databricks API for LLM calls
DB_HOST  = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
DB_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

print("✅ Clients initialized")

# COMMAND ----------

def vector_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Retrieve semantically similar chunks from Databricks Vector Search.
    Returns list of {text, score, id, ...}
    """
    try:
        results = idx.similarity_search(
            query_text=query,
            columns=["id", "text"],  # adjust based on your table columns
            num_results=top_k,
        )

        docs = []
        # results format: {"manifest": {...}, "result": {"data_array": [...]}}
        columns = [col["name"] for col in results["manifest"]["columns"]]
        for row in results["result"]["data_array"]:
            doc = dict(zip(columns, row))
            docs.append(doc)

        return docs

    except Exception as e:
        print(f"⚠️ Vector search failed: {e}")
        return []

# COMMAND ----------

def graph_search(query: str) -> dict:
    """
    Extract structured context from the NetworkX knowledge graph.
    Searches nodes by keyword matching, then traverses relationships.
    """
    query_lower = query.lower()
    context = {
        "matched_issues": [],
        "matched_services": [],
        "related_solutions": [],
        "related_components": [],
        "related_sops": [],
        "related_logs": [],
    }

    # ── 1. Find matching services ──
    services = [n for n, d in G.nodes(data=True)
                if d.get("type") == "Service" and n.lower() in query_lower]

    # If no direct service match, check all node text
    if not services:
        for n, d in G.nodes(data=True):
            searchable = " ".join(str(v) for v in d.values()).lower()
            if any(term in searchable for term in query_lower.split() if len(term) > 3):
                if d.get("type") == "Service":
                    services.append(n)

    context["matched_services"] = services

    # ── 2. Find matching issues (keyword search in title/description) ──
    for n, d in G.nodes(data=True):
        if d.get("type") != "Issue":
            continue
        searchable = f"{d.get('title', '')} {d.get('description', '')}".lower()
        if any(term in searchable for term in query_lower.split() if len(term) > 3):
            issue_info = {"id": n, **d}

            # Get solution
            for _, target, edata in G.out_edges(n, data=True):
                if edata.get("rel") == "RESOLVED_BY":
                    issue_info["solution"] = G.nodes[target].get("text", "")
                    context["related_solutions"].append({
                        "issue": n,
                        "solution": G.nodes[target].get("text", "")
                    })

            # Get components
            for _, target, edata in G.out_edges(n, data=True):
                if edata.get("rel") == "INVOLVES":
                    comp_data = G.nodes[target]
                    context["related_components"].append({
                        "name": comp_data.get("name", target),
                        "service": comp_data.get("service", "")
                    })

            context["matched_issues"].append(issue_info)

    # ── 3. Get SOPs for matched services ──
    for svc in services:
        for source, _, edata in G.in_edges(svc, data=True):
            if edata.get("rel") == "DESCRIBES_SERVICE":
                sop_data = G.nodes[source]
                if sop_data.get("type") == "SOP":
                    context["related_sops"].append({
                        "doc_id": source,
                        "title": sop_data.get("title", ""),
                        "uri": sop_data.get("uri", ""),
                    })

    # ── 4. Get logs for matched services ──
    for svc in services:
        for source, _, edata in G.in_edges(svc, data=True):
            if edata.get("rel") == "AFFECTS":
                log_data = G.nodes[source]
                if log_data.get("type") == "LogEvent":
                    context["related_logs"].append({
                        "id": source,
                        "level": log_data.get("level", ""),
                        "message": log_data.get("message", "")[:150],
                        "error_code": log_data.get("error_code", ""),
                    })

    return context
    

# COMMAND ----------

def call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
    """
    Call Databricks Foundation Model API.
    """
    try:
        response = requests.post(
            f"{DB_HOST}/serving-endpoints/{LLM_ENDPOINT}/invocations",
            headers={"Authorization": f"Bearer {DB_TOKEN}"},
            json={
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.3,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    except Exception as e:
        return f"⚠️ LLM call failed: {e}"

# COMMAND ----------

def rag_query(question: str, top_k: int = 5, verbose: bool = True) -> str:
    """
    Full RAG pipeline:
    1. Vector search for semantic matches
    2. Graph search for structured context
    3. Combine & send to LLM
    """

    if verbose:
        print(f"🔍 Query: {question}\n")

    # ── Step 1: Vector Search ──
    if verbose:
        print("📄 Step 1 — Vector Search...")
    vs_results = vector_search(question, top_k=top_k)

    vs_context = ""
    if vs_results:
        for i, doc in enumerate(vs_results, 1):
            text = doc.get("text", "")[:500]
            score = doc.get("score", "N/A")
            vs_context += f"\n[Doc {i}] (score: {score})\n{text}\n"
        if verbose:
            print(f"   Found {len(vs_results)} chunks")
    else:
        vs_context = "No relevant documents found."
        if verbose:
            print("   No results")

    # ── Step 2: Graph Search ──
    if verbose:
        print("🔗 Step 2 — Graph Search...")
    graph_ctx = graph_search(question)

    graph_context = ""

    if graph_ctx["matched_services"]:
        graph_context += f"\n**Services:** {', '.join(graph_ctx['matched_services'])}\n"

    if graph_ctx["matched_issues"]:
        graph_context += "\n**Related Issues:**\n"
        for issue in graph_ctx["matched_issues"][:5]:
            graph_context += f"- [{issue.get('severity')}] {issue.get('title')}\n"
            if issue.get("solution"):
                graph_context += f"  → Solution: {issue['solution'][:200]}\n"

    if graph_ctx["related_components"]:
        comps = list(set(c["name"] for c in graph_ctx["related_components"]))
        graph_context += f"\n**Components involved:** {', '.join(comps)}\n"

    if graph_ctx["related_sops"]:
        graph_context += "\n**Relevant SOPs:**\n"
        for sop in graph_ctx["related_sops"][:5]:
            graph_context += f"- {sop['title']} ({sop.get('uri', '')})\n"

    if graph_ctx["related_logs"]:
        graph_context += "\n**Related Log Events:**\n"
        for log in graph_ctx["related_logs"][:3]:
            graph_context += f"- [{log['level']}] {log['message']} (code: {log['error_code']})\n"

    if not graph_context:
        graph_context = "No structured graph context found."

    if verbose:
        print(f"   Issues: {len(graph_ctx['matched_issues'])}, "
              f"SOPs: {len(graph_ctx['related_sops'])}, "
              f"Logs: {len(graph_ctx['related_logs'])}")

    # ── Step 3: Combine & Call LLM ──
    if verbose:
        print("🤖 Step 3 — Generating answer...\n")

    system_prompt = """You are an Enterprise Knowledge Copilot for IT infrastructure support.
You help engineers diagnose and resolve issues across Kubernetes, Docker, FastAPI, and Kafka.

You have access to two sources of knowledge:
1. **Document chunks** — semantically retrieved from knowledge base articles, SOPs, and documentation
2. **Knowledge graph** — structured relationships between services, issues, solutions, components, SOPs, and logs

Instructions:
- Synthesize information from BOTH sources to give a comprehensive answer
- Cite specific issues, SOPs, or logs when relevant
- Provide actionable steps when troubleshooting
- If information is insufficient, say so clearly
- Use markdown formatting for readability"""

    user_prompt = f"""## Question
{question}

## Document Context (Vector Search)
{vs_context}

## Structured Context (Knowledge Graph)
{graph_context}

Please provide a comprehensive answer combining both sources."""

    answer = call_llm(system_prompt, user_prompt)

    if verbose:
        print("=" * 60)
        print("💡 ANSWER")
        print("=" * 60)

    return answer


# COMMAND ----------

# ── Example queries ──

# Query 1: Service-specific troubleshooting
answer = rag_query("How do I fix a Kubernetes pod crash loop?")
print(answer)

# COMMAND ----------

# Query 2: Cross-service question
answer = rag_query("What are common Docker container networking issues and how to resolve them?")
print(answer)

# COMMAND ----------

# Query 3: Component-level query
answer = rag_query("Kafka consumer lag is increasing, what should I do?")
print(answer)

# COMMAND ----------

# Query 4: General query
answer = rag_query("What SOPs are available for FastAPI?")
print(answer)

# COMMAND ----------

import requests

response = requests.get(
    f"{DB_HOST}/api/2.0/serving-endpoints",
    headers={"Authorization": f"Bearer {DB_TOKEN}"},
)

endpoints = response.json().get("endpoints", [])
print(f"Found {len(endpoints)} endpoints:\n")
for ep in endpoints:
    print(f"  • {ep['name']}  —  state: {ep.get('state', {}).get('ready', 'unknown')}")

# COMMAND ----------

# MAGIC %md ## 1. Lightweight entity extraction
# MAGIC We use deterministic regex + a controlled vocabulary of components per service.
# MAGIC For a production system swap in an LLM call via `ai_query()` to DBRX.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType

COMPONENT_VOCAB = {
    "Kubernetes": ["Pod", "Deployment", "Service", "Ingress", "ConfigMap", "Secret",
                   "Node", "HPA", "PersistentVolume", "DNS", "Namespace"],
    "Docker":     ["Container", "Image", "Volume", "Network", "Build", "Dockerfile",
                   "Registry", "Compose"],
    "FastAPI":    ["Endpoint", "JWT", "CORS", "Dependency", "Schema", "Middleware",
                   "Router", "BackgroundTask"],
    "Kafka":      ["Broker", "Topic", "Consumer", "Producer", "Partition", "Offset",
                   "ConsumerGroup", "Replication", "ZooKeeper"],
}


def extract_components(text, service):
    if not text or not service:
        return []
    vocab = COMPONENT_VOCAB.get(service, [])
    text_low = text.lower()
    return [c for c in vocab if c.lower() in text_low]


extract_udf = F.udf(extract_components, ArrayType(StringType()))

# Tickets
tickets = spark.table(f"{CATALOG}.{SCHEMA}.silver_tickets")
issues_df = (
    tickets
    .withColumn("service_norm", F.initcap("service"))
    .withColumn("components",
        extract_udf(F.concat_ws(" ", "issue", "description"), "service_norm"))
    .select(
        F.col("ticket_id").alias("issue_id"),
        F.col("issue").alias("title"),
        F.col("severity"),
        F.col("service_norm").alias("service"),
        F.col("description"),
        F.col("resolution"),
        F.col("components"),
    )
)
display(issues_df.limit(5))

# SOPs
sops = spark.table(f"{CATALOG}.{SCHEMA}.silver_sop_chunks").select("doc_id", "service", "path").distinct()
sops_norm = (
    sops
    .withColumn("service", F.initcap("service"))
    .withColumn("title", F.regexp_replace("doc_id", "_", " "))
)

# Logs
logs = (
    spark.table(f"{CATALOG}.{SCHEMA}.silver_logs")
    .select("log_id", "service", "level", "message", "error_code", "related_ticket")
    .withColumn("service", F.initcap("service"))
)

# COMMAND ----------

issues_rows = issues_df.toPandas()
sops_rows   = sops_norm.toPandas()
logs_rows   = logs.toPandas()

print(f"✅ Collected: {len(issues_rows)} issues, {len(sops_rows)} SOPs, {len(logs_rows)} logs")

# COMMAND ----------

# MAGIC %md ## 2. Upsert to Neo4j

# COMMAND ----------

# MAGIC %pip install neo4j
# MAGIC from neo4j import GraphDatabase
# MAGIC import re
# MAGIC from typing import List, Dict

# COMMAND ----------

from neo4j import GraphDatabase
NEO4J_URI = 'neo4j+s://<your-instance-id>.databases.neo4j.io'
NEO4J_USER = '<your-neo4j-user>'
NEO4J_PASSWORD = '<your-neo4j-password>'

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
driver.verify_connectivity()
print("✅ Connected to Neo4j")

CONSTRAINTS = [
    "CREATE CONSTRAINT service_name IF NOT EXISTS FOR (n:Service) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT issue_id IF NOT EXISTS FOR (n:Issue) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT component_key IF NOT EXISTS FOR (n:Component) REQUIRE (n.service, n.name) IS UNIQUE",
    "CREATE CONSTRAINT solution_id IF NOT EXISTS FOR (n:Solution) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT sop_id IF NOT EXISTS FOR (n:SOP) REQUIRE n.doc_id IS UNIQUE",
    "CREATE CONSTRAINT log_id IF NOT EXISTS FOR (n:LogEvent) REQUIRE n.id IS UNIQUE",
]

with driver.session() as s:
    for c in CONSTRAINTS:
        s.run(c)
print("✅ Constraints created")

# Upsert Issues + Service + Components + Solution
UPSERT_ISSUE = """
MERGE (svc:Service {name: $service})
MERGE (i:Issue {id: $issue_id})
  SET i.title = $title, i.severity = $severity, i.description = $description
MERGE (i)-[:AFFECTS]->(svc)
MERGE (sol:Solution {id: $issue_id + '::sol'})
  SET sol.text = $resolution
MERGE (i)-[:RESOLVED_BY]->(sol)
WITH i, $components AS comps, $service AS service
UNWIND comps AS comp
  MERGE (c:Component {service: service, name: comp})
  MERGE (i)-[:INVOLVES]->(c)
"""

UPSERT_SOP = """
MERGE (svc:Service {name: $service})
MERGE (sop:SOP {doc_id: $doc_id})
  SET sop.title = $title, sop.uri = $uri
MERGE (sop)-[:DESCRIBES_SERVICE]->(svc)
"""

UPSERT_LOG = """
MERGE (svc:Service {name: $service})
MERGE (l:LogEvent {id: $log_id})
  SET l.error_code = $error_code, l.level = $level, l.message = $message
MERGE (l)-[:AFFECTS]->(svc)
WITH l, $related_ticket AS tid
MATCH (i:Issue {id: tid})
MERGE (l)-[:RELATES_TO]->(i)
"""

with driver.session() as s:
    for r in issues_rows.itertuples(index=False):
        s.run(UPSERT_ISSUE, dict(
            issue_id=r.issue_id, title=r.title, severity=r.severity,
            service=r.service, description=r.description, resolution=r.resolution,
            components=list(r.components) if r.components is not None else [],
        ))
    for r in sops_rows.itertuples(index=False):
        s.run(UPSERT_SOP, dict(doc_id=r.doc_id, title=r.title, service=r.service, uri=r.path))
    for r in logs_rows.itertuples(index=False):
        s.run(UPSERT_LOG, dict(
            log_id=r.log_id, service=r.service, level=r.level,
            message=r.message, error_code=r.error_code, related_ticket=r.related_ticket,
        ))

print("✅ Knowledge graph build complete!")

# COMMAND ----------

from neo4j import GraphDatabase

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

CONSTRAINTS = [
    "CREATE CONSTRAINT service_name IF NOT EXISTS FOR (n:Service) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT issue_id IF NOT EXISTS FOR (n:Issue) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT component_key IF NOT EXISTS FOR (n:Component) REQUIRE (n.service, n.name) IS UNIQUE",
    "CREATE CONSTRAINT solution_id IF NOT EXISTS FOR (n:Solution) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT sop_id IF NOT EXISTS FOR (n:SOP) REQUIRE n.doc_id IS UNIQUE",
    "CREATE CONSTRAINT log_id IF NOT EXISTS FOR (n:LogEvent) REQUIRE n.id IS UNIQUE",
]

with driver.session() as s:
    for c in CONSTRAINTS:
        s.run(c)

# Upsert Issues + Service + Components + Solution
UPSERT_ISSUE = """
MERGE (svc:Service {name: $service})
MERGE (i:Issue {id: $issue_id})
  SET i.title = $title, i.severity = $severity, i.description = $description
MERGE (i)-[:AFFECTS]->(svc)
MERGE (sol:Solution {id: $issue_id + '::sol'})
  SET sol.text = $resolution
MERGE (i)-[:RESOLVED_BY]->(sol)
WITH i, $components AS comps, $service AS service
UNWIND comps AS comp
  MERGE (c:Component {service: service, name: comp})
  MERGE (i)-[:INVOLVES]->(c)
"""

UPSERT_SOP = """
MERGE (svc:Service {name: $service})
MERGE (sop:SOP {doc_id: $doc_id})
  SET sop.title = $title, sop.uri = $uri
MERGE (sop)-[:DESCRIBES_SERVICE]->(svc)
"""

UPSERT_LOG = """
MERGE (svc:Service {name: $service})
MERGE (l:LogEvent {id: $log_id})
  SET l.error_code = $error_code, l.level = $level, l.message = $message
MERGE (l)-[:AFFECTS]->(svc)
WITH l, $related_ticket AS tid
MATCH (i:Issue {id: tid})
MERGE (l)-[:RELATES_TO]->(i)
"""

with driver.session() as s:
    for r in issues_rows.itertuples(index=False):
        s.run(UPSERT_ISSUE, dict(
            issue_id=r.issue_id, title=r.title, severity=r.severity,
            service=r.service, description=r.description, resolution=r.resolution,
            components=list(r.components) if r.components is not None else [],
        ))
    for r in sops_rows.itertuples(index=False):
        s.run(UPSERT_SOP, dict(doc_id=r.doc_id, title=r.title, service=r.service, uri=r.path))
    for r in logs_rows.itertuples(index=False):
        s.run(UPSERT_LOG, dict(
            log_id=r.log_id, service=r.service, level=r.level,
            message=r.message, error_code=r.error_code, related_ticket=r.related_ticket,
        ))

print("Knowledge graph build complete.")

# COMMAND ----------

# Verify
with driver.session() as s:
    stats = s.run("""
        MATCH (n) RETURN labels(n)[0] AS label, count(*) AS cnt ORDER BY cnt DESC
    """).data()
    print(stats)
driver.close()