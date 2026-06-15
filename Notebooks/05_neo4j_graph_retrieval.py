# Databricks notebook source
# MAGIC %md
# MAGIC # Neo4j Knowledge Graph Retrieval
# MAGIC
# MAGIC Replacement for NetworkX graph_search() used by the RAG pipeline.
# MAGIC
# MAGIC Flow:
# MAGIC 1. Extract entities from question
# MAGIC 2. Query Neo4j using Cypher
# MAGIC 3. Perform multi-hop traversal
# MAGIC 4. Convert graph results into textual context
# MAGIC 5. Feed context into LLM

# COMMAND ----------

# MAGIC %md
# MAGIC (Issue)-[:AFFECTS]->(Service)
# MAGIC
# MAGIC (Issue)-[:RESOLVED_BY]->(Solution)
# MAGIC
# MAGIC (Issue)-[:INVOLVES]->(Component)

# COMMAND ----------

# MAGIC %pip install neo4j
# MAGIC from neo4j import GraphDatabase
# MAGIC import re
# MAGIC from typing import List, Dict
# MAGIC

# COMMAND ----------

NEO4J_URI = 'neo4j+s://<your-instance-id>.databases.neo4j.io'
NEO4J_USER = '<your-neo4j-user>'
NEO4J_PASSWORD = '<your-neo4j-password>'

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)


# COMMAND ----------

COMPONENT_VOCAB = {
    'kafka':['consumer','producer','consumer group'],
    'kubernetes':['pod','deployment','service'],
    'docker':['container'],
    'fastapi':['api']
}

def extract_entities(question:str)->Dict:
    q = question.lower()
    entities = []
    for tech, comps in COMPONENT_VOCAB.items():
        if tech in q:
            entities.append(tech)
        for c in comps:
            if c in q:
                entities.append(c)
    keywords = re.findall(r'\w+', q)
    return {'entities':list(set(entities)), 'keywords':keywords}


# COMMAND ----------

def query_issue_service(tx, keyword):

    cypher = """
    MATCH (i:Issue)-[:AFFECTS]->(s:Service)

    WHERE
        toLower(i.title)
            CONTAINS toLower($keyword)

        OR

        toLower(i.description)
            CONTAINS toLower($keyword)

    RETURN i,s
    LIMIT 20
    """

    return list(tx.run(cypher, keyword=keyword))

def multi_hop_traversal(tx, entity):

    cypher = """
    MATCH p=(n)-[*1..3]-(m)

    WHERE

        (
            n.name IS NOT NULL
            AND
            toLower(n.name)
                CONTAINS toLower($entity)
        )

        OR

        (
            n.title IS NOT NULL
            AND
            toLower(n.title)
                CONTAINS toLower($entity)
        )

    RETURN p
    LIMIT 50
    """

    return list(tx.run(cypher, entity=entity))

# COMMAND ----------

# def format_graph_context(issue_results, path_results):
#     context=[]
#     for r in issue_results:
#         issue=r['i']
#         service=r['s']
#         context.append(f'Issue: {issue.get("name","")}')
#         context.append(f'Affected Service: {service.get("name","")}')

#     for r in path_results:
#         path=r['p']
#         nodes=[dict(x) for x in path.nodes]
#         context.append('Traversal Path:')
#         for n in nodes:
#             context.append(f" - {n.get('name','unknown')}")

#     return '\n'.join(context)
def get_node_text(node):

    props = dict(node)

    if "name" in props:
        return props["name"]

    if "title" in props:
        return props["title"]

    if "text" in props:
        return props["text"]

    if "description" in props:
        return props["description"]

    return str(props)


def format_graph_context(issue_results, path_results):

    context = []

    for r in issue_results:

        issue = dict(r["i"])
        service = dict(r["s"])

        context.append(
            f"Issue: {issue.get('title','')}"
        )

        context.append(
            f"Description: {issue.get('description','')}"
        )

        context.append(
            f"Severity: {issue.get('severity','')}"
        )

        context.append(
            f"Affected Service: {service.get('name','')}"
        )

        context.append("")

    for r in path_results:

        path = r["p"]

        context.append("Graph Traversal:")

        for node in path.nodes:

            node_text = get_node_text(node)

            context.append(f" - {node_text}")

        context.append("")

    return "\n".join(context)


# COMMAND ----------

#create issue search
def search_issues(tx, question):

    cypher = """
    MATCH (i:Issue)-[:AFFECTS]->(s:Service)

    WHERE
        toLower(i.title) CONTAINS toLower($question)

        OR

        toLower(i.description) CONTAINS toLower($question)

    RETURN i,s
    LIMIT 10
    """

    return list(
        tx.run(cypher, question=question)
    )

# COMMAND ----------

#create component search
def search_components(tx, question):

    cypher = """
    MATCH (i:Issue)-[:INVOLVES]->(c:Component)

    WHERE
        toLower(i.title) CONTAINS toLower($question)

        OR

        toLower(i.description) CONTAINS toLower($question)

    RETURN i,c
    LIMIT 10
    """

    return list(
        tx.run(cypher, question=question)
    )

# COMMAND ----------

#create solution search
def search_solutions(tx, question):

    cypher = """
    MATCH (i:Issue)-[:RESOLVED_BY]->(sol:Solution)

    WHERE
        toLower(i.title) CONTAINS toLower($question)

        OR

        toLower(i.description) CONTAINS toLower($question)

    RETURN i,sol
    LIMIT 10
    """

    return list(
        tx.run(cypher, question=question)
    )

# COMMAND ----------

#formatter
def format_graph_context(
    issues,
    components,
    solutions
):

    context=[]

    context.append("=== ISSUES ===")

    for r in issues:

        issue=dict(r["i"])
        service=dict(r["s"])

        context.append(
            f"""
Issue: {issue['title']}
Description: {issue['description']}
Severity: {issue['severity']}
Service: {service['name']}
"""
        )

    context.append("\n=== COMPONENTS ===")

    for r in components:

        issue=dict(r["i"])
        comp=dict(r["c"])

        context.append(
            f"""
Issue: {issue['title']}
Component: {comp['name']}
"""
        )

    context.append("\n=== SOLUTIONS ===")

    for r in solutions:

        issue=dict(r["i"])
        sol=dict(r["sol"])

        context.append(
            f"""
Issue: {issue['title']}
Solution: {sol['text']}
"""
        )

    return "\n".join(context)

# COMMAND ----------

def neo4j_graph_search(question):

    with driver.session() as session:

        issues = session.execute_read(
            search_issues,
            question
        )

        components = session.execute_read(
            search_components,
            question
        )

        solutions = session.execute_read(
            search_solutions,
            question
        )

    return format_graph_context(
        issues,
        components,
        solutions
    )

# COMMAND ----------

question = "kafka consumer lag"

context = neo4j_graph_search(question)

print(context)

# COMMAND ----------

# MAGIC %md
# MAGIC config check

# COMMAND ----------

with driver.session() as s:

    result = s.run("""
    MATCH (i:Issue)
    RETURN properties(i)
    LIMIT 10
    """)

    for r in result:
        print(r["properties(i)"])

# COMMAND ----------

with driver.session() as s:

    result = s.run("""
    MATCH (i:Issue)-[r]->(s)
    RETURN
        labels(i),
        type(r),
        labels(s),
        properties(i),
        properties(s)
    LIMIT 20
    """)

    for r in result:
        print("Issue Labels:", r["labels(i)"])
        print("Relationship:", r["type(r)"])
        print("Target Labels:", r["labels(s)"])
        print("Issue:", r["properties(i)"])
        print("Target:", r["properties(s)"])
        print("="*50)

# COMMAND ----------

with driver.session() as s:

    result = s.run("""
    MATCH (i:Issue)-[:AFFECTS]->(s:Service)

    WHERE toLower(i.title)
          CONTAINS toLower("kafka")

    RETURN i,s
    """)

    rows = list(result)

    print("Rows:", len(rows))

    for r in rows:
        print(dict(r["i"]))

# COMMAND ----------

with driver.session() as s:

    result = s.run("""
    MATCH (i:Issue)-[r]->(n)
    RETURN i,r,n
    LIMIT 20
    """)

    for row in result:
        print(row)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Integration into RAG Pipeline
# MAGIC
# MAGIC Replace:
# MAGIC `graph_search(question)`
# MAGIC
# MAGIC With:
# MAGIC `neo4j_graph_search(question)`
# MAGIC
# MAGIC Then concatenate vector search context + Neo4j graph context before the LLM call.

# COMMAND ----------

# MAGIC %md
# MAGIC User Question
# MAGIC       ↓
# MAGIC Vector Search
# MAGIC       ↓
# MAGIC Single Neo4j Subgraph Retrieval
# MAGIC       ↓
# MAGIC Context Assembly
# MAGIC       ↓
# MAGIC Llama 3.1
# MAGIC       ↓
# MAGIC Answer

# COMMAND ----------

def retrieve_issue_subgraph(tx, query_text):

    cypher = """
    MATCH (i:Issue)-[:AFFECTS]->(s:Service)

    WHERE
        toLower(i.title)
            CONTAINS toLower($query)

        OR

        toLower(i.description)
            CONTAINS toLower($query)

    OPTIONAL MATCH (i)-[:RESOLVED_BY]->(sol)

    OPTIONAL MATCH (i)-[:INVOLVES]->(c)

    

    RETURN
        i,
        s,
        collect(DISTINCT sol) as solutions,
        collect(DISTINCT c) as components
    LIMIT 20
    """

    return list(
        tx.run(
            cypher,
            parameters={"query": query_text}
        )
    )

# COMMAND ----------

def format_subgraph_context(results):

    context=[]

    for r in results:

        issue=dict(r["i"])
        service=dict(r["s"])

        context.append(
            f"Issue: {issue.get('title','')}"
        )

        context.append(
            f"Description: {issue.get('description','')}"
        )

        context.append(
            f"Severity: {issue.get('severity','')}"
        )

        context.append(
            f"Service: {service.get('name','')}"
        )

        for comp in r["components"]:

            comp=dict(comp)

            context.append(
                f"Component: {comp.get('name','')}"
            )

        for sol in r["solutions"]:

            sol=dict(sol)

            context.append(
                f"Solution: {sol.get('text','')}"
            )

        

        context.append("")

    return "\n".join(context)

# COMMAND ----------

def neo4j_graph_search(question):

    with driver.session() as session:

        results = session.execute_read(
            retrieve_issue_subgraph,
            question
        )

    return format_subgraph_context(results)

# COMMAND ----------

# ctx = neo4j_graph_search(
#     "kafka consumer lag increasing"
# )

# print(ctx)

# COMMAND ----------

# MAGIC %md
# MAGIC add a vector search
# MAGIC

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch

# COMMAND ----------

# dbutils.library.restartPython()

# COMMAND ----------

# index_description = idx.describe()

# print(index_description)

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
import requests
import json

CATALOG = "dev_digital_engineering_services"
SCHEMA = "hackathon"

VS_ENDPOINT = "hackathon_vs_endpoint"

VS_INDEX = (
    f"{CATALOG}.{SCHEMA}.gold_knowledge_chunks_idx"
)

vsc = VectorSearchClient(disable_notice=True)

idx = vsc.get_index(
    endpoint_name=VS_ENDPOINT,
    index_name=VS_INDEX
)

DB_HOST = (
    dbutils.notebook.entry_point
    .getDbutils()
    .notebook()
    .getContext()
    .apiUrl()
    .getOrElse(None)
)

DB_TOKEN = (
    dbutils.notebook.entry_point
    .getDbutils()
    .notebook()
    .getContext()
    .apiToken()
    .getOrElse(None)
)

LLM_ENDPOINT = "databricks-claude-opus-4-6"

print("Clients initialized")

# COMMAND ----------

results = idx.similarity_search(
    query_text="kafka",
    columns=["id", "text"],
    num_results=1
)

import json

print(
    json.dumps(
        results,
        indent=2
    )
)

# COMMAND ----------

def vector_search(question, k=5):

    results = idx.similarity_search(
        query_text=question,
        columns=[
            "id",
            "text"
        ],
        num_results=k
    )

    rows = results["result"]["data_array"]

    chunks = []

    for row in rows:

        doc_id = row[0]
        text = row[1]

        chunks.append(
            f"""
Document ID: {doc_id}

Content:
{text}
"""
        )

    return "\n\n".join(chunks)

# COMMAND ----------

def build_context(question):

    print("Running vector search...")

    vector_context = vector_search(
        question,
        k=5
    )

    print("Vector search complete")

    print("Running graph search...")

    graph_context = neo4j_graph_search(
        question
    )

    print("Graph search complete")

    final_context = f"""
========================
VECTOR CONTEXT
========================

{vector_context}

========================
GRAPH CONTEXT
========================

{graph_context}
"""

    return final_context

# COMMAND ----------

context = build_context(
    "why is kafka consumer lag increasing"
)

# COMMAND ----------

results = idx.similarity_search(
    query_text="kafka consumer lag",
    columns=["id","text"],
    num_results=2
)

print(results)

# COMMAND ----------

rows = results["result"]["data_array"]

print(rows)

# COMMAND ----------

question = "why is kafka consumer lag increasing"

context = build_context(question)

print(context)

# COMMAND ----------

# MAGIC %md
# MAGIC LLAMA Prompt

# COMMAND ----------

#llm call
def call_llm(question, context):

    prompt = f"""
You are an expert SRE assistant.

Answer the question using ONLY the provided context.

Question:
{question}

Context:
{context}

Provide:

1. Root Cause
2. Evidence
3. Resolution Steps
4. Related Components
"""

    response = requests.post(
        f"{DB_HOST}/serving-endpoints/{LLM_ENDPOINT}/invocations",
        headers={
            "Authorization": f"Bearer {DB_TOKEN}",
            "Content-Type": "application/json"
        },
        json={
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
    )

    return response.json()

# COMMAND ----------

#RAG query
def rag_query(question):

    print("Step 1 - Retrieving Context")

    context = build_context(question)

    print("Step 2 - Generating Answer")

    answer = call_llm(
        question,
        context
    )

    return answer

# COMMAND ----------

question = "why is kafka consumer lag increasing?"

answer = rag_query(question)

print(answer)