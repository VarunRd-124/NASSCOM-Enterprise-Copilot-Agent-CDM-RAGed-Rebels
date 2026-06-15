"""
Hybrid retrieval — Section 5 of the architecture diagram.

Pipeline:
    query
      ├── vector_search()   (Mosaic AI Vector Search, semantic)
      └── graph_search()    (Neo4j, structured Cypher)
      ↓
    fuse + dedup
      ↓
    rerank()  (HuggingFace bge-reranker-v2-m3)
      ↓
    top_k_final candidates
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

try:
    from .config import CFG
except ImportError:
    from config import CFG

log = logging.getLogger(__name__)

# Module-level singletons (lazy-loaded)
_vsc = None
_neo4j_driver = None
_reranker = None


# ---------- Data classes ----------

@dataclass
class Candidate:
    id: str
    text: str
    service: Optional[str] = None
    source_type: Optional[str] = None   # sop | ticket | log | graph
    source_id: Optional[str] = None
    source_uri: Optional[str] = None
    score: float = 0.0
    origin: str = "vector"              # vector | graph
    rerank_score: float = 0.0
    metadata: Dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


# ---------- Vector retrieval (Mosaic AI Vector Search) ----------

def _get_vsc():
    global _vsc
    if _vsc is None:
        from databricks.vector_search.client import VectorSearchClient
        _vsc = VectorSearchClient(disable_notice=True)
    return _vsc


def vector_search(query: str, service_filter: Optional[str] = None,
                  top_k: int = CFG.top_k_vector) -> List[Candidate]:
    try:
        idx = _get_vsc().get_index(endpoint_name=CFG.vs_endpoint, index_name=CFG.vs_index)
        filters = {"service": service_filter} if service_filter and service_filter != "All" else None
        res = idx.similarity_search(
            query_text=query,
            columns=["id", "text", "service", "source_type", "source_id", "source_uri"],
            num_results=top_k,
            filters=filters,
        )
        rows = res.get("result", {}).get("data_array", [])
        cols = [c["name"] for c in res.get("manifest", {}).get("columns", [])]
        out = []
        for row in rows:
            d = dict(zip(cols, row))
            out.append(Candidate(
                id=d["id"], text=d["text"], service=d.get("service"),
                source_type=d.get("source_type"), source_id=d.get("source_id"),
                source_uri=d.get("source_uri"),
                score=float(d.get("score", 0.0)), origin="vector",
            ))
        return out
    except Exception as e:
        log.exception("vector_search failed: %s", e)
        return []


# ---------- Graph retrieval (Neo4j) ----------

def _neo4j_uri_candidates(uri: str) -> List[str]:
    """Try the configured URI first, then a cert-tolerant (+ssc) variant.

    Behind a TLS-intercepting corporate proxy, full cert verification (+s) fails
    with 'Unable to retrieve routing information'; the +ssc scheme uses the same
    routing protocol but tolerates the proxy's custom CA. See notebook 05.
    """
    candidates = [uri]
    if uri.startswith("neo4j+s://"):
        candidates.append("neo4j+ssc://" + uri[len("neo4j+s://"):])
    elif uri.startswith("bolt+s://"):
        candidates.append("bolt+ssc://" + uri[len("bolt+s://"):])
    return candidates


def _get_neo4j():
    global _neo4j_driver
    if _neo4j_driver is None and CFG.neo4j_uri:
        from neo4j import GraphDatabase
        # The driver logs its own ERROR lines (e.g. "Unable to retrieve routing
        # information") on a failed scheme attempt, independent of our try/except.
        # Silence that noise while we probe candidate schemes, then restore.
        neo4j_logger = logging.getLogger("neo4j")
        prev_level = neo4j_logger.level
        neo4j_logger.setLevel(logging.CRITICAL)
        last_err = None
        try:
            for uri in _neo4j_uri_candidates(CFG.neo4j_uri):
                driver = GraphDatabase.driver(
                    uri, auth=(CFG.neo4j_user, CFG.neo4j_password),
                    connection_timeout=15)
                try:
                    driver.verify_connectivity()
                    if uri != CFG.neo4j_uri:
                        log.info("Neo4j connected via cert-tolerant scheme: %s", uri)
                    _neo4j_driver = driver
                    return _neo4j_driver
                except Exception as e:
                    last_err = e
                    driver.close()
        finally:
            neo4j_logger.setLevel(prev_level)
        log.warning(
            "Neo4j unreachable (%s). Graph retrieval disabled — verify the Aura "
            "instance is RUNNING and the URI/creds are correct.",
            last_err,
        )
    return _neo4j_driver


SERVICE_KEYWORDS = {
    "Kubernetes": ["kubernetes", "k8s", "pod", "kubelet", "kubectl", "deployment", "ingress"],
    "Docker":     ["docker", "container", "dockerfile", "image", "registry"],
    "FastAPI":    ["fastapi", "endpoint", "jwt", "cors", "pydantic", "uvicorn"],
    "Kafka":      ["kafka", "broker", "consumer", "producer", "topic", "partition", "offset"],
}


def detect_service(query: str) -> Optional[str]:
    ql = query.lower()
    for svc, words in SERVICE_KEYWORDS.items():
        if any(w in ql for w in words):
            return svc
    return None


def _query_keywords(query: str) -> List[str]:
    # crude but effective for a demo — keep tokens with length >= 4
    return [t for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", query)]


CYPHER_ISSUES = """
WITH $service AS svc, $kws AS kws
MATCH (i:Issue)
WHERE ($service IS NULL OR EXISTS { MATCH (i)-[:AFFECTS]->(:Service {name: svc}) })
WITH i, [kw IN kws WHERE toLower(i.title) CONTAINS toLower(kw)
                       OR toLower(coalesce(i.description,'')) CONTAINS toLower(kw)] AS hits
WHERE size(hits) > 0
OPTIONAL MATCH (i)-[:RESOLVED_BY]->(sol:Solution)
OPTIONAL MATCH (i)-[:INVOLVES]->(c:Component)
OPTIONAL MATCH (i)-[:AFFECTS]->(s:Service)
RETURN i.id AS id, i.title AS title, i.severity AS severity,
       coalesce(sol.text,'') AS solution,
       collect(DISTINCT c.name) AS components,
       coalesce(s.name,'') AS service,
       size(hits) AS hit_count
ORDER BY hit_count DESC, i.severity DESC
LIMIT $k
"""


def graph_search(query: str, service_filter: Optional[str] = None,
                 top_k: int = CFG.top_k_graph) -> List[Candidate]:
    driver = _get_neo4j()
    if driver is None:
        return []
    svc = service_filter if service_filter and service_filter != "All" else detect_service(query)
    kws = _query_keywords(query)
    if not kws:
        return []
    try:
        with driver.session() as s:
            rows = s.run(CYPHER_ISSUES, service=svc, kws=kws, k=top_k).data()
    except Exception as e:
        log.exception("graph_search failed: %s", e)
        return []

    out = []
    for r in rows:
        text = (
            f"[Known Issue] {r['title']} (severity={r['severity']}, service={r['service']})\n"
            f"Components involved: {', '.join(r['components']) or 'n/a'}\n"
            f"Documented resolution: {r['solution']}"
        )
        out.append(Candidate(
            id=f"graph::{r['id']}", text=text, service=r["service"],
            source_type="graph", source_id=r["id"],
            score=float(r["hit_count"]) / max(len(kws), 1),
            origin="graph",
            metadata={"components": r["components"], "severity": r["severity"]},
        ))
    return out


# ---------- Fusion + dedup ----------

def fuse(vector_hits: List[Candidate], graph_hits: List[Candidate]) -> List[Candidate]:
    """Reciprocal Rank Fusion-lite — combine, dedup by id, keep best score."""
    seen: Dict[str, Candidate] = {}
    for rank, c in enumerate(vector_hits):
        c.score = c.score + 1.0 / (rank + 60)  # RRF k=60
        seen[c.id] = c
    for rank, c in enumerate(graph_hits):
        boost = 1.0 / (rank + 60) + 0.3        # graph results get small prior boost
        if c.id in seen:
            seen[c.id].score += boost
        else:
            c.score += boost
            seen[c.id] = c
    return sorted(seen.values(), key=lambda x: x.score, reverse=True)


# ---------- Reranker (HuggingFace cross-encoder) ----------

def _get_reranker():
    """Try the reranker models in priority order until one loads."""
    global _reranker
    if _reranker is not None:
        return _reranker
    from sentence_transformers import CrossEncoder
    for name in CFG.reranker_models:
        try:
            _reranker = CrossEncoder(name, max_length=512)
            log.info("Loaded reranker: %s", name)
            return _reranker
        except Exception as e:
            log.warning("Reranker %s failed to load: %s", name, e)
    return None


def rerank(query: str, candidates: List[Candidate], top_k: int = CFG.top_k_final) -> List[Candidate]:
    if not candidates:
        return []
    model = _get_reranker()
    if model is None:
        return candidates[:top_k]
    pairs = [(query, c.text[:1500]) for c in candidates]
    scores = model.predict(pairs)
    for c, s in zip(candidates, scores):
        c.rerank_score = float(s)
    return sorted(candidates, key=lambda x: x.rerank_score, reverse=True)[:top_k]


# ---------- Single entry point ----------

def hybrid_retrieve(query: str, service_filter: Optional[str] = None) -> Dict:
    import time
    from concurrent.futures import ThreadPoolExecutor

    def _timed(fn, *a):
        s = time.time()
        r = fn(*a)
        return r, time.time() - s

    # Vector and graph search are independent — run them concurrently so total
    # retrieval time is max(vector, graph) instead of their sum.
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_v = ex.submit(_timed, vector_search, query, service_filter)
        fut_g = ex.submit(_timed, graph_search, query, service_filter)
        v, t_vec = fut_v.result()
        g, t_graph = fut_g.result()

    s = time.time()
    fused = fuse(v, g)
    top = rerank(query, fused)
    t_rerank = time.time() - s

    confidence = top[0].rerank_score if top else 0.0
    return {
        "candidates": top,
        "vector_count": len(v),
        "graph_count": len(g),
        "fused_count": len(fused),
        "confidence": confidence,
        "escalate": confidence < CFG.confidence_threshold,
        "timings": {"vector": t_vec, "graph": t_graph, "rerank": t_rerank},
    }
