"""
Enterprise Knowledge Copilot — Streamlit UI with Chat + Dashboard
NASSCOM Hackathon — CDM RAGed Rebels

Architecture (backend unchanged):
  1. User query → service detection
  2. Hybrid retrieval: Mosaic AI Vector Search ⊕ Neo4j Cypher
  3. Fusion + dedup (RRF)
  4. HF cross-encoder reranking (bge-reranker-v2-m3)
  5. DBRX Instruct generation (Claude fallback)
  6. Inline citations + confidence gate
  7. Escalation path when confidence < threshold
  8. Thumbs up/down feedback → Delta table
"""
from __future__ import annotations

import os
import time
import uuid
import random
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from .retrieval import hybrid_retrieve, detect_service
    from .generation import generate_answer
    from .feedback import record_feedback
    from .config import CFG
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from retrieval import hybrid_retrieve, detect_service
    from generation import generate_answer
    from feedback import record_feedback
    from config import CFG


# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind AI — Enterprise Knowledge Copilot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown span,
section[data-testid="stSidebar"] .stMarkdown label { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stRadio label span { color: #cbd5e1 !important; }

.metric-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #e2e8f0; border-radius: 16px; padding: 24px; text-align: center;
    transition: transform 0.2s, box-shadow 0.2s; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(0,0,0,0.08); }
.metric-value { font-size: 2.2rem; font-weight: 700; line-height: 1.1; margin: 4px 0; }
.metric-label { font-size: 0.82rem; font-weight: 500; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-delta { font-size: 0.78rem; font-weight: 600; margin-top: 6px; }
.delta-up { color: #16a34a; }
.delta-down { color: #dc2626; }

.chat-msg {
    padding: 16px 20px; border-radius: 16px; margin-bottom: 12px;
    font-size: 0.95rem; line-height: 1.65; max-width: 85%;
}
.chat-user {
    background: linear-gradient(135deg, #3b82f6, #2563eb); color: white;
    margin-left: auto; border-bottom-right-radius: 4px;
}
.chat-bot { background: #f1f5f9; color: #1e293b; border-bottom-left-radius: 4px; }
.chat-source-tag {
    display: inline-block; background: #dbeafe; color: #1e40af;
    font-size: 0.72rem; font-weight: 600; padding: 3px 10px; border-radius: 20px; margin: 2px 4px 2px 0;
}
.confidence-badge { display: inline-block; padding: 4px 14px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; }
.conf-high { background: #dcfce7; color: #166534; }
.conf-med  { background: #fef9c3; color: #854d0e; }
.conf-low  { background: #fee2e2; color: #991b1b; }

.status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
.status-online  { background: #22c55e; box-shadow: 0 0 6px #22c55e88; }
.status-offline { background: #ef4444; }

.page-header { padding: 0 0 20px 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 26px; }
.page-header h1 { font-size: 2.5rem !important; font-weight: 800 !important; color: #0f172a !important; margin: 0 !important; letter-spacing: -0.02em; }
.page-header p { color: #64748b; font-size: 1rem; margin-top: 6px; }
.grad-text {
    background: linear-gradient(90deg, #2563eb 0%, #7c3aed 60%, #db2777 100%);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Reclaim the empty space left by the hidden Streamlit header */
[data-testid="stHeader"] { display: none !important; height: 0 !important; }
.block-container, [data-testid="stMainBlockContainer"] {
    padding-top: 1.5rem !important;
}
section[data-testid="stSidebar"] > div:first-child { padding-top: 0.75rem !important; }

.stage-card {
    background: white; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 16px; text-align: center; min-height: 100px;
}
.stage-card h4 { margin: 0 0 4px 0; font-size: 0.85rem; color: #0f172a; }
.stage-card p  { margin: 0; font-size: 0.75rem; color: #64748b; }
.stage-num {
    background: #3b82f6; color: white; width: 28px; height: 28px; border-radius: 50%;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 0.75rem; font-weight: 700; margin-bottom: 8px;
}

.escalation { border: 2px solid #dc2626; background: #fef2f2; padding: 14px; border-radius: 10px; margin-bottom: 8px; }

.small { font-size: 0.85rem; color: #6b7280; }
.pill { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.75rem; font-weight: 600; margin-right: 6px; }
.pill-vec   { background: #dbeafe; color: #1e40af; }
.pill-graph { background: #ede9fe; color: #5b21b6; }
.pill-sop   { background: #dcfce7; color: #166534; }
.pill-doc   { background: #e0f2fe; color: #075985; }
.pill-tkt   { background: #fef3c7; color: #92400e; }
.pill-log   { background: #fee2e2; color: #991b1b; }

#MainMenu { visibility: hidden; }
header    { visibility: hidden; }
footer    { visibility: hidden; }

.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 8px 20px; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Mock data (Dashboard / KB / KG visualizations)
# ─────────────────────────────────────────────
@st.cache_data
def _mock_queries(n: int = 200) -> pd.DataFrame:
    np.random.seed(42)
    services    = ["Kafka", "Kubernetes", "Docker", "FastAPI"]
    categories  = ["Streaming", "Deployment", "Containers", "API", "Networking", "Storage"]
    texts = [
        "Why is Kafka consumer lag increasing in cluster X?",
        "How to scale Kubernetes pods horizontally?",
        "Docker container keeps crashing with exit code 137",
        "FastAPI returns 503 after deploying new version",
        "Kafka topic partition rebalancing failing",
        "Pod eviction due to memory pressure in k8s",
        "Docker compose network bridge issues",
        "FastAPI middleware timeout configuration",
        "How to configure Kafka SSL certificates?",
        "Kubernetes ingress controller not routing traffic",
        "Docker image build fails on multi-stage",
        "FastAPI dependency injection not working",
        "Kafka consumer group rebalance storm",
        "K8s persistent volume claim stuck in pending",
        "Docker overlay network packet loss",
        "FastAPI WebSocket connection drops",
        "How to monitor Kafka broker disk usage?",
        "Kubernetes CrashLoopBackOff troubleshooting",
        "Docker container DNS resolution failure",
        "FastAPI CORS configuration for production",
    ]
    now = datetime.now()
    rows = []
    for _ in range(n):
        conf    = float(np.clip(np.random.beta(5, 2), 0.3, 1.0))
        latency = int(np.random.lognormal(7.5, 0.4))
        rows.append({
            "query_id":   str(uuid.uuid4())[:8],
            "timestamp":  now - timedelta(hours=random.randint(0, 720)),
            "query_text": random.choice(texts),
            "service":    random.choice(services),
            "category":   random.choice(categories),
            "confidence": round(conf, 3),
            "escalated":  conf < 0.70,
            "latency_ms": min(latency, 8000),
            "feedback":   random.choice(["👍", "👍", "👍", "👎", None, None]),
        })
    df = pd.DataFrame(rows)
    return df.sort_values("timestamp", ascending=False).reset_index(drop=True)


@st.cache_data
def _mock_documents() -> pd.DataFrame:
    rows = [
        {"doc_id": "D001", "title": "Kafka Consumer Lag Resolution SOP",      "source_type": "SOP",         "format": "PDF",      "pages": 24, "chunks":   48, "entities":  32, "ingested_at": "2026-05-10 09:15", "status": "Active"},
        {"doc_id": "D002", "title": "Kubernetes Pod Autoscaling Guide",        "source_type": "K8sDocs",     "format": "Markdown", "pages": 18, "chunks":   36, "entities":  21, "ingested_at": "2026-05-10 09:18", "status": "Active"},
        {"doc_id": "D003", "title": "Docker Compose Networking Reference",     "source_type": "DockerDocs",  "format": "Markdown", "pages": 12, "chunks":   25, "entities":  14, "ingested_at": "2026-05-10 09:22", "status": "Active"},
        {"doc_id": "D004", "title": "FastAPI Production Deployment Guide",     "source_type": "FastAPIDocs", "format": "Markdown", "pages": 31, "chunks":   62, "entities":  38, "ingested_at": "2026-05-10 09:25", "status": "Active"},
        {"doc_id": "D005", "title": "Incident Response Runbook v3.2",         "source_type": "SOP",         "format": "PDF",      "pages": 45, "chunks":   91, "entities":  56, "ingested_at": "2026-05-11 14:30", "status": "Active"},
        {"doc_id": "D006", "title": "Kafka Broker Configuration Reference",    "source_type": "KafkaDocs",   "format": "Markdown", "pages": 28, "chunks":   57, "entities":  34, "ingested_at": "2026-05-11 14:35", "status": "Active"},
        {"doc_id": "D007", "title": "K8s CrashLoopBackOff Troubleshooting",   "source_type": "K8sDocs",     "format": "Markdown", "pages":  8, "chunks":   16, "entities":  11, "ingested_at": "2026-05-12 08:10", "status": "Active"},
        {"doc_id": "D008", "title": "Docker Security Best Practices",          "source_type": "DockerDocs",  "format": "PDF",      "pages": 22, "chunks":   44, "entities":  27, "ingested_at": "2026-05-12 08:15", "status": "Active"},
        {"doc_id": "D009", "title": "Support Tickets Export — May 2026",       "source_type": "Tickets",     "format": "CSV",      "pages":  0, "chunks":  312, "entities":  89, "ingested_at": "2026-05-13 06:00", "status": "Active"},
        {"doc_id": "D010", "title": "Application Logs — Week 19",              "source_type": "Logs",        "format": "CSV",      "pages":  0, "chunks": 1240, "entities": 156, "ingested_at": "2026-05-13 06:05", "status": "Active"},
        {"doc_id": "D011", "title": "FastAPI Middleware & CORS Reference",     "source_type": "FastAPIDocs", "format": "Markdown", "pages": 14, "chunks":   29, "entities":  18, "ingested_at": "2026-05-13 10:40", "status": "Active"},
        {"doc_id": "D012", "title": "Kafka SSL/TLS Configuration Guide",       "source_type": "KafkaDocs",   "format": "PDF",      "pages": 16, "chunks":   33, "entities":  22, "ingested_at": "2026-05-13 10:45", "status": "Active"},
    ]
    return pd.DataFrame(rows)


@st.cache_data
def _mock_entities() -> pd.DataFrame:
    rows = [
        {"type": "Component", "name": "Kafka",                  "connections": 45, "tier": "Streaming"},
        {"type": "Component", "name": "Kubernetes",             "connections": 38, "tier": "Infrastructure"},
        {"type": "Component", "name": "Docker",                 "connections": 31, "tier": "Infrastructure"},
        {"type": "Component", "name": "FastAPI",                "connections": 27, "tier": "Application"},
        {"type": "Issue",     "name": "Consumer Lag",           "connections": 18, "tier": "Streaming"},
        {"type": "Issue",     "name": "CrashLoopBackOff",       "connections": 14, "tier": "Infrastructure"},
        {"type": "Issue",     "name": "OOMKilled",              "connections": 11, "tier": "Infrastructure"},
        {"type": "Issue",     "name": "Connection Timeout",     "connections":  9, "tier": "Application"},
        {"type": "Solution",  "name": "Scale Consumers",        "connections": 12, "tier": "Streaming"},
        {"type": "Solution",  "name": "Increase Memory Limits", "connections": 10, "tier": "Infrastructure"},
        {"type": "Solution",  "name": "Restart Deployment",     "connections":  8, "tier": "Infrastructure"},
        {"type": "Team",      "name": "DevOps",                 "connections": 22, "tier": "Organization"},
        {"type": "Team",      "name": "Data Engineering",       "connections": 19, "tier": "Organization"},
        {"type": "Team",      "name": "Backend",                "connections": 15, "tier": "Organization"},
        {"type": "SOP",       "name": "Kafka Lag SOP",          "connections": 16, "tier": "Documentation"},
        {"type": "SOP",       "name": "Incident Response",      "connections": 14, "tier": "Documentation"},
        {"type": "SOP",       "name": "Deployment Runbook",     "connections": 11, "tier": "Documentation"},
    ]
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
_DEFAULTS = {
    "chat_history":  [],
    "history":       [],
    "last_run":      None,
    "show_comment":  False,
    "user_email":    os.getenv("USER_EMAIL", ""),
    "nav":           "💬 Knowledge Chat",
    "feedback_open": False,
    "feedback_msg":  "",
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "query_df" not in st.session_state:
    st.session_state.query_df = _mock_queries()

query_df   = st.session_state.query_df
doc_df     = _mock_documents()
entity_df  = _mock_entities()


# Centered confirmation modal shown after feedback is submitted
@st.dialog("✅ Feedback received")
def _feedback_dialog():
    msg = st.session_state.get("feedback_msg") or "Thanks for your feedback!"
    st.markdown(
        f'<div style="text-align:center; padding:8px 0 4px 0;">'
        f'<div style="font-size:2.8rem; line-height:1;">🎉</div>'
        f'<p style="color:#334155; font-size:1rem; margin-top:10px;">{msg}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if st.button("Close", use_container_width=True, type="primary"):
        st.session_state.feedback_open = False
        st.rerun()


# ─────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:4px 0 6px 0;">
        <div style="font-size:3rem; line-height:1;">🧠</div>
        <h2 style="margin:8px 0 0 0; font-size:1.9rem; font-weight:800; letter-spacing:-0.02em;
                   background:linear-gradient(90deg,#60a5fa 0%,#a78bfa 60%,#f472b6 100%);
                   -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;">
            DocuMind AI
        </h2>
        <p style="font-size:0.8rem; color:#94a3b8; margin:4px 0 0 0; letter-spacing:0.03em;">Enterprise Knowledge Copilot</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Sidebar navigation styling — readable labels, hover + selected highlight
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 4px; }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label {
        padding: 9px 12px;
        border-radius: 10px;
        margin: 0;
        cursor: pointer;
        transition: background 0.15s ease;
    }
    /* Hide the default radio circle for a clean nav-menu look */
    section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
        display: none;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label p {
        color: #cbd5e1 !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background: rgba(148, 163, 184, 0.12);
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover p {
        color: #f8fafc !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(90deg, rgba(96,165,250,0.25), rgba(96,165,250,0.08));
        border-left: 3px solid #60a5fa;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["💬 Knowledge Chat", "📊 Dashboard", "📚 Knowledge Base", "🕸️ Knowledge Graph", "⚙️ System Config"],
        label_visibility="collapsed",
        key="nav",
    )

    st.markdown("---")

    st.markdown("""
    <div style="padding:8px 0;">
        <p style="font-size:0.75rem; font-weight:600; color:#94a3b8; text-transform:uppercase;
                  letter-spacing:0.05em; margin-bottom:10px;">System Status</p>
        <p style="font-size:0.82rem; color:#e2e8f0; margin:6px 0;"><span class="status-dot status-online"></span>Vector Search</p>
        <p style="font-size:0.82rem; color:#e2e8f0; margin:6px 0;"><span class="status-dot status-online"></span>Neo4j Aura</p>
        <p style="font-size:0.82rem; color:#e2e8f0; margin:6px 0;"><span class="status-dot status-online"></span>LLM Endpoint</p>
        <p style="font-size:0.82rem; color:#e2e8f0; margin:6px 0;"><span class="status-dot status-online"></span>Reranker</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; padding:4px 0;">
        <p style="font-size:0.7rem; color:#64748b;">CDM RAGed Rebels · v2.0</p>
        <p style="font-size:0.65rem; color:#475569;">NASSCOM AI Hackathon 2026</p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# PAGE: Dashboard
# ═══════════════════════════════════════════════════════
if page == "📊 Dashboard":

    st.markdown("""
    <div class="page-header">
        <h1>📊 Operations Dashboard</h1>
        <p>Real-time metrics for the DocuMind AI knowledge copilot pipeline</p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI row ──
    c1, c2, c3, c4, c5 = st.columns(5)
    total_docs     = len(doc_df)
    total_chunks   = int(doc_df["chunks"].sum())
    total_entities = int(doc_df["entities"].sum())
    avg_conf       = query_df["confidence"].mean()
    avg_lat        = query_df["latency_ms"].mean()

    for col, label, value, color, delta, delta_dir in [
        (c1, "Documents Ingested", str(total_docs),           "#3b82f6", "↑ 3 this week",       "up"),
        (c2, "Total Chunks",       f"{total_chunks:,}",       "#8b5cf6", "↑ 142 new",            "up"),
        (c3, "Graph Entities",     str(total_entities),       "#06b6d4", "↑ 28 extracted",       "up"),
        (c4, "Avg Confidence",     f"{avg_conf:.1%}",         "#16a34a", "↑ 2.1% vs last week",  "up"),
        (c5, "Avg Latency",        f"{avg_lat:,.0f}ms",       "#f59e0b", "↑ 120ms vs target",    "down"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="color:{color};">{value}</div>
                <div class="metric-delta delta-{delta_dir}">{delta}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row 1 ──
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("##### Queries Over Time")
        daily = query_df.copy()
        daily["date"] = daily["timestamp"].dt.date
        counts = daily.groupby("date").size().reset_index(name="queries").sort_values("date")
        fig = px.area(counts, x="date", y="queries", color_discrete_sequence=["#3b82f6"])
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0), xaxis_title="", yaxis_title="Query Count",
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"))
        fig.update_xaxes(showgrid=False); fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
        st.plotly_chart(fig, use_container_width=True)

    with cc2:
        st.markdown("##### Confidence Score Distribution")
        fig = px.histogram(query_df, x="confidence", nbins=25, color_discrete_sequence=["#8b5cf6"])
        fig.add_vline(x=CFG.confidence_threshold, line_dash="dash", line_color="#ef4444",
                      annotation_text=f"Threshold ({CFG.confidence_threshold})", annotation_position="top right")
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0), xaxis_title="Confidence Score", yaxis_title="Count",
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"))
        fig.update_xaxes(showgrid=False); fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
        st.plotly_chart(fig, use_container_width=True)

    # ── Charts row 2 ──
    cc3, cc4 = st.columns(2)
    with cc3:
        st.markdown("##### Queries by Service")
        svc = query_df["service"].value_counts().reset_index()
        svc.columns = ["service", "count"]
        fig = px.bar(svc, x="service", y="count", color="service",
                     color_discrete_map={"Kafka":"#f97316","Kubernetes":"#3b82f6","Docker":"#06b6d4","FastAPI":"#10b981"})
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0), showlegend=False, xaxis_title="", yaxis_title="Count",
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"))
        fig.update_xaxes(showgrid=False); fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
        st.plotly_chart(fig, use_container_width=True)

    with cc4:
        st.markdown("##### Avg Latency by Category")
        cat_lat = query_df.groupby("category")["latency_ms"].mean().reset_index()
        cat_lat.columns = ["category", "avg_latency"]
        cat_lat = cat_lat.sort_values("avg_latency")
        fig = px.bar(cat_lat, x="avg_latency", y="category", orientation="h", color_discrete_sequence=["#f59e0b"])
        fig.add_vline(x=3000, line_dash="dash", line_color="#ef4444", annotation_text="SLA (3s)")
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0), xaxis_title="Avg Latency (ms)", yaxis_title="",
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"))
        fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9"); fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Charts row 3 ──
    cc5, cc6 = st.columns(2)
    with cc5:
        st.markdown("##### Escalation Rate Over Time")
        esc = daily.groupby("date").agg(escalated=("escalated","mean")).reset_index()
        esc["escalated"] *= 100
        fig = px.line(esc, x="date", y="escalated", color_discrete_sequence=["#ef4444"])
        fig.add_hline(y=30, line_dash="dot", line_color="#94a3b8", annotation_text="Target <30%")
        fig.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0), xaxis_title="", yaxis_title="Escalation %",
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"))
        fig.update_xaxes(showgrid=False); fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
        st.plotly_chart(fig, use_container_width=True)

    with cc6:
        st.markdown("##### User Feedback Distribution")
        fb = query_df["feedback"].dropna().value_counts().reset_index()
        fb.columns = ["feedback", "count"]
        fig = px.pie(fb, values="count", names="feedback",
                     color_discrete_sequence=["#22c55e","#ef4444"], hole=0.55)
        fig.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0), font=dict(family="Inter"))
        st.plotly_chart(fig, use_container_width=True)

    # ── Recent queries table ──
    st.markdown("##### Recent Queries")
    disp = query_df.head(15)[["query_id","timestamp","query_text","service","confidence","escalated","latency_ms","feedback"]].copy()
    disp["timestamp"] = disp["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    disp.columns = ["ID","Time","Query","Service","Confidence","Escalated","Latency (ms)","Feedback"]
    st.dataframe(disp, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════
# PAGE: Knowledge Chat  (real backend)
# ═══════════════════════════════════════════════════════
elif page == "💬 Knowledge Chat":

    st.markdown("""
    <div class="page-header">
        <h1>💬 <span class="grad-text">Knowledge Copilot</span></h1>
        <p>Ask questions about your enterprise knowledge base — Kafka, Kubernetes, Docker, FastAPI</p>
    </div>
    """, unsafe_allow_html=True)

    # Show the centered confirmation modal after feedback submission
    if st.session_state.get("feedback_open"):
        _feedback_dialog()

    chat_col, info_col = st.columns([3, 1])

    # ── Left: chat thread ──
    with chat_col:
        # Capture input first (chat_input always pins to the bottom visually).
        # Append the question and rerun immediately so it shows at once — the
        # slow generation happens on the next pass, not while the box still
        # shows the welcome state.
        user_input = st.chat_input("Ask a question about your knowledge base…")
        pending    = st.session_state.pop("pending_query", None)
        active_q   = user_input or pending
        if active_q and active_q.strip() and not st.session_state.get("awaiting_answer"):
            st.session_state.chat_history.append({"role": "user", "content": active_q.strip()})
            st.session_state.awaiting_answer = active_q.strip()
            st.rerun()

        chat_box = st.container(height=500)

        with chat_box:
            if not st.session_state.chat_history:
                st.markdown("""
                <div style="text-align:center; padding:60px 20px; color:#94a3b8;">
                    <span style="font-size:3rem;">🧠</span>
                    <h3 style="color:#475569; margin:12px 0 4px 0;">Welcome to DocuMind AI</h3>
                    <p style="font-size:0.9rem; max-width:440px; margin:0 auto;">
                        Ask me anything about your enterprise knowledge base.
                        I search across SOPs, tickets, logs, and the knowledge graph
                        to give you cited, confidence-scored answers.
                    </p>
                    <div style="margin-top:24px; display:flex; gap:8px; justify-content:center; flex-wrap:wrap;">
                        <span style="background:#f1f5f9; color:#475569; padding:6px 14px; border-radius:20px; font-size:0.8rem;">
                            "Why is Kafka consumer lag increasing?"
                        </span>
                        <span style="background:#f1f5f9; color:#475569; padding:6px 14px; border-radius:20px; font-size:0.8rem;">
                            "How to fix CrashLoopBackOff?"
                        </span>
                        <span style="background:#f1f5f9; color:#475569; padding:6px 14px; border-radius:20px; font-size:0.8rem;">
                            "Docker exit code 137"
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                for msg in st.session_state.chat_history:
                    if msg["role"] == "user":
                        with st.chat_message("user"):
                            st.markdown(msg["content"])
                    else:
                        conf = msg.get("confidence", 0.0)
                        badge_cls = (
                            "conf-high" if conf >= 0.80
                            else "conf-med" if conf >= CFG.confidence_threshold
                            else "conf-low"
                        )
                        sources_html = "".join(
                            f'<span class="chat-source-tag">📄 {s["doc_name"]}</span>'
                            for s in msg.get("sources", [])
                        )
                        with st.chat_message("assistant"):
                            if msg.get("escalated"):
                                st.markdown(
                                    f'<div class="escalation">⚠️ <b>Escalation path</b> — confidence '
                                    f'<code>{conf:.2f}</code> is below threshold <code>{CFG.confidence_threshold}</code>. '
                                    f'Consult the SOPs below or escalate to on-call.</div>',
                                    unsafe_allow_html=True,
                                )
                            # Native Markdown — renders headings, bold, lists, and code blocks
                            st.markdown(msg["content"])
                            # Footer: confidence badge + latency/model (single-line HTML)
                            st.markdown(
                                f'<div style="margin-top:10px; padding-top:8px; border-top:1px solid #e2e8f0;">'
                                f'<span class="confidence-badge {badge_cls}">Confidence: {conf:.0%}</span>'
                                f'<span style="font-size:0.75rem; color:#94a3b8; margin-left:8px;">'
                                f'{msg.get("latency", 0)}ms &nbsp;·&nbsp; {msg.get("model_used", "")}</span></div>',
                                unsafe_allow_html=True,
                            )
                            if sources_html:
                                st.markdown(
                                    f'<div style="margin-top:8px;">{sources_html}</div>',
                                    unsafe_allow_html=True,
                                )

            # While generating, show a thinking bubble in the thread
            if st.session_state.get("awaiting_answer"):
                with st.chat_message("assistant"):
                    st.markdown("🔎 _Searching SOPs, tickets, logs, and the knowledge graph…_")

        # Reasoning trace for the latest answer — wide column, shown when toggled on
        if st.session_state.get("show_trace"):
            _last_bot = next(
                (m for m in reversed(st.session_state.chat_history) if m["role"] == "assistant"),
                None,
            )
            if _last_bot and _last_bot.get("reasoning"):
                with st.expander("🧠 Reasoning Trace", expanded=True):
                    for step in _last_bot["reasoning"]:
                        icon = "💭" if step.startswith("Thought") else ("⚡" if step.startswith("Action") else "👁️")
                        # Render `inline code` segments as <code> (single-line HTML)
                        parts = step.split("`")
                        body = "".join(
                            p if i % 2 == 0 else f"<code>{p}</code>"
                            for i, p in enumerate(parts)
                        )
                        st.markdown(
                            f'<div style="font-size:0.82rem; color:#475569; margin-bottom:6px; '
                            f'border-left:2px solid #e2e8f0; padding:4px 10px;">{icon} {body}</div>',
                            unsafe_allow_html=True,
                        )

        # Trace placeholder — populated during live pipeline run, cleared after
        trace_slot = st.empty()

        # Generate the answer for the pending question (set on the previous run)
        if st.session_state.get("awaiting_answer"):
            query   = st.session_state.pop("awaiting_answer")
            service = st.session_state.get("service_filter", "All")

            # ── Live pipeline trace ──
            t0 = time.time()
            stages = [
                ("Service detect", "…",                       "active"),
                ("Vector search",  "Mosaic AI Vector Search", "pending"),
                ("Graph search",   "Neo4j Cypher",            "pending"),
                ("Fuse + dedup",   "RRF",                     "pending"),
                ("Rerank",         "HF bge-reranker-v2-m3",   "pending"),
                ("Generate",       CFG.llm_endpoint,          "pending"),
            ]

            def _draw_trace(stages):
                with trace_slot.container():
                    st.markdown("##### 🔎 Retrieval pipeline")
                    cols = st.columns(len(stages))
                    for col, (title, detail, status) in zip(cols, stages):
                        icon = "✅" if status == "done" else ("⏳" if status == "active" else "·")
                        col.markdown(
                            f'<div style="border:1px solid #e5e7eb; border-radius:10px; padding:8px 10px; background:#fafafa;">'
                            f'<div style="font-weight:600; font-size:0.85rem; color:#0f172a;">{icon} {title}</div>'
                            f'<div style="font-size:0.78rem; color:#374151;">{detail}</div></div>',
                            unsafe_allow_html=True)

            _draw_trace(stages)

            detected_svc = service if service != "All" else (detect_service(query) or "auto")
            stages[0] = ("Service detect", f"`{detected_svc}`", "done")
            stages[1] = (stages[1][0], stages[1][1], "active")
            _draw_trace(stages)

            with st.spinner("Searching SOPs, tickets, logs, and the knowledge graph…"):
                result = hybrid_retrieve(query, service_filter=service)

            _tm = result.get("timings", {})
            stages[1] = ("Vector search", f"{result['vector_count']} hits · {_tm.get('vector',0):.1f}s",  "done")
            stages[2] = ("Graph search",  f"{result['graph_count']} hits · {_tm.get('graph',0):.1f}s",   "done")
            stages[3] = ("Fuse + dedup",  f"{result['fused_count']} merged", "done")
            stages[4] = ("Rerank",        f"top {len(result['candidates'])}, conf={result['confidence']:.2f} · {_tm.get('rerank',0):.1f}s", "done")
            stages[5] = (stages[5][0], stages[5][1], "active")
            _draw_trace(stages)

            if result["escalate"]:
                gen = {
                    "answer": (
                        "I'm not confident enough in the retrieved evidence to answer this directly. "
                        "See the closest matching SOPs below, or escalate to the on-call channel."
                    ),
                    "model_used": "escalation",
                    "citations": [
                        {
                            "idx": i + 1,
                            "source_type": c.source_type,
                            "source_id":   c.source_id,
                            "service":     c.service,
                            "uri":         c.source_uri,
                            "preview":     c.text[:220] + ("…" if len(c.text) > 220 else ""),
                            "rerank_score": round(c.rerank_score, 3),
                        }
                        for i, c in enumerate(result["candidates"])
                    ],
                }
                t_gen = 0.0
            else:
                _gs = time.time()
                with st.spinner("Synthesizing answer…"):
                    # Pass prior turns (everything before the current question,
                    # already appended in the capture phase) for LLM continuity
                    prior_turns = st.session_state.chat_history[:-1]
                    gen = generate_answer(query, result["candidates"], history=prior_turns)
                t_gen = time.time() - _gs

            elapsed = time.time() - t0
            stages[5] = ("Generate", f"`{gen['model_used']}` · {t_gen:.1f}s", "done")
            _draw_trace(stages)

            # Build reasoning trace from pipeline stages
            reasoning = [
                f"Thought: Detecting service context from query",
                f"Action: detect_service(query) → `{detected_svc}`",
                f"Action: vector_search(index={CFG.vs_index}, top_k={CFG.top_k_vector})",
                f"Observation: {result['vector_count']} vector matches from Mosaic AI",
                f"Action: graph_search(Neo4j Cypher, top_k={CFG.top_k_graph})",
                f"Observation: {result['graph_count']} graph matches from Neo4j",
                f"Action: fuse_rrf() + rerank(bge-reranker-v2-m3)",
                f"Observation: {result['fused_count']} fused → top {len(result['candidates'])}, confidence={result['confidence']:.2f}",
                f"Action: generate_answer(model={gen['model_used']})",
                f"Observation: Answer generated with {len(gen['citations'])} citations in {elapsed:.1f}s",
            ]

            # Normalise citations to the display format
            sources = [
                {
                    "doc_name":    c["source_id"],
                    "score":       c["rerank_score"],
                    "source_type": c.get("source_type", "doc"),
                    "service":     c.get("service", ""),
                    "uri":         c.get("uri", ""),
                    "preview":     c.get("preview", ""),
                }
                for c in gen["citations"]
            ]

            st.session_state.chat_history.append({
                "role":       "assistant",
                "content":    gen["answer"],
                "confidence": result["confidence"],
                "sources":    sources,
                "reasoning":  reasoning,
                "latency":    int(elapsed * 1000),
                "model_used": gen["model_used"],
                "escalated":  result["escalate"],
            })

            run = {"query": query, "result": result, "gen": gen,
                   "elapsed": elapsed, "service_used": detected_svc}
            st.session_state.last_run = run
            st.session_state.history.append(run)

            trace_slot.empty()
            st.rerun()

    # ── Right: controls + query details ──
    with info_col:
        st.markdown("##### ⚙️ Settings")
        show_trace = st.toggle("Show reasoning trace", value=False, key="show_trace")

        st.markdown("##### 💡 Sample Queries")
        for sample in [
            "Why is Kafka consumer lag increasing?",
            "Pod stuck in CrashLoopBackOff",
            "FastAPI 401 after deploy",
            "Docker exit code 137",
            "Kubernetes HPA not scaling",
        ]:
            if st.button(sample, use_container_width=True, key=f"sq_{hash(sample)}"):
                st.session_state.pending_query = sample
                st.rerun()

        st.markdown("---")

        # Details for most recent answer
        last_bot = next(
            (m for m in reversed(st.session_state.chat_history) if m["role"] == "assistant"),
            None,
        )

        if last_bot:
            conf = last_bot.get("confidence", 0.0)

            # Confidence gauge
            st.markdown("##### 🎯 Confidence")
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=conf * 100,
                number={"suffix": "%", "font": {"size": 26}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar":  {"color": "#3b82f6"},
                    "steps": [
                        {"range": [0,   CFG.confidence_threshold * 100], "color": "#fee2e2"},
                        {"range": [CFG.confidence_threshold * 100, 85],  "color": "#fef9c3"},
                        {"range": [85, 100],                             "color": "#dcfce7"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 2},
                        "thickness": 0.8,
                        "value": CFG.confidence_threshold * 100,
                    },
                },
            ))
            fig_g.update_layout(height=170, margin=dict(l=20, r=20, t=30, b=0))
            st.plotly_chart(fig_g, use_container_width=True)

            # Sources
            st.markdown("##### 📄 Sources")
            _PILL = {"sop": "pill-sop", "doc": "pill-doc", "ticket": "pill-tkt",
                     "log": "pill-log", "graph": "pill-graph"}
            # Official documentation per service — used when a source has no real URI
            _DOCS_LINKS = {
                "kubernetes": "https://kubernetes.io/docs/home/",
                "fastapi":    "https://fastapi.tiangolo.com/",
                "docker":     "https://docs.docker.com/",
                "kafka":      "https://kafka.apache.org/documentation/",
            }

            def _source_link(src):
                uri = (src.get("uri") or "").strip()
                if uri.startswith("http"):
                    return uri
                # Fall back to the official docs based on service / doc name
                hay = f"{src.get('service','')} {src.get('doc_name','')}".lower()
                for key, url in _DOCS_LINKS.items():
                    if key in hay:
                        return url
                return None

            for i, src in enumerate(last_bot.get("sources", [])):
                pill_cls = _PILL.get(src.get("source_type", ""), "pill-vec")
                st.markdown(f"""
                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px;
                            padding:10px 12px; margin-bottom:6px; font-size:0.82rem;">
                    <span class="pill {pill_cls}">{src.get('source_type','doc')}</span>
                    <div style="font-weight:600; color:#1e293b; margin-top:4px;">{src['doc_name']}</div>
                    <div style="color:#64748b; font-size:0.75rem;">Relevance: {src['score']:.0%}</div>
                </div>""", unsafe_allow_html=True)
                # Only the top source shows the documentation link
                link = _source_link(src) if i == 0 else None
                if link:
                    st.markdown(
                        f'<a href="{link}" target="_blank" rel="noopener noreferrer" '
                        f'style="font-size:0.8rem;">Open documentation →</a>',
                        unsafe_allow_html=True,
                    )

            # Feedback
            st.markdown("##### 💬 Feedback")
            run = st.session_state.last_run
            if run:
                fc1, fc2 = st.columns(2)
                if fc1.button("👍", use_container_width=True, help="Helpful"):
                    r = record_feedback(
                        run["query"], run["gen"]["answer"], run["gen"]["model_used"],
                        "up", None, run["gen"]["citations"], st.session_state.user_email,
                    )
                    st.session_state.feedback_msg = (
                        "Your positive feedback was recorded."
                        if r.get("persisted") else "Saved locally — thanks!"
                    )
                    st.session_state.feedback_open = True
                    st.rerun()
                if fc2.button("👎", use_container_width=True, help="Not helpful"):
                    st.session_state.show_comment = True

                if st.session_state.show_comment:
                    cmt = st.text_area("What went wrong?", key="cmt_box")
                    if st.button("Submit", type="primary"):
                        record_feedback(
                            run["query"], run["gen"]["answer"], run["gen"]["model_used"],
                            "down", cmt, run["gen"]["citations"], st.session_state.user_email,
                        )
                        st.session_state.show_comment = False
                        st.session_state.feedback_msg = "Logged — thanks for the feedback."
                        st.session_state.feedback_open = True
                        st.rerun()
        else:
            st.markdown("""
            <div style="text-align:center; padding:40px 10px; color:#94a3b8;">
                <p style="font-size:0.85rem;">Query details appear here after you ask a question.</p>
            </div>""", unsafe_allow_html=True)

        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.last_run     = None
                st.rerun()

        # Session history accordion
        if st.session_state.history:
            with st.expander(f"📋 History ({len(st.session_state.history)})"):
                for h in reversed(st.session_state.history):
                    q_short = h["query"][:50] + ("…" if len(h["query"]) > 50 else "")
                    st.markdown(
                        f"**Q:** {q_short}  \n"
                        f"*model:* `{h['gen']['model_used']}` · "
                        f"*conf:* {h['result']['confidence']:.2f}"
                    )


# ═══════════════════════════════════════════════════════
# PAGE: Knowledge Base
# ═══════════════════════════════════════════════════════
elif page == "📚 Knowledge Base":

    st.markdown("""
    <div class="page-header">
        <h1>📚 Knowledge Base</h1>
        <p>Manage ingested documents, monitor chunk counts, and track entity extraction</p>
    </div>
    """, unsafe_allow_html=True)

    kc1, kc2, kc3, kc4 = st.columns(4)
    kc1.metric("Total Documents",  len(doc_df))
    kc2.metric("Total Chunks",     f"{int(doc_df['chunks'].sum()):,}")
    kc3.metric("Total Entities",   int(doc_df["entities"].sum()))
    kc4.metric("Source Types",     doc_df["source_type"].nunique())

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("📤 Upload New Document", expanded=False):
        up1, up2 = st.columns(2)
        with up1:
            uploaded = st.file_uploader(
                "Drop a file to ingest",
                type=["pdf", "docx", "csv", "md", "json", "txt"],
            )
        with up2:
            st.selectbox("Source Type",  ["SOP", "K8sDocs", "FastAPIDocs", "DockerDocs", "KafkaDocs", "Tickets", "Logs"])
            st.selectbox("User Group",   ["engineering", "devops", "data-eng", "default"])
        if uploaded:
            if st.button("🚀 Start Ingestion", type="primary"):
                bar  = st.progress(0)
                stat = st.empty()
                for label, pct in [
                    ("Parsing document…", 15), ("Extracting text…", 30),
                    ("Chunking (512-token segments)…", 50), ("Generating BGE embeddings…", 70),
                    ("Extracting entities (NER)…", 85), ("Upserting to Vector Search & Neo4j…", 95),
                    ("Ingestion complete ✅", 100),
                ]:
                    stat.markdown(f"**{label}**")
                    bar.progress(pct)
                    time.sleep(0.5)
                st.success(f"✅ **{uploaded.name}** ingested — 24 chunks, 15 entities extracted.")

    f1, f2 = st.columns(2)
    with f1:
        sel_type = st.multiselect("Filter by Source Type", doc_df["source_type"].unique(), default=doc_df["source_type"].unique())
    with f2:
        sel_fmt  = st.multiselect("Filter by Format",      doc_df["format"].unique(),      default=doc_df["format"].unique())

    filtered = doc_df[doc_df["source_type"].isin(sel_type) & doc_df["format"].isin(sel_fmt)]
    st.dataframe(
        filtered, use_container_width=True, hide_index=True,
        column_config={
            "doc_id":      st.column_config.TextColumn("ID",             width="small"),
            "title":       st.column_config.TextColumn("Document Title", width="large"),
            "source_type": st.column_config.TextColumn("Source",         width="small"),
            "format":      st.column_config.TextColumn("Format",         width="small"),
            "pages":       st.column_config.NumberColumn("Pages",        width="small"),
            "chunks":      st.column_config.NumberColumn("Chunks",       width="small"),
            "entities":    st.column_config.NumberColumn("Entities",     width="small"),
            "ingested_at": st.column_config.TextColumn("Ingested At",    width="medium"),
            "status":      st.column_config.TextColumn("Status",         width="small"),
        },
    )

    st.markdown("<br>", unsafe_allow_html=True)
    kb1, kb2 = st.columns(2)
    with kb1:
        st.markdown("##### Chunks by Source Type")
        cbs = doc_df.groupby("source_type")["chunks"].sum().reset_index()
        fig = px.bar(cbs, x="source_type", y="chunks", color="source_type",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0), showlegend=False,
                          xaxis_title="", yaxis_title="Chunks",
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with kb2:
        st.markdown("##### Documents by Format")
        fmt = doc_df["format"].value_counts().reset_index()
        fmt.columns = ["format", "count"]
        fig = px.pie(fmt, values="count", names="format",
                     color_discrete_sequence=["#3b82f6","#8b5cf6","#f59e0b"], hole=0.5)
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════
# PAGE: Knowledge Graph
# ═══════════════════════════════════════════════════════
elif page == "🕸️ Knowledge Graph":

    st.markdown("""
    <div class="page-header">
        <h1>🕸️ Knowledge Graph Explorer</h1>
        <p>Browse entities and relationships stored in Neo4j Aura</p>
    </div>
    """, unsafe_allow_html=True)

    kg1, kg2, kg3, kg4 = st.columns(4)
    kg1.metric("Total Nodes",         len(entity_df))
    kg2.metric("Total Relationships", int(entity_df["connections"].sum()))
    kg3.metric("Node Types",          entity_df["type"].nunique())
    kg4.metric("Service Tiers",       entity_df["tier"].nunique())

    st.markdown("<br>", unsafe_allow_html=True)

    sel_types = st.multiselect("Filter Entity Types", entity_df["type"].unique(), default=entity_df["type"].unique())
    filt_ent  = entity_df[entity_df["type"].isin(sel_types)]

    kg_c1, kg_c2 = st.columns(2)
    with kg_c1:
        st.markdown("##### Entities by Type")
        tc = filt_ent["type"].value_counts().reset_index()
        tc.columns = ["type", "count"]
        fig = px.bar(tc, x="type", y="count", color="type",
                     color_discrete_sequence=["#3b82f6","#f97316","#10b981","#8b5cf6","#ef4444"])
        fig.update_layout(height=320, margin=dict(l=0,r=0,t=10,b=0), showlegend=False,
                          xaxis_title="", yaxis_title="Count",
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with kg_c2:
        st.markdown("##### Top Entities by Connections")
        top10 = filt_ent.sort_values("connections", ascending=True).tail(10)
        fig   = px.bar(top10, x="connections", y="name", orientation="h", color="type",
                       color_discrete_sequence=["#3b82f6","#f97316","#10b981","#8b5cf6","#ef4444"])
        fig.update_layout(height=320, margin=dict(l=0,r=0,t=10,b=0),
                          xaxis_title="Connections", yaxis_title="",
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### All Entities")
    st.dataframe(
        filt_ent.sort_values("connections", ascending=False),
        use_container_width=True, hide_index=True,
        column_config={
            "type":        st.column_config.TextColumn("Type",       width="small"),
            "name":        st.column_config.TextColumn("Entity Name", width="medium"),
            "connections": st.column_config.ProgressColumn("Connections", min_value=0, max_value=50),
            "tier":        st.column_config.TextColumn("Tier",       width="small"),
        },
    )

    st.markdown("##### Neo4j Relationship Schema")
    st.dataframe(pd.DataFrame([
        {"From": ":Issue",     "Relationship": "RESOLVED_BY",    "To": ":Solution",  "Example": "Consumer Lag → Scale Consumers"},
        {"From": ":Issue",     "Relationship": "AFFECTS",        "To": ":Component", "Example": "OOMKilled → Kubernetes"},
        {"From": ":Component", "Relationship": "DEPENDS_ON",     "To": ":Component", "Example": "FastAPI → Kafka"},
        {"From": ":SOP",       "Relationship": "DESCRIBES",      "To": ":Component", "Example": "Kafka Lag SOP → Kafka"},
        {"From": ":Team",      "Relationship": "OWNS",           "To": ":Component", "Example": "DevOps → Kubernetes"},
        {"From": ":Issue",     "Relationship": "DOCUMENTED_IN",  "To": ":SOP",       "Example": "Consumer Lag → Kafka Lag SOP"},
        {"From": ":Component", "Relationship": "EMITS",          "To": ":Issue",     "Example": "Docker → OOMKilled"},
    ]), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════
# PAGE: System Config
# ═══════════════════════════════════════════════════════
elif page == "⚙️ System Config":

    st.markdown("""
    <div class="page-header">
        <h1>⚙️ System Configuration</h1>
        <p>Pipeline parameters, model endpoints, and system settings</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["🔧 Pipeline Config", "🤖 Model Endpoints", "🏗️ Architecture", "📋 API Contract"])

    with tab1:
        st.markdown("##### Ingestion & Retrieval Parameters")
        cfg1, cfg2 = st.columns(2)
        with cfg1:
            st.markdown("**Chunking**")
            st.slider("Chunk Size (tokens)",    128, 1024, 512, 64)
            st.slider("Chunk Overlap (tokens)",   0,  256, 128, 32)
            st.markdown("---")
            st.markdown("**Retrieval**")
            st.slider("Vector Top-K",  5, 50, CFG.top_k_vector, 5)
            st.slider("Final Top-K",   1, 10, CFG.top_k_final,  1)
        with cfg2:
            st.markdown("**Confidence**")
            conf_thresh = st.slider("Confidence Threshold", 0.0, 1.0, CFG.confidence_threshold, 0.05)
            rrw = st.slider("Reranker Weight", 0.0, 1.0, 0.60, 0.05)
            st.markdown(f"Vector weight: **{1.0 - rrw:.2f}** &nbsp;|&nbsp; Formula: `{rrw:.2f}×reranker + {1.0-rrw:.2f}×cosine`")
            st.markdown("---")
            st.markdown("**Agent**")
            st.slider("Max ReAct Steps", 1, 10, 5)
            st.selectbox("LLM Router", [CFG.llm_endpoint, CFG.claude_model, "Mistral Large"])
        if st.button("💾 Save Configuration", type="primary"):
            st.success("Configuration saved to `config.py`")

    with tab2:
        st.markdown("##### Databricks Model Serving Endpoints")
        st.dataframe(pd.DataFrame([
            {"Endpoint": "bge-large",      "Model": "BAAI/bge-large-en-v1.5",    "Type": "Embedding",       "Status": "🟢 Running", "Latency": "45ms",   "Throughput": "120 req/s"},
            {"Endpoint": "bge-reranker",   "Model": CFG.reranker_models[0],      "Type": "Reranker",        "Status": "🟢 Running", "Latency": "180ms",  "Throughput": "40 req/s"},
            {"Endpoint": "llm-endpoint",   "Model": f"{CFG.llm_endpoint} / {CFG.claude_model}", "Type": "Generation", "Status": "🟢 Running", "Latency": "1200ms", "Throughput": "15 req/s"},
            {"Endpoint": "bm25-index",     "Model": "rank_bm25 (in-memory)",     "Type": "Sparse Retrieval","Status": "🟢 Running", "Latency": "12ms",   "Throughput": "500 req/s"},
            {"Endpoint": "ner-pipeline",   "Model": "spaCy en_core_web_lg",      "Type": "Entity Extraction","Status": "🟢 Running", "Latency": "85ms",   "Throughput": "80 req/s"},
        ]), use_container_width=True, hide_index=True)

        st.markdown("##### External Services")
        st.dataframe(pd.DataFrame([
            {"Service": "Neo4j Aura",              "Endpoint": CFG.neo4j_uri or "not configured",           "Status": "🟢 Connected" if CFG.neo4j_uri else "🔴 Not configured", "Region": "Azure East US"},
            {"Service": "Mosaic AI Vector Search", "Endpoint": CFG.vs_index,                                "Status": "🟢 Synced",      "Region": "Azure East US"},
            {"Service": "Unity Catalog",           "Endpoint": CFG.vs_index.rsplit(".", 1)[0],              "Status": "🟢 Active",      "Region": "Azure East US"},
            {"Service": "MLflow Experiment",       "Endpoint": "rag_traces",                                "Status": "🟢 Tracking",    "Region": "Azure East US"},
        ]), use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("##### Eight-Stage Pipeline Architecture")
        arch_stages = [
            ("1", "Data Sources",    "PDF, CSV, Logs, Markdown"),
            ("2", "Ingestion",       "Bronze → Silver → Gold (Delta)"),
            ("3", "Knowledge Graph", "Neo4j — entities & relations"),
            ("4", "Vector Store",    "Mosaic AI Vector Search (BGE)"),
            ("5", "Hybrid Retrieval","Vector + BM25 + Graph → RRF"),
            ("6", "Reranking",       "BGE cross-encoder reranker"),
            ("7", "Generation",      "ReAct Agent → Claude/DBRX"),
            ("8", "Answer + FB",     "Cited answer + confidence gate"),
        ]
        r1 = st.columns(4)
        r2 = st.columns(4)
        for i, (num, name, desc) in enumerate(arch_stages):
            c = r1[i] if i < 4 else r2[i - 4]
            with c:
                st.markdown(f"""
                <div class="stage-card">
                    <div class="stage-num">{num}</div>
                    <h4>{name}</h4>
                    <p>{desc}</p>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### Workspace Structure")
        st.code(f"""
Databricks Workspace (Azure)
├── Unity Catalog
│   ├── Volumes/              ← raw documents (PDF / DOCX / CSV / logs)
│   └── Schemas/
│       ├── {CFG.vs_index}
│       └── {CFG.feedback_table}
├── Mosaic AI Vector Search
│   └── {CFG.vs_endpoint}    ← Delta Sync Index (BGE vectors)
├── Model Serving Endpoints
│   ├── bge-large             ← embedding endpoint
│   ├── bge-reranker          ← reranker endpoint
│   └── {CFG.llm_endpoint}   ← LLM router
└── MLflow Experiment
    └── rag_traces            ← per-query traces
        """, language="text")

    with tab4:
        st.markdown("##### POST `/invocations` — Request")
        st.code("""{
    "query": "What causes Kafka consumer lag?",
    "user_group": "engineering",
    "top_k": 5,
    "min_confidence": 0.70
}""", language="json")

        st.markdown("##### Response Schema")
        st.code("""{
    "answer": "Kafka consumer lag is caused by...",
    "sources": [{
        "doc_name": "kafka-lag-sop.pdf",
        "page": 8,
        "section": "Consumer Configuration",
        "chunk_text": "...relevant passage...",
        "relevance_score": 0.94
    }],
    "confidence_score": 0.92,
    "reasoning_steps": [
        "Thought: I need to search for Kafka consumer lag",
        "Action: vector_search(kafka consumer lag causes)",
        "Observation: Found 5 relevant chunks"
    ],
    "escalated": false,
    "latency_ms": 2840
}""", language="json")

        st.markdown("##### Response Fields")
        st.dataframe(pd.DataFrame([
            {"Field": "answer",           "Type": "string",       "Description": "Generated answer grounded in retrieved context"},
            {"Field": "sources",          "Type": "list[Source]", "Description": "Cited chunks with doc name, page, text, score"},
            {"Field": "confidence_score", "Type": "float [0-1]",  "Description": "Composite anti-hallucination score"},
            {"Field": "reasoning_steps",  "Type": "list[string]", "Description": "ReAct thought / action / observation trace"},
            {"Field": "escalated",        "Type": "boolean",      "Description": "True if confidence < threshold"},
            {"Field": "latency_ms",       "Type": "int",          "Description": "End-to-end processing time in milliseconds"},
        ]), use_container_width=True, hide_index=True)
