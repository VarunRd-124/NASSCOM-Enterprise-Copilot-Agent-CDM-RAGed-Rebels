"""Persist user feedback — closes the loop in the architecture diagram.

Write target, in priority order:
  1. Delta table via SQL warehouse  (only if DATABRICKS_WAREHOUSE_ID is set)
  2. Local CSV file                 (default — no Databricks compute needed)
"""
import csv
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

try:
    from .config import CFG
except ImportError:
    from config import CFG

log = logging.getLogger(__name__)

# Local fallback file (override with FEEDBACK_FILE env var)
FEEDBACK_FILE = Path(os.getenv("FEEDBACK_FILE", "feedback_log.csv"))


def _write_local_csv(row: Dict) -> Dict:
    """Append one feedback row to a local CSV, creating the header on first write."""
    try:
        is_new = not FEEDBACK_FILE.exists()
        with FEEDBACK_FILE.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if is_new:
                writer.writeheader()
            writer.writerow(row)
        return {"persisted": True, "row": row, "target": str(FEEDBACK_FILE)}
    except Exception as e:
        log.exception("Local feedback write failed: %s", e)
        return {"persisted": False, "row": row, "error": str(e)}


def record_feedback(query: str, answer: str, model_used: str,
                    rating: str, comment: Optional[str], citations: list,
                    user_email: Optional[str]) -> Dict:
    """rating ∈ {'up','down'}. Writes one row to the feedback Delta table."""
    row = {
        "feedback_id": str(uuid.uuid4()),
        "ts": datetime.now(timezone.utc).isoformat(),
        "user_email": user_email or "anonymous",
        "query": query,
        "answer": answer,
        "model_used": model_used,
        "rating": rating,
        "comment": comment or "",
        "n_citations": len(citations),
        "citation_ids": ",".join(str(c.get("source_id", "")) for c in citations),
    }
    # No warehouse configured → persist to a local CSV file.
    wh = os.getenv("DATABRICKS_WAREHOUSE_ID")
    if not wh:
        return _write_local_csv(row)

    # Warehouse available → write to the Delta table via the SQL Statement Execution API.
    try:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.sql import StatementParameterListItem
        w = WorkspaceClient()
        cols = ",".join(row.keys())
        placeholders = ",".join(f":{k}" for k in row.keys())
        sql = f"INSERT INTO {CFG.feedback_table} ({cols}) VALUES ({placeholders})"
        params = [StatementParameterListItem(name=k, value=str(v)) for k, v in row.items()]
        w.statement_execution.execute_statement(
            statement=sql, warehouse_id=wh, parameters=params, wait_timeout="20s"
        )
        return {"persisted": True, "row": row, "target": CFG.feedback_table}
    except Exception as e:
        log.exception("Delta feedback write failed, falling back to local CSV: %s", e)
        return _write_local_csv(row)
