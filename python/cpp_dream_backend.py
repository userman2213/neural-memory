"""
CppDreamBackend — Dream backend that uses C++ bridge for MSSQL operations.

All graph operations (strengthen, weaken, prune, spreading activation)
go through the C++ libneural_memory.so → ODBC → MSSQL.

SQLite fallback is handled by SQLiteDreamBackend in dream_engine.py.
"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import the C++ bridge
sys.path.insert(0, str(Path(__file__).parent))
from cpp_bridge import NeuralMemoryCpp

from dream_engine import DreamBackend

logger = logging.getLogger(__name__)


class CppDreamBackend(DreamBackend):
    """Dream backend backed by C++ bridge → MSSQL.

    All graph operations (strengthen, weaken, prune, spreading activation)
    go through libneural_memory.so ODBC → MSSQL. No pyodbc in Python.

    For dream session tracking, we still use a lightweight SQLite file
    (dream sessions are metadata, not graph operations).
    """

    def __init__(self, dim: int = 384):
        import sqlite3

        self._cpp = NeuralMemoryCpp()
        self._cpp.initialize(dim=dim)
        logger.info("CppDreamBackend initialized (dim=%d)", dim)

        # Lightweight SQLite for dream session tracking only
        db_path = str(Path.home() / ".neural_memory" / "dream_sessions.db")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._session_db = db_path

        conn = sqlite3.connect(db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS dream_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at REAL NOT NULL,
                finished_at REAL,
                phase TEXT NOT NULL,
                memories_processed INTEGER DEFAULT 0,
                connections_strengthened INTEGER DEFAULT 0,
                connections_pruned INTEGER DEFAULT 0,
                bridges_found INTEGER DEFAULT 0,
                insights_created INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS dream_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                insight_type TEXT NOT NULL,
                source_memory_id INTEGER,
                content TEXT,
                confidence REAL DEFAULT 0.0,
                created_at REAL NOT NULL
            );
        """)
        conn.commit()
        conn.close()

    def close(self):
        """Shutdown C++ bridge."""
        if self._cpp:
            self._cpp.shutdown()
            self._cpp = None

    # -- Dream Sessions (lightweight SQLite for tracking) --

    def start_session(self, phase: str) -> int:
        import sqlite3, time
        conn = sqlite3.connect(self._session_db)
        cur = conn.execute(
            "INSERT INTO dream_sessions (started_at, phase) VALUES (?, ?)",
            (time.time(), phase)
        )
        conn.commit()
        sid = cur.lastrowid
        conn.close()
        return sid

    def finish_session(self, session_id: int, stats: Dict[str, Any]) -> None:
        if session_id < 0:
            return
        import sqlite3, time
        conn = sqlite3.connect(self._session_db)
        conn.execute(
            "UPDATE dream_sessions SET finished_at=?, memories_processed=?, "
            "connections_strengthened=?, connections_pruned=?, "
            "bridges_found=?, insights_created=? WHERE id=?",
            (time.time(),
             stats.get("processed", stats.get("explored", 0)),
             stats.get("strengthened", 0),
             stats.get("pruned", 0),
             stats.get("bridges", 0),
             stats.get("insights", 0),
             session_id)
        )
        conn.commit()
        conn.close()

    # -- Graph Operations (ALL via C++ → MSSQL) --

    def get_recent_memories(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent memories from MSSQL via C++ stats."""
        stats = self._cpp.get_stats()
        total = stats.get('graph_nodes', 0)
        if total == 0:
            return []
        # Return the last N IDs (MSSQL stores them sequentially)
        return [{"id": i} for i in range(max(1, total - limit + 1), total + 1)]

    def get_random_memories(self, limit: int = 30) -> List[Dict[str, Any]]:
        """Get a random sample of memories from the full history.

        Picks random IDs from the sequential ID space.
        """
        stats = self._cpp.get_stats()
        total = stats.get('graph_nodes', 0)
        if total == 0:
            return []
        sample_size = min(limit, total)
        ids = random.sample(range(1, total + 1), sample_size)
        return [{"id": i} for i in ids]

    def get_isolated_memories(self, max_connections: int = 3,
                               limit: int = 50,
                               oldest_first: bool = False) -> List[Dict[str, Any]]:
        """Find memories with few edges via C++ get_edges."""
        stats = self._cpp.get_stats()
        total_nodes = stats.get('graph_nodes', 0)
        if total_nodes == 0:
            return []

        # Scan direction: oldest-first (ascending) or newest-first (descending)
        if oldest_first:
            scan_range = range(1, total_nodes + 1)
        else:
            scan_range = range(total_nodes, 0, -1)

        isolated = []
        for node_id in scan_range:
            edges = self._cpp.get_edges(node_id, max_edges=20)
            if len(edges) < max_connections:
                isolated.append({
                    "id": node_id,
                    "connection_count": len(edges),
                })
            if len(isolated) >= limit:
                break
        return isolated

    def get_connections(self) -> List[Dict[str, Any]]:
        """Get all edges from MSSQL via C++ count + iterate."""
        # For the insight phase, we need all edges
        # C++ get_edges is per-node, so we iterate
        stats = self._cpp.get_stats()
        total_nodes = stats.get('graph_nodes', 0)
        if total_nodes == 0:
            return []

        all_edges = []
        seen = set()
        for node_id in range(1, min(total_nodes + 1, 500)):  # Cap for perf
            edges = self._cpp.get_edges(node_id, max_edges=100)
            for e in edges:
                key = (min(e['from_id'], e['to_id']), max(e['from_id'], e['to_id']))
                if key not in seen:
                    seen.add(key)
                    all_edges.append({
                        "source_id": e['from_id'],
                        "target_id": e['to_id'],
                        "weight": e['weight'],
                    })
        return all_edges

    def strengthen_connection(self, source_id: int, target_id: int,
                               delta: float = 0.05) -> None:
        """Strengthen via C++ batch_strengthen_edges."""
        self._cpp.batch_strengthen_edges([(source_id, target_id)], delta)

    def weaken_connection(self, source_id: int, target_id: int,
                           delta: float = 0.01) -> None:
        """Weaken individual edge — batch_strengthen with negative delta."""
        # For individual weaken, use add_edge with reduced weight
        # (C++ doesn't have single-weaken, use bulk for batch)
        edges = self._cpp.get_edges(source_id, max_edges=100)
        for e in edges:
            if e['to_id'] == target_id or e['from_id'] == target_id:
                new_w = max(e['weight'] - delta, 0.0)
                self._cpp.add_edge(e['from_id'], e['to_id'], new_w)
                break

    def add_bridge(self, source_id: int, target_id: int,
                    weight: float = 0.3) -> None:
        """Add bridge edge via C++."""
        self._cpp.add_edge(source_id, target_id, weight, "bridge")

    def prune_weak(self, threshold: float = 0.05) -> int:
        """Prune weak edges via C++ bulk_weaken_prune."""
        return self._cpp.bulk_weaken_prune(0.0, threshold)

    def log_connection_change(self, source_id: int, target_id: int,
                               old_weight: float, new_weight: float,
                               reason: str) -> None:
        """Skip — C++ handles this internally via GraphEdges updates."""
        pass

    def add_insight(self, session_id: int, insight_type: str,
                    source_memory_id: int, content: str,
                    confidence: float = 0.0) -> None:
        """Store insight in lightweight SQLite."""
        import sqlite3, time
        conn = sqlite3.connect(self._session_db)
        conn.execute(
            "INSERT INTO dream_insights "
            "(session_id, insight_type, source_memory_id, content, confidence, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, insight_type, source_memory_id, content, confidence, time.time())
        )
        conn.commit()
        conn.close()

    def get_dream_stats(self) -> Dict[str, Any]:
        """Get dream stats from SQLite tracking + C++ graph stats."""
        import sqlite3
        conn = sqlite3.connect(self._session_db)
        try:
            s = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(memories_processed),0), "
                "COALESCE(SUM(connections_strengthened),0), "
                "COALESCE(SUM(connections_pruned),0), "
                "COALESCE(SUM(bridges_found),0), "
                "COALESCE(SUM(insights_created),0) FROM dream_sessions"
            ).fetchone()
            return {
                "sessions": s[0] if s else 0,
                "total_processed": s[1] if s else 0,
                "total_strengthened": s[2] if s else 0,
                "total_pruned": s[3] if s else 0,
                "total_bridges": s[4] if s else 0,
                "total_insights": s[5] if s else 0,
                "graph_stats": self._cpp.get_stats(),
            }
        finally:
            conn.close()
