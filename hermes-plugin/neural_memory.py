#!/usr/bin/env python3
"""
neural_memory.py - THE Unified Neural Memory API
One import to rule them all.

Architecture:
  C++ MSSQL (primary) ─── GraphNodes/GraphEdges/NeuralMemory tables
  C++ SQLite (fallback) ── memory.db local cache
  Python ──────────────── Dream engine, embedding, orchestration

Usage:
    from neural_memory import Memory
    
    mem = Memory()  # Auto-detects MSSQL vs SQLite
    mem.remember("The user has a dog named Lou")
    results = mem.recall("What pet does the user have?")
    mem.think(results[0].id)
    mem.consolidate()
    mem.close()
"""

import os
import re
import sys
from pathlib import Path
from typing import Optional

# Add python dir to path
sys.path.insert(0, str(Path(__file__).parent))

from embed_provider import EmbeddingProvider
from memory_client import NeuralMemory, SQLiteStore

# Try MSSQL, fall back to SQLite
try:
    from mssql_store import MSSQLStore
    HAS_MSSQL = True
except ImportError:
    HAS_MSSQL = False


class Memory:
    """
    Unified Neural Memory interface.
    
    Backend priority:
    1. MSSQL (via pyodbc) — when MSSQL is installed and running
    2. SQLite (via Python) — fallback when MSSQL unavailable
    """
    
    def __init__(self, 
                 db_path: Optional[str] = None,
                 embedding_backend: str = "auto",
                 use_cpp: bool = True,
                 use_mssql: Optional[bool] = None,
                 default_chunk_size: int = 512):
        
        Path.home().joinpath(".neural_memory").mkdir(parents=True, exist_ok=True)
        
        self._db_path = db_path or str(Path.home() / ".neural_memory" / "memory.db")
        self._default_chunk_size = default_chunk_size
        self._mssql_store = None
        self._sqlite_memory = None
        
        # Embedder (shared regardless of backend)
        from embed_provider import EmbeddingProvider
        self._embedder = EmbeddingProvider(backend=embedding_backend)
        self._dim = self._embedder.dim
        
        # Auto-detect MSSQL
        if use_mssql is None:
            use_mssql = bool(os.environ.get("MSSQL_SERVER") and os.environ.get("MSSQL_PASSWORD"))
        
        # Try MSSQL first
        if use_mssql:
            try:
                from mssql_store import MSSQLStore
                self._mssql_store = MSSQLStore()
                print(f"[neural] MSSQL backend: {self._embedder.backend.__class__.__name__} ({self._dim}d)")
                return  # MSSQL is primary — done
            except Exception as e:
                print(f"[neural] MSSQL unavailable ({e}), falling back to SQLite")
        
        # SQLite fallback
        from memory_client import NeuralMemory
        self._sqlite_memory = NeuralMemory(db_path=self._db_path, embedding_backend=embedding_backend)
        print(f"[neural] SQLite backend: {self._embedder.backend.__class__.__name__} ({self._dim}d)")
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
        """
        Split text into overlapping chunks at sentence boundaries.
        
        Each chunk is approximately chunk_size characters with overlap
        characters shared between adjacent chunks. Preserves sentence
        integrity (never cuts mid-sentence).
        
        Args:
            text: Text to chunk
            chunk_size: Target size of each chunk in characters (default 512)
            overlap: Number of overlapping characters between chunks (default 64)
        
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s for s in sentences if s.strip()]
        
        if not sentences:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            candidate = (current_chunk + " " + sentence).strip() if current_chunk else sentence
            
            if len(candidate) <= chunk_size:
                current_chunk = candidate
            else:
                # Current chunk is full — start a new one
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If single sentence exceeds chunk_size, add it alone
                if len(sentence) > chunk_size:
                    chunks.append(sentence)
                    current_chunk = ""
                else:
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        # Add overlap: prepend end of previous chunk to each chunk
        if overlap > 0 and len(chunks) > 1:
            overlapped = [chunks[0]]
            for i in range(1, len(chunks)):
                prev = chunks[i - 1]
                # Take last `overlap` chars, but start at a word boundary
                overlap_text = prev[-overlap:]
                space_idx = overlap_text.find(' ')
                if space_idx > 0:
                    overlap_text = overlap_text[space_idx + 1:]
                overlapped.append(overlap_text + " " + chunks[i])
            chunks = overlapped
        
        return chunks
    
    def remember_chunked(self, text: str, label: str = "",
                         chunk_size: int = None, overlap: int = 64) -> list[int]:
        """
        Store text as multiple chunked memories.
        
        Splits text into overlapping chunks at sentence boundaries and stores
        each chunk as a separate memory with the same base label.
        
        Args:
            text: Text to store
            label: Base label (chunk index appended)
            chunk_size: Chunk size (default: self._default_chunk_size)
            overlap: Overlap between chunks (default 64)
        
        Returns:
            List of memory IDs
        """
        if chunk_size is None:
            chunk_size = self._default_chunk_size
        
        chunks = self.chunk_text(text, chunk_size, overlap)
        
        if len(chunks) == 1:
            return [self.remember(chunks[0], label, auto_chunk=False)]
        
        ids = []
        for i, chunk in enumerate(chunks):
            chunk_label = f"{label} [chunk {i+1}/{len(chunks)}]" if label else f"[chunk {i+1}/{len(chunks)}]"
            ids.append(self.remember(chunk, chunk_label, auto_chunk=False))
        
        return ids
    
    def remember(self, text: str, label: str = "", auto_chunk: bool = True,
                 auto_connect: bool = True, detect_conflicts: bool = True) -> int | list[int]:
        """Store a memory. Returns memory ID."""
        if auto_chunk and len(text) > self._default_chunk_size * 2:
            return self.remember_chunked(text, label)
        
        embedding = self._embedder.embed(text)
        
        if self._mssql_store:
            return self._mssql_store.store(label or text[:60], text, embedding)
        else:
            return self._sqlite_memory.remember(text, label, auto_connect=auto_connect,
                                                detect_conflicts=detect_conflicts)
    
    def remember_embedding(self, embedding: list[float], label: str = "", 
                           content: str = "") -> int:
        """Store a memory with pre-computed embedding."""
        if self._mssql_store:
            return self._mssql_store.store(label or content[:60], content, embedding)
        else:
            return self._sqlite_memory.store.store(label or content[:60], content, embedding)
    
    def recall(self, query: str, k: int = 5) -> list[dict]:
        """Semantic search. Returns [{id, label, content, similarity}]."""
        embedding = self._embedder.embed(query)
        
        if self._mssql_store:
            results = self._mssql_store.recall(embedding, k)
            # Strip embeddings, add connections
            for r in results:
                r.pop('embedding', None)
                conns = self._mssql_store.get_connections(r['id'])
                r['connections'] = [{'id': c['target'] if c['source'] == r['id'] else c['source'],
                                     'weight': c['weight']} for c in conns[:3]]
            return results
        else:
            return self._sqlite_memory.recall(query, k)

    def recall_multihop(self, query: str, k: int = 5, hops: int = 2) -> list[dict]:
        """Multi-hop retrieval: cosine similarity + graph expansion."""
        results = self.recall(query, k)
        if not self._mssql_store:
            return results
        
        expanded = []
        seen = {r['id'] for r in results}
        for r in results:
            conns = self._mssql_store.get_connections(r['id'])
            for c in conns:
                other = c['target'] if c['source'] == r['id'] else c['source']
                if other not in seen:
                    seen.add(other)
                    mem = self._mssql_store.get(other)
                    if mem:
                        expanded.append({'id': other, 'label': mem['label'],
                                        'content': mem['content'], 'activation': c['weight']})
        return results[:k]

    def think(self, start_id: int, depth: int = 3, decay: float = 0.85) -> list[dict]:
        """Spreading activation from a starting memory."""
        if self._mssql_store:
            visited = {start_id}
            frontier = [start_id]
            results = []
            current_decay = decay
            
            for _ in range(depth):
                next_frontier = []
                for nid in frontier:
                    conns = self._mssql_store.get_connections(nid)
                    for c in conns:
                        other = c['target'] if c['source'] == nid else c['source']
                        if other not in visited:
                            visited.add(other)
                            activation = c['weight'] * current_decay
                            mem = self._mssql_store.get(other)
                            results.append({
                                'id': other,
                                'label': mem['label'] if mem else f'node_{other}',
                                'activation': round(activation, 4),
                            })
                            next_frontier.append(other)
                frontier = next_frontier
                current_decay *= decay
            
            results.sort(key=lambda x: -x['activation'])
            return results[:20]
        else:
            return self._sqlite_memory.think(start_id, depth, decay)
    
    def connections(self, mem_id: int) -> list[dict]:
        """Get connections for a memory."""
        if self._mssql_store:
            return self._mssql_store.get_connections(mem_id)
        return self._sqlite_memory.connections(mem_id) if self._sqlite_memory else []
    
    def graph(self) -> dict:
        """Knowledge graph stats."""
        if self._mssql_store:
            s = self._mssql_store.stats()
            return {
                'nodes': s['memories'],
                'edges': s['connections'],
                'backend': 'mssql',
            }
        return self._sqlite_memory.graph()
    
    def consolidate(self) -> int:
        """Run memory consolidation."""
        return 0  # TODO: implement MSSQL consolidation
    
    def stats(self) -> dict:
        """System statistics."""
        if self._mssql_store:
            s = self._mssql_store.stats()
            s['embedding_dim'] = self._dim
            s['embedding_backend'] = self._embedder.backend.__class__.__name__
            s['backend'] = 'mssql'
            return s
        
        s = self._sqlite_memory.stats()
        s['backend'] = 'sqlite'
        return s
    
    def close(self):
        """Clean shutdown."""
        if self._mssql_store:
            self._mssql_store.close()
            self._mssql_store = None
        if self._sqlite_memory:
            self._sqlite_memory.close()
            self._sqlite_memory = None
    
    @property
    def dim(self) -> int:
        return self._dim
    
    @property
    def backend(self) -> str:
        if self._mssql_store:
            return "mssql"
        return self._embedder.backend.__class__.__name__
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    def __repr__(self):
        return f"Memory(backend={self.backend}, dim={self.dim}, db={self._db_path})"
