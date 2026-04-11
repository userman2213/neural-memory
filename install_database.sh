#!/bin/bash
# install_database.sh — Neural Memory Database Setup
# Sets up the NeuralMemory database (SQLite and/or MSSQL) with all tables
# including dream engine tables.
#
# Usage: bash install_database.sh [--lite|--full]
#   --lite   SQLite only (no MSSQL dependency)
#   --full   MSSQL + SQLite (production)
#   (no arg) Interactive menu
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NEURAL_DIR="$HOME/.neural_memory"
ENV_FILE="$HOME/.hermes/.env"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_ok()   { echo -e "  ${GREEN}[OK]${NC} $1"; }
print_info() { echo -e "  ${BLUE}[..]${NC} $1"; }
print_warn() { echo -e "  ${YELLOW}[!!]${NC} $1"; }
print_err()  { echo -e "  ${RED}[XX]${NC} $1"; }

# ---------------------------------------------------------------------------
# SQLite Setup (always included)
# ---------------------------------------------------------------------------
setup_sqlite() {
    echo ""
    echo -e "${CYAN}--- SQLite Setup ---${NC}"

    mkdir -p "$NEURAL_DIR"

    local DB_PATH="$NEURAL_DIR/memory.db"
    python3 << PYEOF
import sqlite3, os, sys

db = "$DB_PATH"
conn = sqlite3.connect(db)

# Core tables
conn.executescript("""
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT,
    content TEXT,
    embedding BLOB,
    salience REAL DEFAULT 1.0,
    created_at REAL DEFAULT 0,
    last_accessed REAL DEFAULT 0,
    access_count INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS connections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER,
    target_id INTEGER,
    weight REAL DEFAULT 0.5,
    created_at REAL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_mem_label ON memories(label);
CREATE INDEX IF NOT EXISTS idx_conn_source ON connections(source_id);
CREATE INDEX IF NOT EXISTS idx_conn_target ON connections(target_id);

-- Dream tables
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
CREATE TABLE IF NOT EXISTS connection_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL,
    target_id INTEGER NOT NULL,
    old_weight REAL,
    new_weight REAL,
    reason TEXT,
    changed_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dream_insights_type ON dream_insights(insight_type);
CREATE INDEX IF NOT EXISTS idx_dream_insights_session ON dream_insights(session_id);
CREATE INDEX IF NOT EXISTS idx_conn_history_nodes ON connection_history(source_id, target_id);
""")

# Verify
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
print(f"  Tables: {', '.join(t[0] for t in tables)}")
conn.close()
print(f"  Database: {db}")
print("  SQLite: OK")
PYEOF
    print_ok "SQLite database ready: $DB_PATH"
}

# ---------------------------------------------------------------------------
# MSSQL Setup (Full Stack only)
# ---------------------------------------------------------------------------
setup_mssql() {
    echo ""
    echo -e "${CYAN}--- MSSQL Setup ---${NC}"

    # Check if MSSQL is running
    if ! systemctl is-active --quiet mssql-server 2>/dev/null; then
        print_warn "MSSQL service not running. Trying to start..."
        sudo systemctl start mssql-server 2>/dev/null || {
            print_err "Cannot start MSSQL. Is it installed? Run install.sh first."
            return 1
        }
    fi
    print_ok "MSSQL service running"

    # Read credentials from .env or prompt
    local SA_PASS=""
    if [ -f "$ENV_FILE" ]; then
        SA_PASS=$(grep "^MSSQL_PASSWORD=" "$ENV_FILE" 2>/dev/null | cut -d= -f2- | tr -d '"' | tr -d "'")
    fi

    if [ -z "$SA_PASS" ]; then
        echo ""
        echo -n "  Enter MSSQL SA password: "
        read -s SA_PASS
        echo ""
    fi

    if [ -z "$SA_PASS" ]; then
        print_err "No SA password provided."
        return 1
    fi

    local SQLCMD="/opt/mssql-tools/bin/sqlcmd -S localhost -U sa -P '$SA_PASS' -C"

    # Create database
    print_info "Creating NeuralMemory database..."
    eval "$SQLCMD -Q \"
        IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = 'NeuralMemory')
            CREATE DATABASE NeuralMemory;
    \"" 2>/dev/null
    print_ok "Database NeuralMemory"

    # Create tables
    print_info "Creating tables..."
    eval "$SQLCMD -d NeuralMemory -Q \"
        -- Core tables
        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'memories')
        CREATE TABLE memories (
            id BIGINT IDENTITY(1,1) PRIMARY KEY,
            label NVARCHAR(256),
            content NVARCHAR(MAX),
            embedding VARBINARY(8000),
            vector_dim INT NOT NULL DEFAULT 384,
            salience FLOAT DEFAULT 1.0,
            created_at DATETIME2(7) DEFAULT SYSUTCDATETIME(),
            last_accessed DATETIME2(7) DEFAULT SYSUTCDATETIME(),
            access_count INT DEFAULT 0
        );

        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'connections')
        CREATE TABLE connections (
            id BIGINT IDENTITY(1,1) PRIMARY KEY,
            source_id BIGINT,
            target_id BIGINT,
            weight FLOAT DEFAULT 0.5,
            edge_type NVARCHAR(50) DEFAULT 'similar',
            created_at DATETIME2(7) DEFAULT SYSUTCDATETIME()
        );

        IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_conn_source')
        CREATE INDEX idx_conn_source ON connections(source_id);

        IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_conn_target')
        CREATE INDEX idx_conn_target ON connections(target_id);

        -- Dream tables
        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'dream_sessions')
        CREATE TABLE dream_sessions (
            id BIGINT IDENTITY(1,1) PRIMARY KEY,
            started_at FLOAT NOT NULL,
            finished_at FLOAT,
            phase NVARCHAR(20) NOT NULL,
            memories_processed INT DEFAULT 0,
            connections_strengthened INT DEFAULT 0,
            connections_pruned INT DEFAULT 0,
            bridges_found INT DEFAULT 0,
            insights_created INT DEFAULT 0
        );

        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'dream_insights')
        CREATE TABLE dream_insights (
            id BIGINT IDENTITY(1,1) PRIMARY KEY,
            session_id BIGINT,
            insight_type NVARCHAR(50) NOT NULL,
            source_memory_id BIGINT,
            content NVARCHAR(MAX),
            confidence FLOAT DEFAULT 0.0,
            created_at FLOAT NOT NULL
        );

        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'connection_history')
        CREATE TABLE connection_history (
            id BIGINT IDENTITY(1,1) PRIMARY KEY,
            source_id BIGINT NOT NULL,
            target_id BIGINT NOT NULL,
            old_weight FLOAT,
            new_weight FLOAT,
            reason NVARCHAR(100),
            changed_at FLOAT NOT NULL
        );

        IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_dream_insights_type')
        CREATE INDEX idx_dream_insights_type ON dream_insights(insight_type);

        IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_dream_insights_session')
        CREATE INDEX idx_dream_insights_session ON dream_insights(session_id);

        IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_dream_conn_history')
        CREATE INDEX idx_dream_conn_history ON connection_history(source_id, target_id);
    \"" 2>/dev/null
    print_ok "All tables created"

    # Verify
    local TABLES=$(eval "$SQLCMD -d NeuralMemory -Q 'SELECT name FROM sys.tables ORDER BY name' -h -1 -W" 2>/dev/null | grep -v '^$' | grep -v '^-' | tr '\n' ', ' | sed 's/,$//')
    print_ok "Tables: $TABLES"

    # Save to .env
    print_info "Updating $ENV_FILE..."
    if [ -f "$ENV_FILE" ]; then
        # Remove old MSSQL entries
        sed -i '/^MSSQL_/d' "$ENV_FILE" 2>/dev/null
    fi
    cat >> "$ENV_FILE" << ENVEOF

# MSSQL (Neural Memory)
MSSQL_SERVER=localhost
MSSQL_DATABASE=NeuralMemory
MSSQL_USERNAME=SA
MSSQL_PASSWORD=$SA_PASS
MSSQL_DRIVER={ODBC Driver 18 for SQL Server}
ENVEOF
    print_ok "Credentials saved to $ENV_FILE"
}

# ---------------------------------------------------------------------------
# Verify installation
# ---------------------------------------------------------------------------
verify() {
    echo ""
    echo -e "${CYAN}--- Verification ---${NC}"

    python3 << 'PYEOF'
import sys
sys.path.insert(0, "python")

# SQLite check
import sqlite3, os
db = os.path.expanduser("~/.neural_memory/memory.db")
if os.path.exists(db):
    conn = sqlite3.connect(db)
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    conn.close()
    print(f"  [OK] SQLite: {len(tables)} tables ({db})")
else:
    print(f"  [!!] SQLite: not found at {db}")

# MSSQL check (optional)
try:
    import pyodbc
    from dream_mssql_store import DreamMSSQLStore
    store = DreamMSSQLStore()
    stats = store.get_dream_stats()
    store.close()
    print(f"  [OK] MSSQL: {stats.get('sessions', 0)} dream sessions")
except ImportError:
    print("  [--] MSSQL: pyodbc not installed (OK for lite mode)")
except Exception as e:
    print(f"  [--] MSSQL: not available ({e})")

# sentence-transformers check
try:
    import sentence_transformers
    print(f"  [OK] sentence-transformers: {sentence_transformers.__version__}")
except ImportError:
    print("  [--] sentence-transformers: not installed (OK for lite mode)")
PYEOF
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "  Neural Memory — Database Setup"
echo "=============================================="

MODE="${1:-}"

if [ -z "$MODE" ]; then
    echo ""
    echo "  Select installation mode:"
    echo ""
    echo -e "    ${GREEN}[1]${NC} Lite        — SQLite only, hash/tfidf embeddings"
    echo -e "                     Budget VPS friendly (~50MB RAM, no GPU)"
    echo ""
    echo -e "    ${BLUE}[2]${NC} Full Stack  — SQLite + MSSQL + sentence-transformers"
    echo -e "                     Production (~500MB RAM, optional GPU)"
    echo ""
    echo -n "  Choice [1/2]: "
    read -n 1 -r CHOICE
    echo ""

    case "$CHOICE" in
        2) MODE="full" ;;
        *) MODE="lite" ;;
    esac
fi

case "$MODE" in
    --full|full)
        echo -e "  Mode: ${BLUE}Full Stack${NC}"
        setup_sqlite
        setup_mssql
        ;;
    --lite|lite|*)
        echo -e "  Mode: ${GREEN}Lite${NC}"
        setup_sqlite
        echo ""
        print_warn "MSSQL skipped (lite mode). Run with --full for MSSQL."
        ;;
esac

verify

echo ""
echo "=============================================="
echo "  Database Setup Complete!"
echo "=============================================="
echo ""
echo "  SQLite: ~/.neural_memory/memory.db"
if [ "$MODE" = "full" ] || [ "$MODE" = "--full" ]; then
    echo "  MSSQL:  localhost/NeuralMemory"
fi
echo ""
echo "  Next: bash install.sh (install plugin)"
echo ""
