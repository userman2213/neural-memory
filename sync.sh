#!/bin/bash
# Sync source of truth (python/) to hermes-plugin/ and hermes-agent
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$SCRIPT_DIR/python"
PLUGIN="$SCRIPT_DIR/hermes-plugin"
HERMES="$HOME/.hermes/hermes-agent/plugins/memory/neural"

echo "Syncing: python/ → hermes-plugin/ + ~/.hermes/"

for f in memory_client.py cpp_bridge.py embed_provider.py neural_memory.py mssql_store.py dream_mssql_store.py dream_engine.py; do
    if [ -f "$SRC/$f" ]; then
        cp "$SRC/$f" "$PLUGIN/$f" 2>/dev/null && echo "  $f → hermes-plugin/"
        cp "$SRC/$f" "$HERMES/$f" 2>/dev/null && echo "  $f → ~/.hermes/"
    fi
done

# fast_ops (.pyx + .so)
if [ -f "$SRC/fast_ops.pyx" ]; then
    cp "$SRC/fast_ops.pyx" "$PLUGIN/" 2>/dev/null
    cp "$SRC/fast_ops.pyx" "$HERMES/" 2>/dev/null
fi
if ls "$SRC"/fast_ops.cpython*.so 1>/dev/null 2>&1; then
    cp "$SRC"/fast_ops.cpython*.so "$PLUGIN/" 2>/dev/null
    cp "$SRC"/fast_ops.cpython*.so "$HERMES/" 2>/dev/null
fi

echo "Done."
