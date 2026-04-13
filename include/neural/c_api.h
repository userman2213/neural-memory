// neural/c_api.h - C-compatible API for Python/ctypes integration
// All functions use extern "C" linkage for ABI stability.
#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef _WIN32
    #define NEURAL_API __declspec(dllexport)
#else
    #define NEURAL_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Opaque handle
// ============================================================================
typedef void* NeuralMemoryHandle;

// ============================================================================
// Result structures
// ============================================================================
typedef struct {
    uint64_t id;
    float*   embedding;      // Caller must NOT free (valid until next call)
    int      embedding_dim;
    char     label[256];
    char     content[4096];
    float    similarity;
    float    salience;
} NeuralMemoryResult;

typedef struct {
    size_t   episodic_count;
    size_t   semantic_count;
    float    episodic_occupancy;
    float    semantic_occupancy;
    size_t   hopfield_patterns;
    float    hopfield_occupancy;
    size_t   graph_nodes;
    size_t   graph_edges;
    float    graph_density;
    uint64_t avg_store_us;
    uint64_t avg_retrieve_us;
    uint64_t avg_search_us;
    uint64_t total_stores;
    uint64_t total_retrieves;
    uint64_t total_searches;
    uint64_t total_consolidations;
} NeuralMemoryStats;

// ============================================================================
// Lifecycle
// ============================================================================

// Create a new adapter with default config (384-dim vectors).
// Returns NULL on failure.
NEURAL_API NeuralMemoryHandle neural_memory_create(void);

// Create with explicit vector dimension.
NEURAL_API NeuralMemoryHandle neural_memory_create_dim(int vector_dim);

// Destroy adapter and free resources.
NEURAL_API void neural_memory_destroy(NeuralMemoryHandle handle);

// ============================================================================
// Core operations
// ============================================================================

// Store a vector. Returns memory ID (0 on failure).
NEURAL_API uint64_t neural_memory_store(
    NeuralMemoryHandle handle,
    const float* vec,
    int dim,
    const char* label,
    const char* content
);

// Store text (uses internal text->embedding).
NEURAL_API uint64_t neural_memory_store_text(
    NeuralMemoryHandle handle,
    const char* text,
    const char* label
);

// Retrieve top-k memories by vector similarity.
// Writes up to k results into ids[] and scores[].
// Returns actual number of results written.
NEURAL_API int neural_memory_retrieve(
    NeuralMemoryHandle handle,
    const float* vec,
    int dim,
    int k,
    uint64_t* ids,
    float* scores
);

// Retrieve top-k with full result detail.
// results must point to an array of at least k NeuralMemoryResult.
// Returns actual number of results written.
NEURAL_API int neural_memory_retrieve_full(
    NeuralMemoryHandle handle,
    const float* vec,
    int dim,
    int k,
    NeuralMemoryResult* results
);

// Text-based search.
NEURAL_API int neural_memory_search(
    NeuralMemoryHandle handle,
    const char* query,
    int k,
    uint64_t* ids,
    float* scores
);

// Read a specific memory by ID. Returns 1 on success, 0 if not found.
NEURAL_API int neural_memory_read(
    NeuralMemoryHandle handle,
    uint64_t id,
    NeuralMemoryResult* result
);

// ============================================================================
// Graph / Edge Operations (MSSQL-backed)
// ============================================================================

// Store a vector into NeuralMemory table + create GraphNode. Returns node ID.
NEURAL_API uint64_t neural_memory_store_mssql(
    NeuralMemoryHandle handle,
    const float* vec,
    int dim,
    const char* label,
    const char* content
);

// Add an edge to GraphEdges. Returns 1 on success.
NEURAL_API int neural_memory_add_edge(
    NeuralMemoryHandle handle,
    uint64_t from_id,
    uint64_t to_id,
    float weight,
    const char* edge_type
);

// Batch add edges. Returns number of edges added.
NEURAL_API int neural_memory_batch_add_edges(
    NeuralMemoryHandle handle,
    const uint64_t* from_ids,
    const uint64_t* to_ids,
    const float* weights,
    int count,
    const char* edge_type
);

// Batch strengthen edges: update weight = min(weight + delta, 1.0).
// from_ids/to_ids are parallel arrays of length count.
NEURAL_API int neural_memory_batch_strengthen_edges(
    NeuralMemoryHandle handle,
    const uint64_t* from_ids,
    const uint64_t* to_ids,
    int count,
    float delta
);

// Bulk weaken all edges: UPDATE GraphEdges SET weight = MAX(weight - delta, 0.0) WHERE weight > threshold.
// Then prunes edges below threshold. Returns number of edges pruned.
NEURAL_API int neural_memory_bulk_weaken_prune(
    NeuralMemoryHandle handle,
    float delta,
    float threshold
);

// Get all edges for a node (from OR to). Writes up to max_edges into buffer.
// Returns actual number of edges written.
// edge_ids[2*i] = from_id, edge_ids[2*i+1] = to_id, weights[i] = weight
NEURAL_API int neural_memory_get_edges(
    NeuralMemoryHandle handle,
    uint64_t node_id,
    uint64_t* edge_ids,     // Must hold max_edges * 2 uint64_t
    float* weights,         // Must hold max_edges float
    int max_edges
);

// Count edges in GraphEdges table.
NEURAL_API int64_t neural_memory_count_edges(NeuralMemoryHandle handle);

// Spreading activation via C++ (runs on GraphNodes/GraphEdges from MSSQL).
// Returns number of activated nodes. Writes to node_ids[] and activations[].
NEURAL_API int neural_memory_think(
    NeuralMemoryHandle handle,
    uint64_t start_id,
    int depth,
    uint64_t* node_ids,
    float* activations,
    int max_results
);

// ============================================================================
// Consolidation & Decay
// ============================================================================

// Force consolidation pass. Returns number of memories consolidated.
NEURAL_API size_t neural_memory_consolidate(NeuralMemoryHandle handle);

// Apply decay to all memories and edges.
NEURAL_API void neural_memory_decay(NeuralMemoryHandle handle);

// Run link prediction. Returns number of new edges added.
NEURAL_API size_t neural_memory_predict_links(NeuralMemoryHandle handle);

// ============================================================================
// Configuration
// ============================================================================

NEURAL_API void neural_memory_set_beta(NeuralMemoryHandle handle, float beta);
NEURAL_API float neural_memory_get_beta(NeuralMemoryHandle handle);
NEURAL_API void neural_memory_set_consolidation_threshold(NeuralMemoryHandle handle, float threshold);

// ============================================================================
// Statistics
// ============================================================================

NEURAL_API void neural_memory_stats(NeuralMemoryHandle handle, NeuralMemoryStats* stats);

#ifdef __cplusplus
}
#endif
