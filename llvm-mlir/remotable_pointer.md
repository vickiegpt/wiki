# MLIR Remotable Pointers for Heterogeneous Memory

This document explores the design of remotable pointer abstractions in MLIR, drawing from the CIRA (CXL Intermediate Representation for Asynchrony) framework for compiler-driven heterogeneous memory access.

## Motivation: The CXL Memory Challenge

Traditional MLIR memory abstractions (`memref`, raw pointers) assume uniform memory access latency. With CXL-attached memory, this assumption breaks down:

| Memory Type | Latency | Characteristics |
|-------------|---------|-----------------|
| Local DRAM | ~80ns | Low, predictable |
| CXL Memory | 150-300ns | Higher, variable |
| Remote CXL | 300-500ns | Highest, network-dependent |

The **memory wall problem** is amplified: pointer-chasing workloads stall 65%+ of execution time on CXL memory accesses due to serialized dependency chains.

## Remotable Pointer Type System

### Core Types

CIRA extends MLIR's type system with CXL-aware abstractions:

```mlir
// A handle to memory guaranteed to reside in CXL space
!cira.handle<T>

// A future representing an in-flight asynchronous load
!cira.future<T>

// A descriptor for recurring memory access patterns
!cira.stream
```

### Type Definitions in TableGen

```tablegen
// CXL Memory Handle - typed pointer to remote memory
def CIRA_HandleType : CIRA_Type<"Handle", "handle"> {
  let summary = "A pointer guaranteed to reside in CXL memory space";
  let description = [{
    Represents a typed reference to data in CXL-attached memory.
    Preserves aliasing information necessary for correct LLC management.
    Can be dereferenced synchronously or asynchronously.
  }];
  let parameters = (ins "Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
}

// Asynchronous Future - represents pending memory operation
def CIRA_FutureType : CIRA_Type<"Future", "future"> {
  let summary = "Token representing an in-flight asynchronous load";
  let description = [{
    Decouples memory access initiation from value consumption.
    Enables computation overlap with memory latency.
    Must be awaited before the loaded value can be used.
  }];
  let parameters = (ins "Type":$valueType);
  let assemblyFormat = "`<` $valueType `>`";
}

// Stream Descriptor - captures recurring access patterns
def CIRA_StreamType : CIRA_Type<"Stream", "stream"> {
  let summary = "Descriptor for a recurring memory access pattern";
  let description = [{
    Encodes stride, indirect access chains, or learned patterns.
    Used by prefetch engines to drive speculative memory access.
    Can represent regular (array) or irregular (pointer-chasing) patterns.
  }];
}
```

## Memory Operations

### Asynchronous Load/Store

Separate initiation from completion to enable latency hiding:

```mlir
// Initiate async load - returns immediately with a future
%future = cira.load_async %handle : !cira.handle<f32> -> !cira.future<f32>

// ... perform independent computation here ...

// Block until load completes and extract value
%value = cira.await %future : !cira.future<f32> -> f32

// Async store (fire-and-forget with optional fence)
cira.store_async %value, %handle : f32, !cira.handle<f32>
```

### Prefetch Operations

Guide memory-side accelerators for intelligent prefetching:

```mlir
// Stream prefetch for regular access patterns (arrays)
cira.prefetch_stream %base, %stride, %count : !cira.handle<f32>, index, index

// Indirect prefetch for pointer-chasing (linked structures)
cira.prefetch_indirect %head, %offset, %depth : !cira.handle<!cira.handle<Node>>, index, index

// Install cache line explicitly at specified LLC level
cira.install_cacheline %addr, %level : !cira.handle<i8>, i32

// Hint for cache eviction (prevent pollution from streaming data)
cira.evict_hint %addr : !cira.handle<i8>
```

### Synchronization Primitives

Coordinate execution across heterogeneous processors:

```mlir
// Barrier between host CPU and memory-side accelerator
cira.barrier

// Signal completion of computation phase
cira.release

// Create explicit synchronization token
%token = cira.sync_token_create
cira.sync_token_wait %token
```

## Stream-Based Pointer Chasing

The key innovation: transform serial pointer chains into prefetch streams.

### Original Code (Problematic)

```mlir
// Standard linked list traversal - each load depends on previous
func.func @traverse_list(%head: !cira.handle<Node>) -> f32 {
  %zero = arith.constant 0.0 : f32
  %result = scf.while (%node = %head, %sum = %zero)
      : (!cira.handle<Node>, f32) -> f32 {
    %is_null = cira.handle_is_null %node : !cira.handle<Node>
    scf.condition(%is_null) %sum : f32
  } do {
  ^bb0(%node: !cira.handle<Node>, %sum: f32):
    // PROBLEM: This load stalls 150-300ns on CXL memory
    %data = cira.load %node[0] : !cira.handle<Node> -> f32
    %next = cira.load %node[1] : !cira.handle<Node> -> !cira.handle<Node>
    %new_sum = arith.addf %sum, %data : f32
    scf.yield %next, %new_sum : !cira.handle<Node>, f32
  }
  return %result : f32
}
```

### Transformed Code (CIRA Optimized)

```mlir
func.func @traverse_list_optimized(%head: !cira.handle<Node>) -> f32 {
  %zero = arith.constant 0.0 : f32
  %c8 = arith.constant 8 : index   // Offset to 'next' pointer
  %c16 = arith.constant 16 : index // Prefetch depth

  // Create indirect stream for pointer chasing
  %stream = cira.stream_create_indirect %head, %c8 :
      !cira.handle<Node>, index -> !cira.stream

  // Offload prefetching to memory-side Vortex RISC-V core
  cira.offload @vortex_core_0 {
    cira.prefetch_chain %stream, %c16 : !cira.stream, index
  }

  // Main loop - data now arrives from LLC instead of CXL
  %result = scf.while (%node = %head, %sum = %zero)
      : (!cira.handle<Node>, f32) -> f32 {
    %is_null = cira.handle_is_null %node : !cira.handle<Node>
    scf.condition(%is_null) %sum : f32
  } do {
  ^bb0(%node: !cira.handle<Node>, %sum: f32):
    // Peek stream - data prefetched by Vortex core
    %future = cira.peek_stream %stream : !cira.stream -> !cira.future<Node>
    %node_data = cira.await %future : !cira.future<Node> -> Node

    %data = cira.extract %node_data[0] : Node -> f32
    %next = cira.extract %node_data[1] : Node -> !cira.handle<Node>

    %new_sum = arith.addf %sum, %data : f32
    cira.advance_stream %stream : !cira.stream
    scf.yield %next, %new_sum : !cira.handle<Node>, f32
  }
  return %result : f32
}
```

## Dialect Integration

### Frontend Convergence

CIRA acts as a convergence point for multiple MLIR frontends:

```
ClangIR (C++ semantics)  ─┐
                          ├──> CIRA IR ──> Heterogeneous Backends
TOSA (Tensor ops)        ─┤                    ├── x86 (computation)
                          │                    └── RISC-V SIMT (memory mgmt)
SCF (Structured control) ─┘
```

### Lowering from SCF

```mlir
// Input: SCF loop with memory access
scf.for %i = %c0 to %n step %c1 {
  %ptr = memref.load %ptrs[%i] : memref<?x!cira.handle<f32>>
  %val = cira.load %ptr : !cira.handle<f32> -> f32
  // ... use %val ...
}

// After CIRA transformation: prefetch + async access
%stream = cira.stream_create_strided %ptrs, %c0, %c1, %n : ...
cira.offload @vortex {
  cira.prefetch_stream %stream, %prefetch_distance : !cira.stream, index
}
scf.for %i = %c0 to %n step %c1 {
  %future = cira.peek_stream %stream : !cira.stream -> !cira.future<f32>
  %val = cira.await %future : !cira.future<f32> -> f32
  cira.advance_stream %stream
  // ... use %val ...
}
```

### Lowering from TOSA

Tensor operations benefit from bulk prefetching:

```mlir
// TOSA matmul with CXL-resident weights
tosa.matmul %input, %weights : (tensor<64x128xf32>, tensor<128x256xf32>)
    -> tensor<64x256xf32>

// Lowered with tiled prefetching
%tile_stream = cira.stream_create_tiled %weights, %tile_size : ...
cira.offload @vortex {
  cira.prefetch_tiles %tile_stream, %lookahead : !cira.stream, index
}
// ... tiled matmul with prefetched tiles in LLC ...
```

## Backend Code Generation

### Partitioning Strategy

The compiler partitions operations between processor types:

| Operation Type | Target | Rationale |
|---------------|--------|-----------|
| Arithmetic, SIMD | x86 | Superior single-thread perf |
| Memory prefetch | Vortex RISC-V | Power-efficient parallelism |
| Cache management | Vortex RISC-V | Near-memory positioning |
| Synchronization | Both | Coordination required |

### Cost Model

Offloading decision based on latency hiding potential:

```
Gain = Σ (L_CXL - L_LLC) - (C_sync + C_vortex_busy)
       i∈Ops

Where:
  L_CXL    = CXL memory latency (~200ns)
  L_LLC    = LLC hit latency (~15ns)
  C_sync   = Synchronization overhead (~50ns)
  C_vortex = Vortex core utilization cost
```

Only offload when dependency chain depth × latency saving > synchronization cost.

### x86 Backend

```mlir
// CIRA operation
%future = cira.load_async %handle : !cira.handle<f32> -> !cira.future<f32>

// Lowered to LLVM IR with runtime calls
%req_id = call @cira_runtime_async_load(%handle_ptr, %size)
// ... independent computation ...
%value = call @cira_runtime_await(%req_id)
```

### Vortex RISC-V Backend

```mlir
// CIRA prefetch chain
cira.prefetch_chain %stream, %depth : !cira.stream, index

// Lowered to Vortex SIMT kernel
// Uses ws_spawn for warp-level parallelism
vortex.kernel @prefetch_chain {
  %warp_id = vortex.warp_id
  %lane_id = vortex.lane_id
  %addr = compute_prefetch_addr(%stream, %warp_id, %lane_id)
  vortex.prefetch %addr
}
```

## Runtime Profiling and Adaptation

### Feedback Loop

```
                    ┌─────────────────┐
                    │ Runtime Profiler │
                    │ (PMU counters)   │
                    └────────┬────────┘
                             │ access patterns,
                             │ miss rates
                             ▼
┌──────────┐    ┌─────────────────────┐    ┌──────────────┐
│ CIRA IR  │───>│ Adaptive Optimizer  │───>│ Recompiled   │
│          │    │ (pattern learning)  │    │ Code         │
└──────────┘    └─────────────────────┘    └──────────────┘
```

### Adaptive Policies

- **Prefetch Distance**: Increased when memory stalls dominate; reduced when cache pollution detected
- **Offload Granularity**: More operations offloaded under high memory pressure
- **Stream Depth**: Adjusted based on branch prediction accuracy for indirect streams

## Example: Graph BFS with Remotable Pointers

```mlir
module @graph_bfs {
  // Graph stored in CXL memory
  func.func @bfs(%graph: !cira.handle<Graph>, %source: index)
      -> memref<?xi32> {
    %frontier = memref.alloc(%num_vertices) : memref<?xi32>
    %next_frontier = memref.alloc(%num_vertices) : memref<?xi32>

    // Initialize
    memref.store %c1, %frontier[%source] : memref<?xi32>

    scf.while (%level = %c0) : (index) -> () {
      %frontier_empty = call @is_empty(%frontier) : ...
      scf.condition(%frontier_empty)
    } do {
    ^bb0(%level: index):
      // Create stream for frontier vertices' adjacency lists
      %adj_stream = cira.stream_create_frontier %graph, %frontier :
          !cira.handle<Graph>, memref<?xi32> -> !cira.stream

      // Offload adjacency list prefetching
      cira.offload @vortex {
        // Vortex cores prefetch neighbor lists in parallel
        cira.prefetch_adjacency %adj_stream, %lookahead :
            !cira.stream, index
      }

      // Process frontier (data arrives from LLC)
      scf.parallel (%v) = (%c0) to (%frontier_size) step (%c1) {
        %vertex = memref.load %frontier[%v] : memref<?xi32>
        %adj_future = cira.peek_adjacency %adj_stream, %v :
            !cira.stream, index -> !cira.future<AdjList>
        %neighbors = cira.await %adj_future : !cira.future<AdjList>

        // Process neighbors
        scf.for %n = %c0 to %num_neighbors step %c1 {
          %neighbor = cira.extract %neighbors[%n] : AdjList -> index
          // ... update next_frontier ...
        }
        cira.advance_stream %adj_stream
      }

      // Swap frontiers
      call @swap_frontiers(%frontier, %next_frontier) : ...
      %next_level = arith.addi %level, %c1 : index
      scf.yield %next_level : index
    }

    return %frontier : memref<?xi32>
  }
}
```

## Performance Characteristics

Based on CIRA evaluation:

| Workload | Baseline | With Remotable Pointers | Speedup |
|----------|----------|------------------------|---------|
| MCF (pointer-chasing) | 1.0x | 1.52x | 52% |
| Graph BFS | 1.0x | 1.38x | 38% |
| Graph SSSP | 1.0x | 1.45x | 45% |
| PageRank | 1.0x | 1.24x | 24% |
| TPC-H Queries | 1.0x | 1.18x | 18% |

**Key insight**: Latency-critical workloads (pointer-chasing) benefit most (1.2-1.5x), while compute-bound workloads see minimal improvement.

## Implementation Notes

### Adding to MLIR

1. **Define the dialect** in `mlir/include/mlir/Dialect/CIRA/`
2. **Implement operations** with custom verifiers for type safety
3. **Write conversion passes** from SCF/TOSA to CIRA
4. **Implement backend lowering** to LLVM (x86) and Vortex (RISC-V)
5. **Build runtime library** for async memory operations

### Key Files

```
mlir/
├── include/mlir/Dialect/CIRA/
│   ├── CIRADialect.td       # Dialect definition
│   ├── CIRAOps.td           # Operation definitions
│   ├── CIRATypes.td         # Type definitions
│   └── CIRAInterfaces.td    # Memory/async interfaces
├── lib/Dialect/CIRA/
│   ├── CIRADialect.cpp      # Dialect implementation
│   ├── CIRAOps.cpp          # Operation implementations
│   └── Transforms/
│       ├── StreamCreation.cpp    # Create streams from loops
│       ├── OffloadPartition.cpp  # Partition computation
│       └── AsyncConversion.cpp   # Convert sync to async
└── lib/Conversion/
    ├── CIRAToLLVM/          # x86 backend
    └── CIRAToVortex/        # RISC-V SIMT backend
```

## References

1. CIRA: Revolutionizing Memory Access Through Compiler-Driven Heterogeneous Execution (2025)
2. [MLIR Documentation](https://mlir.llvm.org/)
3. [Vortex RISC-V GPGPU](https://github.com/vortexgpgpu/vortex)
4. [CXL Specification](https://computeexpresslink.org/)
5. Ainsworth & Jones - Software Prefetching for Indirect Memory Accesses (2019)
