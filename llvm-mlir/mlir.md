# Dialect
MLIR is a modern compiler infrastructure built for flexibility and extensibility. Instead of offering a fixed set of instructions and types, MLIR operates through dialects, which are collections of user-defined operations, attributes, and types that can interoperate. These operations are a generalization of IR instructions and can be complex, potentially containing nested regions with more IR, resulting in a hierarchical representation. Operations in MLIR define and use values according to single static assignment (SSA) rules. For instance, MLIR dialects can represent entire instruction sets like NVVM (a virtual IR for NVIDIA GPUs), other IRs such as LLVM IR, control flow constructs like loops, parallel programming models such as OpenMP and OpenACC, machine learning graphs, and more.
## scf
structured control flow
```c++
scf.for %i = 0 to 128 step %c1 {
  %y = scf.execute_region -> i32 {
    %x = load %A[%i] : memref<128xi32>
    scf.yield %x : i32
  }
}
```
scf.yield that passes the current values of all loop-carried variables to the next iteration.

## memref
a multi dimentions array

## parallel for
below can be lowered into NVVM dialect or OpenMP dialect.
```c++
func @launch(%h_out : memref<?xf32>, %h_in : memref<?xf32>, %n : i64) {
  // Parallel for across all blocks in a grid.
  parallel.for (%gx, %gy, %gz) = (0, 0, 0)
    to (grid.x, grid.y, grid.z) {
      // Shared memory = stack allocation in a block.
      %shared_val = memref.alloca : memref<f32>
      // Parallel for across all threads in a block.
      parallel.for (%tx, %ty, %tz) = (0, 0, 0)
        to (blk.x, blk.y, blk.z) {
          // Control-flow is directly preserved.
          if %tx == 0 {
            %sum = func.call @sum(%d_in, %n)
            memref.store %sum, %shared_val[] : memref<f32>
          }
          // Syncronization via explicit operation.
          polygeist.barrier(%tx, %ty, %tz)
          %tid = %gx + grid.x * %tx
          if %tid < %n {
            %res = ...
            store %res, %d_out[%tid] : memref<?xf32>
          }
        }
    }
}
```
## MLIR JIT
mlir-cpu-runner

## Heterogeneous Memory Dialects

### [Remotable Pointers (CIRA)](remotable_pointer.md)

For CXL-attached and disaggregated memory systems, standard `memref` abstractions are insufficient. The CIRA dialect introduces remotable pointer types that enable:

- **Asynchronous memory access**: Decouple load initiation from value consumption
- **Prefetch streams**: Compiler-directed prefetching for pointer-chasing workloads
- **Heterogeneous execution**: Offload memory management to near-memory accelerators

Key types:
```mlir
!cira.handle<T>   // Pointer to CXL-resident data
!cira.future<T>   // In-flight async load token
!cira.stream      // Recurring access pattern descriptor
```

Example transformation:
```mlir
// Before: Serial pointer chase (stalls on CXL latency)
%next = cira.load %node[1] : !cira.handle<Node> -> !cira.handle<Node>

// After: Stream-based prefetch (data arrives from LLC)
%stream = cira.stream_create_indirect %head, %offset : ...
cira.offload @vortex { cira.prefetch_chain %stream, %depth }
%future = cira.peek_stream %stream : !cira.stream -> !cira.future<Node>
%next = cira.await %future : !cira.future<Node> -> !cira.handle<Node>
```

## Write a pass that emit two dialect
### Conversion
MLIR to MLIR rewrite.
### Dialect Definition
Define Dialect Operation and Attributes. Attributes are embedded for static Dataflow computation.
### Transform
Transform the dialect to target dialect.
### Lowering
MLIR to LLVM IR rewrite.


## Reference
1. [使用MLIR完成一个端到端的编译流程 -- 一条通路](https://zhuanlan.zhihu.com/p/328993481)