## pmr

Polymorphic Memory Resource (PMR) - analysis the lifetime for SWMR struct.

### Overview

C++17 introduced `std::pmr` (polymorphic memory resources) to provide runtime-polymorphic allocators. This enables memory allocation strategies to be selected at runtime without changing container types.

The region-based memory model extends PMR to support heterogeneous memory architectures, particularly CXL (Compute Express Link) memory systems.

### Memory Region Model

The region memory model defines three distinct memory classifications based on coherency and consistency guarantees:

| Region | Coherency | Consistency | Use Case |
|--------|-----------|-------------|----------|
| Local | Strict | Sequential | CPU-attached DDR memory |
| Middle | Relaxed | Acquire-Release | CXL-attached memory |
| Remote | Eventual | Relaxed | Pooled memory via CXL switches |

#### Local Region

CPU-attached memory with full hardware cache coherency:
- Strict coherency maintained by hardware
- Sequential consistency semantics
- Baseline latency and bandwidth
- Full atomic operation support

#### Middle Region

CXL Type-3 device memory with relaxed guarantees:
- Relaxed coherency requiring explicit barriers
- Acquire-release consistency semantics
- Supports bias modes (host/device) for cache optimization
- 1.5-3x latency overhead, ~80% relative bandwidth

#### Remote Region

Pooled memory accessible through CXL switches:
- Eventual consistency without hardware coherency
- Requires explicit software synchronization
- Best for bulk data staging and cold storage
- 5-10x latency overhead, ~40% relative bandwidth

### Core Enumerations

```cpp
namespace std::pmr {

enum class memory_region {
    local,   // CPU-attached, strict coherency
    middle,  // CXL-attached, relaxed coherency
    remote   // Pooled, eventual consistency
};

enum class coherency_model {
    strict,   // Hardware cache coherency
    relaxed,  // Software-assisted coherency
    eventual  // No coherency guarantees
};

enum class consistency_model {
    sequential,      // Total ordering
    acquire_release, // Synchronizes-with semantics
    relaxed          // No ordering guarantees
};

enum class bias_mode {
    host,   // Optimize for CPU access
    device  // Optimize for device access
};

}
```

### Region Traits

Compile-time region properties via template specialization:

```cpp
namespace std::pmr {

template<memory_region R>
struct region_traits;

template<>
struct region_traits<memory_region::local> {
    static constexpr coherency_model coherency = coherency_model::strict;
    static constexpr consistency_model consistency = consistency_model::sequential;
    static constexpr size_t coherency_granularity = hardware_destructive_interference_size;
    static constexpr bool supports_atomic_operations = true;
    static constexpr bool requires_explicit_flush = false;
};

template<>
struct region_traits<memory_region::middle> {
    static constexpr coherency_model coherency = coherency_model::relaxed;
    static constexpr consistency_model consistency = consistency_model::acquire_release;
    static constexpr size_t coherency_granularity = 64;  // CXL cache line
    static constexpr bool supports_atomic_operations = true;
    static constexpr bool requires_explicit_flush = true;
};

template<>
struct region_traits<memory_region::remote> {
    static constexpr coherency_model coherency = coherency_model::eventual;
    static constexpr consistency_model consistency = consistency_model::relaxed;
    static constexpr size_t coherency_granularity = 4096;  // Page granularity
    static constexpr bool supports_atomic_operations = false;
    static constexpr bool requires_explicit_flush = true;
};

}
```

### Memory Resource Interface

Region-aware polymorphic allocator extending `std::pmr::memory_resource`:

```cpp
namespace std::pmr {

class region_memory_resource : public memory_resource {
public:
    explicit region_memory_resource(memory_region region);

    memory_region get_region() const noexcept;
    coherency_model get_coherency() const noexcept;
    consistency_model get_consistency() const noexcept;

protected:
    void* do_allocate(size_t bytes, size_t alignment) override;
    void do_deallocate(void* p, size_t bytes, size_t alignment) override;
    bool do_is_equal(const memory_resource& other) const noexcept override;

private:
    memory_region region_;
};

// Factory functions
region_memory_resource* local_memory_resource() noexcept;
region_memory_resource* middle_memory_resource() noexcept;
region_memory_resource* remote_memory_resource() noexcept;

}
```

### Synchronization Primitives

Explicit synchronization for cross-region operations:

```cpp
namespace std::pmr {

// Memory barriers
void coherence_barrier(memory_region region);

// CXL bias control for middle region
void set_bias_mode(void* ptr, size_t size, bias_mode mode);

// Remote region flush
void flush_remote(void* ptr, size_t size);

// Async bulk transfers
std::future<void> async_get(void* local_dst, const void* remote_src, size_t size);
std::future<void> async_put(void* remote_dst, const void* local_src, size_t size);

}
```

#### Synchronization Semantics

| Primitive | Region | Effect |
|-----------|--------|--------|
| `coherence_barrier` | middle | Ensures all prior writes visible to other agents |
| `set_bias_mode` | middle | Switches cache ownership between host/device |
| `flush_remote` | remote | Ensures writes propagated to persistent storage |
| `async_get` | remote→local | DMA transfer from remote to local |
| `async_put` | local→remote | DMA transfer from local to remote |

### Pointer Annotations

Attribute syntax for region-aware pointers:

```cpp
// Annotated pointer declarations
int* [[region::local]] local_ptr;
int* [[region::middle]] cxl_ptr;
int* [[region::remote]] pooled_ptr;

// Function parameter annotations
void process_data(
    int* [[region::local]] input,
    int* [[region::remote]] output,
    size_t count);
```

### Usage Examples

#### Basic Allocation

```cpp
#include <memory_resource>
#include <vector>

// Local region allocation (default behavior)
std::pmr::region_memory_resource local_mem(std::pmr::memory_region::local);
std::pmr::vector<int> local_data(&local_mem);
local_data.resize(1000);

// Middle region allocation (CXL memory)
std::pmr::region_memory_resource cxl_mem(std::pmr::memory_region::middle);
std::pmr::vector<float> cxl_data(&cxl_mem);
cxl_data.resize(10000);
```

#### Cross-Region Data Transfer

```cpp
// Transfer from local to remote
std::pmr::region_memory_resource local_res(std::pmr::memory_region::local);
std::pmr::region_memory_resource remote_res(std::pmr::memory_region::remote);

auto* local_buffer = static_cast<int*>(local_res.allocate(4096 * sizeof(int)));
auto* remote_buffer = static_cast<int*>(remote_res.allocate(4096 * sizeof(int)));

// Fill local buffer
std::iota(local_buffer, local_buffer + 4096, 0);

// Async transfer to remote
auto future = std::pmr::async_put(remote_buffer, local_buffer, 4096 * sizeof(int));
future.wait();

// Ensure persistence
std::pmr::flush_remote(remote_buffer, 4096 * sizeof(int));
```

#### CXL Bias Mode Control

```cpp
std::pmr::region_memory_resource middle_res(std::pmr::memory_region::middle);
auto* buffer = middle_res.allocate(size);

// Switch to device bias for accelerator processing
std::pmr::set_bias_mode(buffer, size, std::pmr::bias_mode::device);

// ... device processes data ...

// Switch back to host bias
std::pmr::set_bias_mode(buffer, size, std::pmr::bias_mode::host);
std::pmr::coherence_barrier(std::pmr::memory_region::middle);

// Now safe to read from host
process_results(buffer);
```

#### Heterogeneous Memory Pool

```cpp
class tiered_memory_pool {
    std::pmr::region_memory_resource local_tier_{std::pmr::memory_region::local};
    std::pmr::region_memory_resource middle_tier_{std::pmr::memory_region::middle};
    std::pmr::region_memory_resource remote_tier_{std::pmr::memory_region::remote};

public:
    template<typename T>
    T* allocate_hot() {
        return static_cast<T*>(local_tier_.allocate(sizeof(T), alignof(T)));
    }

    template<typename T>
    T* allocate_warm() {
        return static_cast<T*>(middle_tier_.allocate(sizeof(T), alignof(T)));
    }

    template<typename T>
    T* allocate_cold() {
        return static_cast<T*>(remote_tier_.allocate(sizeof(T), alignof(T)));
    }

    void migrate_to_cold(void* ptr, size_t size) {
        auto* cold_ptr = remote_tier_.allocate(size);
        auto future = std::pmr::async_put(cold_ptr, ptr, size);
        future.wait();
        std::pmr::flush_remote(cold_ptr, size);
    }
};
```

### SWMR (Single Writer Multiple Reader) Pattern

For SWMR structures across memory regions:

```cpp
template<typename T>
class swmr_buffer {
    std::pmr::region_memory_resource& resource_;
    T* data_;
    std::atomic<uint64_t> version_{0};

public:
    explicit swmr_buffer(std::pmr::region_memory_resource& res, size_t count)
        : resource_(res)
        , data_(static_cast<T*>(res.allocate(count * sizeof(T), alignof(T))))
    {}

    // Writer: single producer
    void write(const T* src, size_t count) {
        std::memcpy(data_, src, count * sizeof(T));

        // Ensure visibility based on region
        if (resource_.get_region() == std::pmr::memory_region::middle) {
            std::pmr::coherence_barrier(std::pmr::memory_region::middle);
        } else if (resource_.get_region() == std::pmr::memory_region::remote) {
            std::pmr::flush_remote(data_, count * sizeof(T));
        }

        version_.fetch_add(1, std::memory_order_release);
    }

    // Reader: multiple consumers
    uint64_t read(T* dst, size_t count) {
        uint64_t ver = version_.load(std::memory_order_acquire);
        std::memcpy(dst, data_, count * sizeof(T));
        return ver;
    }
};
```

### Lifetime Considerations

Memory lifetime analysis for region-aware allocations:

1. **Local Region**: Standard C++ lifetime rules apply. Objects destroyed when deallocated.

2. **Middle Region**: Lifetime extends until explicit deallocation. Bias mode transitions don't affect lifetime.

3. **Remote Region**: Lifetime may persist across program invocations (persistent memory). Requires explicit management.

```cpp
// Lifetime spans across regions
template<typename T>
class cross_region_handle {
    T* local_copy_;
    T* remote_copy_;
    std::pmr::region_memory_resource& local_res_;
    std::pmr::region_memory_resource& remote_res_;

public:
    cross_region_handle(std::pmr::region_memory_resource& local,
                        std::pmr::region_memory_resource& remote)
        : local_res_(local), remote_res_(remote)
    {
        local_copy_ = static_cast<T*>(local.allocate(sizeof(T), alignof(T)));
        remote_copy_ = static_cast<T*>(remote.allocate(sizeof(T), alignof(T)));
    }

    ~cross_region_handle() {
        // Must deallocate from both regions
        local_res_.deallocate(local_copy_, sizeof(T), alignof(T));
        remote_res_.deallocate(remote_copy_, sizeof(T), alignof(T));
    }

    void sync_to_remote() {
        auto f = std::pmr::async_put(remote_copy_, local_copy_, sizeof(T));
        f.wait();
        std::pmr::flush_remote(remote_copy_, sizeof(T));
    }

    void sync_from_remote() {
        auto f = std::pmr::async_get(local_copy_, remote_copy_, sizeof(T));
        f.wait();
    }
};
```

### Standard Library Headers

| Header | Additions |
|--------|-----------|
| `<memory_resource>` | `region_memory_resource`, region enums, traits |
| `<memory>` | Region-aware allocators, pointer annotations |
| `<atomic>` | Region-specific atomic support |
| `<region_memory>` | Synchronization primitives, async transfers |

### Performance Guidelines

1. **Hot Path**: Use local region for frequently accessed data
2. **Capacity Tier**: Use middle region (CXL) for large working sets
3. **Cold Storage**: Use remote region for archival/infrequent access
4. **Batch Transfers**: Prefer `async_get`/`async_put` over pointer access for remote
5. **Bias Transitions**: Minimize bias mode switches in middle region

### Backward Compatibility

All existing PMR code works unchanged:
- Default allocations use local region semantics
- Existing containers work with region memory resources
- No ABI breaks with existing allocator interfaces
