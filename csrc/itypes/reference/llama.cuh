#pragma once

#define LLAMA_GLOBAL_WORK_QUEUE
#define LLAMA_STRICT_BARRIERS
#define LLAMA_BROADCAST_LM_HEAD_NORM
// #define LLAMA_DETERMINISTIC
// #define PRINT_DEADLOCKS

#define OPCODE_AttnNorm 1
#define OPCODE_QKV_RopeAppend 2
#define OPCODE_GQA_AttentionPrefill 3
#define OPCODE_GQA_AttentionDecode 4
#define OPCODE_O_ProjResidual 5

#define OPCODE_MlpNorm 6
#define OPCODE_GateSiLU 7
#define OPCODE_UpMatmul 8
#define OPCODE_DownProjResidual 9

#define OPCODE_LM_HeadNorm 10
#define OPCODE_LM_Head 11
#define OPCODE_Barrier_Inc 12
#define OPCODE_AllDeviceBarrier 13

#define LLAMA_NUM_LAYERS 80
#define LLAMA_HIDDEN_DIM 8192
#define LLAMA_INTERMEDIATE_DIM 28672
#define LLAMA_HEAD_DIM 128
#define LLAMA_NUM_ATTENTION_HEADS 64
#define LLAMA_NUM_KV_HEADS 8
#define LLAMA_KV_PAGE_SIZE 128 /*128*/
#define LLAMA_PREFILL_KV_BLOCK_SIZE 128
#define LLAMA_DECODE_KV_BLOCK_SIZE 16
#define LLAMA_MATMUL_OUT_BLOCK_SIZE 256

#ifdef KITTENS_BLACKWELL
#define LLAMA_MATMUL_BATCH_BLOCK_SIZE 256
#else
#define LLAMA_MATMUL_BATCH_BLOCK_SIZE 128
#endif

#ifdef KITTENS_BLACKWELL
#define SM_COUNT 148
#else
#define SM_COUNT 132
#endif

struct base_llama_config {
    // Instruction pipeline
    static constexpr int INSTRUCTION_PIPELINE_STAGES = 2;

    // num bits required to represent num pipeline stages
    static constexpr int INSTRUCTION_PIPELINE_STAGES_BITS = 1;

    static constexpr int INSTRUCTION_WIDTH = 32;  // 128 bytes per instruction.
    using instruction_t = int[INSTRUCTION_WIDTH];

    // Timing info
    static constexpr int TIMING_WIDTH = 128;
    using timing_t = int[TIMING_WIDTH];

    static constexpr bool ENABLE_GLOBAL_WORK_QUEUE = true;

    // How many semaphores are available for dynamic use?
    static constexpr int DYNAMIC_SEMAPHORES = 128;

    // One controller warp, one load warp, one store warp, and one mma warp.
#ifdef KITTENS_BLACKWELL
    static constexpr int NUM_CONSUMER_WARPS = 16;
#else
    static constexpr int NUM_CONSUMER_WARPS = 8;
#endif
    static constexpr int NUM_WARPS = 4 + NUM_CONSUMER_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * ::kittens::WARP_THREADS;
    static constexpr int NUM_BLOCKS = 1;
    static constexpr int CLUSTER_BLOCKS = 1;
    static constexpr int MAX_SHARED_MEMORY = kittens::MAX_SHARED_MEMORY;

    // Shared memory declared statically
    static constexpr int SCRATCH_BYTES = 8192 + 2048;
    static constexpr int STATIC_SHARED_MEMORY =
        512 +
        INSTRUCTION_PIPELINE_STAGES * (SCRATCH_BYTES + (INSTRUCTION_WIDTH + TIMING_WIDTH) * 4 + DYNAMIC_SEMAPHORES * 8);
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    // Shared memory declared dynamically
    static constexpr int PAGE_SIZE = 32768;
    static constexpr int NUM_PAGES = DYNAMIC_SHARED_MEMORY / PAGE_SIZE;
    static_assert(NUM_PAGES == 6, "NUM_PAGES must be 6");

    static constexpr bool GMEM_SPIN_LOOP_SLEEP_NANOS = 20;

#ifdef KITTENS_BLACKWELL
    static constexpr int CONSUMER_REGISTERS = 104;
#else
    static constexpr int CONSUMER_REGISTERS = 208;
#endif
    static constexpr int NON_CONSUMER_REGISTERS = 64;
};
struct llama_config_timer : public base_llama_config {
    static constexpr bool TIMING_RECORD_ENABLED = true;
};
struct llama_config : public base_llama_config {
    static constexpr bool TIMING_RECORD_ENABLED = false;
};

template <typename config, int _num_hidden_layers, int _hidden_dim, int _intermediate_dim, int _head_dim,
          int _num_attention_heads, int _num_kv_heads, int _kv_page_size, int _prefill_kv_block_size,
          int _decode_kv_block_size, int _matmul_out_block_size, int _matmul_batch_block_size, int _sm_count>
struct globals_t {
    constexpr static int num_devices = 8;

    constexpr static int num_hidden_layers = _num_hidden_layers;
    constexpr static int matmul_out_block_size = _matmul_out_block_size;
    constexpr static int matmul_batch_block_size = _matmul_batch_block_size;
    constexpr static int kv_page_size = _kv_page_size;
    constexpr static int prefill_kv_block_size = _prefill_kv_block_size;
    constexpr static int decode_kv_block_size = _decode_kv_block_size;
    constexpr static int head_dim = _head_dim;
    constexpr static int hidden_dim = _hidden_dim;
    constexpr static int intermediate_dim = _intermediate_dim;
    constexpr static int num_attention_heads = _num_attention_heads;
    constexpr static int num_kv_heads = _num_kv_heads;
    constexpr static int sm_count = _sm_count;

    constexpr static int num_output_blocks = hidden_dim / matmul_out_block_size;

    using instruction_layout = megakernel::instruction_layout<config>;
    using timing_layout = megakernel::timing_layout<config>;

    using weights_t = kittens::gl<kittens::bf16, 1, -1, -1, hidden_dim, kittens::st_bf<256, 64>>;
    using weights_big_indim_t =
        kittens::gl<kittens::bf16, 1, -1, -1, intermediate_dim / num_devices, kittens::st_bf<256, 64>>;

    using activations_t = kittens::gl<kittens::bf16, 1, 1, -1, -1, kittens::sv_bf<hidden_dim>, kittens::st_bf<16, 128>,
                                      kittens::st_bf<64, 64>, kittens::sv_bf<head_dim>, kittens::st_bf<16, 64>>;

    using activations_parallel_t =
        kittens::pgl<kittens::gl<kittens::bf16, 1, 1, -1, hidden_dim, kittens::st_bf<64, 64>,
                                 kittens::sv_bf<hidden_dim>, kittens::st_bf<64, 256>, kittens::sv_bf<head_dim>>,
                     num_devices, false, false>;

    using activations_parallel_mc_t =
        kittens::pgl<kittens::gl<kittens::bf16, 1, 1, -1, hidden_dim, kittens::st_bf<64, 64>>, num_devices, true, true,
                     kittens::sv_bf<hidden_dim>>;

    using activations_big_indim_t =
        kittens::gl<kittens::bf16, 1, 1, -1, intermediate_dim / num_devices, kittens::st_bf<64, 256>,
                    kittens::st_bf<64, 64>, kittens::st_bf<16, 256>>;

    using logits_t = kittens::gl<kittens::bf16, 1, 1, -1, -1, kittens::st_bf<64, 256>>;

    using norm_weights_t = kittens::gl<kittens::bf16, 1, 1, -1, hidden_dim, kittens::sv_bf<hidden_dim>>;
    using rope_table_t = kittens::gl<float, 1, 1, -1, head_dim, kittens::sv_fl<head_dim>>;

    // KV Cache format: (num_layers * num_pages, page_size, num_heads, head_dim)
    using kv_cache_t = kittens::gl<kittens::bf16, -1, -1, num_kv_heads / num_devices, head_dim,
                                   kittens::tma::descriptor<kittens::st_bf<decode_kv_block_size, head_dim>, 1>,
                                   kittens::tma::descriptor<kittens::st_bf<prefill_kv_block_size, head_dim>, 1>,
                                   // kittens::tma::descriptor<kittens::st_bf<16, 128>, 0>,
                                   kittens::sv_bf<head_dim>>;

    using barriers = kittens::pgl<kittens::gl<uint, -1, -1, -1, -1>, num_devices,
                                  false>;  // no need to initialize multicast

    // vm stuff
    barriers Bar;
    instruction_layout instructions;
    timing_layout timings;
    kittens::gl<int, 1, 1, 1, 1> global_instruction_index;

    // model weights
    weights_t qkv_weights;
    norm_weights_t attn_norm_weights;
    weights_t o_weights;
    norm_weights_t mlp_norm_weights;

    weights_t up_weights;
    weights_t gate_weights;
    weights_big_indim_t down_weights;

    norm_weights_t lm_head_norm_weights;
    weights_t lm_head_weights;

    // kv cache
    kv_cache_t k_cache;
    kv_cache_t v_cache;

    // other buffers
    rope_table_t rope_cos;
    rope_table_t rope_sin;

    // activation buffers
    activations_parallel_t hidden_states;
    activations_parallel_mc_t rms_rope_intermediates;
    activations_parallel_mc_t rms_gate_intermediates;

    activations_t q_post_rope;
    activations_parallel_t attn_out;
    activations_big_indim_t silu_out;

#ifdef LLAMA_BROADCAST_LM_HEAD_NORM
    activations_parallel_mc_t rms_lm_head_intermediates;
#else
    activations_t rms_lm_head_intermediates;
#endif

    logits_t logits;

    using int32_vector_t = kittens::gl<int, 1, 1, 1, -1>;

    // unsigned int pos_id;
    int32_vector_t position_ids;
    int32_vector_t kv_append_indices;

    int32_vector_t prefill_qo_indptr;
    int32_vector_t prefill_kv_indptr;
    int32_vector_t prefill_kv_indices;
    int32_vector_t prefill_kv_last_page_len;
    int32_vector_t decode_kv_indptr;
    int32_vector_t decode_kv_indices;
    int32_vector_t decode_kv_last_page_len;

    float attn_scale;
    float rms_norm_eps;
    int num_pages;
    int batch_size;
    int num_prefill_tokens;

    int dev_idx;  // this is filled up on the C++ side (cf. pyutils.cuh)
    dim3 grid() { return dim3(sm_count); }
    dim3 block() { return dim3(llama_config::NUM_THREADS); }
    int dynamic_shared_memory() { return llama_config::DYNAMIC_SHARED_MEMORY; }
};

typedef globals_t<llama_config, LLAMA_NUM_LAYERS, LLAMA_HIDDEN_DIM, LLAMA_INTERMEDIATE_DIM, LLAMA_HEAD_DIM,
                  LLAMA_NUM_ATTENTION_HEADS, LLAMA_NUM_KV_HEADS, LLAMA_KV_PAGE_SIZE, LLAMA_PREFILL_KV_BLOCK_SIZE,
                  LLAMA_DECODE_KV_BLOCK_SIZE, LLAMA_MATMUL_OUT_BLOCK_SIZE, LLAMA_MATMUL_BATCH_BLOCK_SIZE, SM_COUNT>
    llama_70b_globals;

typedef globals_t<llama_config_timer, LLAMA_NUM_LAYERS, LLAMA_HIDDEN_DIM, LLAMA_INTERMEDIATE_DIM, LLAMA_HEAD_DIM,
                  LLAMA_NUM_ATTENTION_HEADS, LLAMA_NUM_KV_HEADS, LLAMA_KV_PAGE_SIZE, LLAMA_PREFILL_KV_BLOCK_SIZE,
                  LLAMA_DECODE_KV_BLOCK_SIZE, LLAMA_MATMUL_OUT_BLOCK_SIZE, LLAMA_MATMUL_BATCH_BLOCK_SIZE, SM_COUNT>
    llama_70b_globals_timer;


template <typename Globals>
__device__ inline int2 global_block_idx_to_local_block_info(const Globals& globs, int global_block_idx) {
    auto pos_in_block = global_block_idx % globs.matmul_batch_block_size;

    auto device_idx = global_block_idx % globs.num_devices;
    auto local_block_idx = global_block_idx / globs.num_devices;
    return {device_idx, local_block_idx};
}

template <typename Globals>
__device__ inline int2 global_batch_idx_to_local_idx_info(const Globals& globs, int global_batch_idx) {
    auto global_block_idx = global_batch_idx / globs.matmul_batch_block_size;
    auto pos_in_block = global_batch_idx % globs.matmul_batch_block_size;

    auto local_block_info = global_block_idx_to_local_block_info(globs, global_block_idx);

    auto device_idx = local_block_info.x;
    auto local_block_idx = local_block_info.y;
    auto local_batch_idx = local_block_idx * globs.matmul_batch_block_size + pos_in_block;
    return {device_idx, local_batch_idx};
}

template <typename Globals>
__device__ inline int local_batch_idx_to_global_batch_idx(const Globals& globs, int local_batch_idx) {
    auto local_block_idx = local_batch_idx / globs.matmul_batch_block_size;
    auto pos_in_block = local_batch_idx % globs.matmul_batch_block_size;
    auto global_block_idx = local_block_idx * globs.num_devices + globs.dev_idx;
    return global_block_idx * globs.matmul_batch_block_size + pos_in_block;
}

__device__ inline void fence_proxy_async_shared_cta() { asm volatile("fence.proxy.async.shared::cta;\n"); }

__device__ inline void trap() { asm volatile("trap;"); }

enum Scope {
    SYS, GPU
};
enum Sem {
    RELEASE, ACQUIRE, RELAXED, WEAK
};

template<Sem sem, Scope scope> __device__ __forceinline__ void fence() {
    static_assert(scope == Scope::SYS || scope == Scope::GPU, "Invalid fence scope");
    static_assert(sem == Sem::RELEASE || sem == Sem::ACQUIRE, "Invalid fence semantics");
    if constexpr (scope == Scope::SYS) {
        if constexpr (sem == Sem::RELEASE) {
            asm volatile("fence.acq_rel.sys;");
        }
        else if constexpr (sem == Sem::ACQUIRE) {
            asm volatile("fence.acquire.sys;");
            asm volatile("fence.proxy.async;");
        }
    } else if constexpr (scope == Scope::GPU) {
        if constexpr (sem == Sem::RELEASE) {
            asm volatile("fence.acq_rel.gpu;");
        }
        else if constexpr (sem == Sem::ACQUIRE) {
            asm volatile("fence.acquire.gpu;");
            asm volatile("fence.proxy.async;");
        }
    }
}

template<Sem sem, Scope scope> __device__ __forceinline__ void redAdd(uint32_t* addr, uint32_t val) {
    static_assert(scope == Scope::SYS || scope == Scope::GPU, "Invalid atomic scope");
    static_assert(sem == Sem::RELEASE || sem == Sem::RELAXED, "Invalid atomic semantics");
    if constexpr (scope == Scope::SYS) {
        if constexpr (sem == Sem::RELEASE) {
            asm volatile("red.release.sys.global.add.u32 [%0], %1;\n" :: "l"(addr), "r"(val) : "memory");
        } else if constexpr (sem == Sem::RELAXED) {
            asm volatile("red.relaxed.sys.global.add.u32 [%0], %1;\n" :: "l"(addr), "r"(val) : "memory");
        }
    } else if constexpr (scope == Scope::GPU) {
        if constexpr (sem == Sem::RELEASE) {
            asm volatile("red.release.gpu.global.add.u32 [%0], %1;\n" :: "l"(addr), "r"(val) : "memory");
        } else if constexpr (sem == Sem::RELAXED) {
            asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;\n" :: "l"(addr), "r"(val) : "memory");
        }
    }
}

template<Sem sem, Scope scope> __device__ __forceinline__ uint32_t strongLoad(const uint32_t* addr) {
    uint32_t val;
    static_assert(scope == Scope::SYS || scope == Scope::GPU, "Invalid strong load scope");
    static_assert(sem == Sem::ACQUIRE || sem == Sem::RELAXED, "Invalid strong load semantics");
    if constexpr (scope == Scope::SYS) {
        if constexpr (sem == Sem::ACQUIRE) {
            asm volatile("ld.acquire.sys.global.u32 %0, [%1];\n" : "=r"(val) : "l"(addr) : "memory");
        } else if constexpr (sem == Sem::RELAXED) {
            asm volatile("ld.relaxed.sys.global.u32 %0, [%1];\n" : "=r"(val) : "l"(addr) : "memory");
        }
    } else if constexpr (scope == Scope::GPU) {
        if constexpr (sem == Sem::ACQUIRE) {
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(val) : "l"(addr) : "memory");
        } else if constexpr (sem == Sem::RELAXED) {
            asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];\n" : "=r"(val) : "l"(addr) : "memory");
        }
    }
    return val;
}

// Timeout for spin loops - 1B cycles
static constexpr uint64_t SPIN_LOOP_TIMEOUT_CYCLES = 1000000000;

// Barrier satisfaction check
#ifdef LLAMA_STRICT_BARRIERS
static __device__ bool barrier_satisfied(int a, int b) { return a == b; }
#else
static __device__ bool barrier_satisfied(int a, int b) { return a >= b; }
#endif

// Simple barrier wait function without file/line info for now
template<Scope scope, typename... Args>
__device__ __forceinline__ void wait_on_barrier(
    uint32_t* barrier_addr, int expected_val, const char* name,
    const char* fmt, Args... args) {

#ifdef PRINT_DEADLOCKS
    uint64_t start_time = clock64();
#endif
    int current_val = strongLoad<Sem::ACQUIRE, scope>(barrier_addr);

    while (!barrier_satisfied(current_val, expected_val)) {
        __nanosleep(20);
#ifdef PRINT_DEADLOCKS
        uint64_t end_time = clock64();

        if (end_time - start_time > SPIN_LOOP_TIMEOUT_CYCLES) {
            if (kittens::laneid() == 0) {
                printf("[DEADLOCK] %s - ", name);
                printf(fmt, args...);
                printf(", current=%d, expected=%d\n", current_val, expected_val);
            }
            kittens::warp::sync();
            __nanosleep(1000000); // 1ms for flush
            kittens::warp::sync();
            asm volatile("trap;");
        }
#endif
        current_val = strongLoad<Sem::ACQUIRE, scope>(barrier_addr);
    }
}
