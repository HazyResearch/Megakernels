#pragma once

#include "kittens.cuh"

namespace megakittens {

// warp reduce
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = kittens::WARP_THREADS / 2; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

// uint64 -> 4 fp32s    
__device__ __forceinline__ void unpack_x4(uint64_t packed, float &a, float &b, float &c, float &d) {
    uint32_t lo = static_cast<uint32_t>(packed);
    uint32_t hi = static_cast<uint32_t>(packed >> 32);
    a = __bfloat162float(__ushort_as_bfloat16(static_cast<uint16_t>(lo)));
    b = __bfloat162float(__ushort_as_bfloat16(static_cast<uint16_t>(lo >> 16)));
    c = __bfloat162float(__ushort_as_bfloat16(static_cast<uint16_t>(hi)));
    d = __bfloat162float(__ushort_as_bfloat16(static_cast<uint16_t>(hi >> 16)));
}

// 4 fp32s -> 2 x bf16_2 -> uint64
__device__ __forceinline__ uint64_t pack_x4(float a, float b, float c, float d) {
    uint32_t lo = static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16(a)))
                | (static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16(b))) << 16);
    uint32_t hi = static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16(c)))
                | (static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16(d))) << 16);
    return static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
}

// ── RMSNorm instruction type ────────────────────────────────────────────────
//
// Pages are flat row buffers. One instruction processes ROWS_PER_INST full rows.
//
// instruction.indices layout:
//   [0] = starting row index
//   [1] = N (number of columns)
//   [2] = rows_this_instruction (may be < max for last chunk)
//
// SRC0 = x      (M, N) bf16
// SRC1 = weight (N,)   bf16
// DST  = output (M, N) bf16

template <typename Config, typename Globals, int SRC0, int SRC1, int DST>
struct RMSNorm {
    static constexpr int PAGE_BYTES = Config::PAGE_SIZE;
    static constexpr float EPS = 1e-6f;

    // Compute how many pages are needed for a given number of rows at width N
    __device__ static __forceinline__ int pages_needed(int num_rows, int N) {
        int total_bytes = num_rows * N * 2;
        return min((total_bytes + PAGE_BYTES - 1) / PAGE_BYTES, (int)Config::NUM_PAGES);
    }

    __device__ static __forceinline__ kittens::semaphore &inputs_arrived(state_t<Config> &s) { return s.semaphores()[0]; }
    __device__ static __forceinline__ kittens::semaphore &output_arrived(state_t<Config> &s) { return s.semaphores()[1]; }

    // ── helpers to treat pages as a flat byte buffer ────────────────────────
    // Map a flat byte offset into the page array to a device pointer.
    __device__ static __forceinline__ void *flat_ptr(state_t<Config> &s, int byte_offset) {
        const int page_idx = byte_offset / PAGE_BYTES;
        const int page_off = byte_offset % PAGE_BYTES;
        return s.pages[s.lid_to_pid(page_idx)].ptr(page_off);
    }

    // ── controller ─────────────────────────────────────────────────────────
    struct controller {
        __device__ __forceinline__ static int lid_release_order(const Globals &g, state_t<Config> &s, int lid) {
            const auto &instruction = s.instruction();
            int np = pages_needed(instruction.indices[2], instruction.indices[1]);
            return (lid + np) % Config::NUM_PAGES;
        }
        __device__ __forceinline__ static int init_semaphores(const Globals &g, state_t<Config> &s) {
            if (kittens::warp::elect_leader()) {
                kittens::init_semaphore(inputs_arrived(s), 1);  // loader signals when done
                kittens::init_semaphore(output_arrived(s), 1);  // consumer signals when done
            }
            return 2;
        }
    };

    // ── loader ─────────────────────────────────────────────────────────────
    // cp.async load rows_this_instruction × N bf16 values into pages (flat buffer).
    // All 32 threads in the loader warp participate for maximum bandwidth.
    struct loader {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            const auto &instruction = s.instruction();
            const int row_start = instruction.indices[0];
            const int N         = instruction.indices[1];
            const int num_rows  = instruction.indices[2];
            const int lane      = kittens::laneid();

            // Wait for pages and barriers (single-thread operations)
            const int num_pages = pages_needed(num_rows, N);
            if (kittens::warp::elect_leader()) {
                for (int i = 0; i < num_pages; i++)
                    s.page_wait(s.lid_to_pid(i));
                all_barrier_wait<Config>(g, instruction);
            }
            __syncwarp();

            // All 32 threads load in parallel
            const auto &x_gl = g.template gls<SRC0>();
            const kittens::bf16 *x_base = reinterpret_cast<const kittens::bf16 *>(x_gl.raw_ptr);

            // Total 16-byte chunks across all rows
            const int chunks_per_row = (N * 2) / 16;
            const int total_chunks = num_rows * chunks_per_row;

            // Each thread handles every 32nd chunk (stride by warp size)
            for (int chunk_idx = lane; chunk_idx < total_chunks; chunk_idx += kittens::WARP_THREADS) {
                const int row = chunk_idx / chunks_per_row;
                const int chunk = chunk_idx % chunks_per_row;

                const int row_byte_offset = row * N * 2;
                const int byte_off = row_byte_offset + chunk * 16;
                void *dst = flat_ptr(s, byte_off);

                const kittens::bf16 *src_row = x_base + (row_start + row) * static_cast<int64_t>(N);
                const void *src = reinterpret_cast<const void *>(
                    reinterpret_cast<const char *>(src_row) + chunk * 16);

                asm volatile(
                    "cp.async.cg.shared.global [%0], [%1], 16;"
                    :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
                       "l"(src)
                );
            }

            // Commit and wait for all cp.async to complete
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 0;");
            __syncwarp();

            if (kittens::warp::elect_leader()) {
                kittens::arrive(inputs_arrived(s));
            }
        }
    };

    // ── launcher ───────────────────────────────────────────────────────────
    struct launcher {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            s.tensor_wait();
            if (kittens::warp::elect_leader()) s.tensor_finish();
        }
    };

    // ── consumer ───────────────────────────────────────────────────────────
    // Row-partitioned: each warp owns ceil(num_rows / NUM_CONSUMER_WARPS) rows.
    // Pass 1: compute sum(x^2) per row → rstd
    // Pass 2: normalize x[i,j] * rstd[i] * weight[j] → output in pages
    struct consumer {
        using consumer_group = kittens::group<Config::NUM_CONSUMER_WARPS>;
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            kittens::wait(inputs_arrived(s), 0);

            const auto &instruction = s.instruction();
            const int N        = instruction.indices[1];
            const int num_rows = instruction.indices[2];
            const int lane     = kittens::laneid();
            const int warp_id  = kittens::warpid(); // consumer warps are 0..NUM_CONSUMER_WARPS-1

            // Get weight pointer: weight is (N,) bf16
            const auto &w_gl = g.template gls<SRC1>();
            const kittens::bf16 *weight_ptr = reinterpret_cast<const kittens::bf16 *>(w_gl.raw_ptr);

            // Row partitioning: each warp handles a contiguous chunk of rows
            const int rows_per_warp = (num_rows + Config::NUM_CONSUMER_WARPS - 1) / Config::NUM_CONSUMER_WARPS;
            const int my_row_start  = warp_id * rows_per_warp;
            const int my_row_end    = min(my_row_start + rows_per_warp, num_rows);

            // Elements per thread for column iteration
            // Each thread processes 4 bf16 elements at a time (one uint64 = 8 bytes)
            const int elems_per_iter = kittens::WARP_THREADS * 4; // 32 threads * 4 = 128 elements per warp iteration

            for (int row = my_row_start; row < my_row_end; row++) {
                const int row_byte_offset = row * N * 2;

                // ── Pass 1: Accumulate sum(x^2) ────────────────────────
                float sum_sq = 0.0f;
                for (int col_base = 0; col_base < N; col_base += elems_per_iter) {
                    const int my_col = col_base + lane * 4;
                    if (my_col + 3 < N) {
                        const int byte_off = row_byte_offset + my_col * 2;
                        uint64_t packed = *reinterpret_cast<const uint64_t *>(
                            flat_ptr(s, byte_off));
                        float a, b, c, d;
                        unpack_x4(packed, a, b, c, d);
                        sum_sq += a * a + b * b + c * c + d * d;
                    } else {
                        // Handle tail elements
                        for (int k = 0; k < 4 && my_col + k < N; k++) {
                            const int byte_off = row_byte_offset + (my_col + k) * 2;
                            kittens::bf16 val = *reinterpret_cast<const kittens::bf16 *>(
                                flat_ptr(s, byte_off));
                            float fval = __bfloat162float(val);
                            sum_sq += fval * fval;
                        }
                    }
                }
                sum_sq = warp_reduce_sum(sum_sq);
                float rstd = rsqrtf(sum_sq / static_cast<float>(N) + EPS);

                // ── Pass 2: Normalize and write back ───────────────────
                for (int col_base = 0; col_base < N; col_base += elems_per_iter) {
                    const int my_col = col_base + lane * 4;
                    if (my_col + 3 < N) {
                        const int byte_off = row_byte_offset + my_col * 2;
                        void *smem_addr = flat_ptr(s, byte_off);
                        uint64_t packed = *reinterpret_cast<const uint64_t *>(smem_addr);
                        float a, b, c, d;
                        unpack_x4(packed, a, b, c, d);

                        // Load weights
                        uint64_t w_packed = *reinterpret_cast<const uint64_t *>(
                            weight_ptr + my_col);
                        float wa, wb, wc, wd;
                        unpack_x4(w_packed, wa, wb, wc, wd);

                        a = a * rstd * wa;
                        b = b * rstd * wb;
                        c = c * rstd * wc;
                        d = d * rstd * wd;

                        *reinterpret_cast<uint64_t *>(smem_addr) = pack_x4(a, b, c, d);
                    } else {
                        for (int k = 0; k < 4 && my_col + k < N; k++) {
                            const int byte_off = row_byte_offset + (my_col + k) * 2;
                            void *smem_addr = flat_ptr(s, byte_off);
                            kittens::bf16 val = *reinterpret_cast<const kittens::bf16 *>(smem_addr);
                            float fval = __bfloat162float(val);
                            float w = __bfloat162float(*(weight_ptr + my_col + k));
                            fval = fval * rstd * w;
                            *reinterpret_cast<kittens::bf16 *>(smem_addr) = __float2bfloat16(fval);
                        }
                    }
                }
            }

            // Sync all consumer warps before signaling storer
            consumer_group::sync(1);

            if (consumer_group::elect_leader()) {
                kittens::arrive(output_arrived(s));
            }
        }
    };

    // ── storer ─────────────────────────────────────────────────────────────
    // Write normalized rows from pages back to global memory.
    // All 32 threads in the storer warp participate for maximum bandwidth.
    struct storer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            const auto &instruction = s.instruction();
            const int row_start = instruction.indices[0];
            const int N         = instruction.indices[1];
            const int num_rows  = instruction.indices[2];
            const int lane      = kittens::laneid();

            kittens::wait(output_arrived(s), 0);

            auto &y_gl = g.template gls<DST>();
            kittens::bf16 *y_base = reinterpret_cast<kittens::bf16 *>(y_gl.raw_ptr);

            // Total 16-byte chunks across all rows
            const int chunks_per_row = (N * 2) / 16;
            const int total_chunks = num_rows * chunks_per_row;

            // Each thread handles every 32nd chunk
            for (int chunk_idx = lane; chunk_idx < total_chunks; chunk_idx += kittens::WARP_THREADS) {
                const int row = chunk_idx / chunks_per_row;
                const int chunk = chunk_idx % chunks_per_row;

                const int row_byte_offset = row * N * 2;
                const int byte_off = row_byte_offset + chunk * 16;
                const void *src = flat_ptr(s, byte_off);

                kittens::bf16 *dst_row = y_base + (row_start + row) * static_cast<int64_t>(N);
                void *dst = reinterpret_cast<void *>(
                    reinterpret_cast<char *>(dst_row) + chunk * 16);

                *reinterpret_cast<uint4 *>(dst) = *reinterpret_cast<const uint4 *>(src);
            }

            __syncwarp();

            const int num_pages = pages_needed(num_rows, N);
            if (kittens::warp::elect_leader()) {
                for (int i = 0; i < num_pages; i++)
                    s.page_finish(s.lid_to_pid(i));

                all_barrier_arrive<Config>(g, instruction);
            }
        }
    };
};

} // namespace megakittens
